from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from tensor2tensor.layers import common_hparams
from tensor2tensor.models import resnet
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import tensorflow as tf

import modeling


def get_scaffold_fn(ckpt_path):
  variable_map = get_variable_map(ckpt_path)

  def scaffold_fn():
    tf.train.init_from_checkpoint(ckpt_path, variable_map)

  return scaffold_fn


def get_variable_map(ckpt_path):
  """Initialize variables from given directory."""

  def _get_body_name(var_name):
    if '/body/' not in var_name:
      return var_name
    body_name = var_name.split('/body/')
    assert len(body_name) == 2, var_name
    return body_name[-1]

  ckpt_vars = {}
  for var_name, shape in tf.train.list_variables(ckpt_path):
    ckpt_vars[_get_body_name(var_name)] = var_name, shape

  variable_map = {}
  fails = []
  for var in tf.contrib.framework.get_trainable_variables():
    model_var_name = var.op.name
    model_var_body_name = _get_body_name(model_var_name)
    if model_var_body_name in ckpt_vars:
      ckpt_var_name, shape = ckpt_vars[model_var_body_name]
      assert var.shape.as_list() == list(shape)
      tf.logging.info('>>> LOAD ckpt var {} to model var {}'.format(
          ckpt_var_name, model_var_name))
      variable_map[ckpt_var_name] = var
    else:
      fails.append(model_var_name)

  for model_var_name in fails:
    tf.logging.info('>>> FAIL to find {} in checkpoint'.format(
        model_var_name))

  if fails:
    raise ValueError('Failed to initialize from checkpoint.')

  return variable_map


def dec2bin32(dec):
  if dec.dtype is not tf.int32:
    raise ValueError('Only support int32.')
  return tf.mod(
    tf.bitwise.right_shift(
        tf.expand_dims(dec, -1), 
        tf.range(32, dtype=tf.int32)), 2)


def dec_to_bin_att_mask(dec, seq_len=128):
  return tf.reshape(dec2bin32(dec), [-1, seq_len, seq_len])



def premise_gather_nd(sequence_output, premise):
  """Gather nd.

  Example:
  sequence_output = array([[ 0,  1,  2,  3],
                           [ 4,  5,  6,  7],
                           [ 8,  9, 10, 11]])
  premise = [[0, 1], 
             [3, 1],
             [2, 3]]

  Return: array([[ 0,  1 ],
                 [ 7,  5 ],
                 [ 10, 11]])

  Args:
    sequence_output: float [batch, sequence_len, hid_size]
    premise: int [batch, premise_len]

  Returns:
    float [batch, premise_len, hid_size]
  """
  batch_size = tf.shape(premise)[0]
  batch_range = tf.range(batch_size)  # [batch_size]
  batch_range = tf.reshape(batch_range, [-1, 1])  # [batch_size, 1]
  
  premise_len = premise.shape.as_list()[-1]
  batch_range = tf.tile(batch_range, [1, premise_len])  # [batch_size, premise_len]
  premise_nd = tf.concat([
      tf.expand_dims(batch_range, -1),  # [batch, premise_len, 1]
      tf.expand_dims(premise, -1)  # [batch, premise_len, 1]
  ], axis=-1)  # [batch, premise_len, 2]

  # [batch, premise_len, hid_size]
  premise_vecs = tf.gather_nd(sequence_output,  # [batch, seq_len, hid_size]
                              premise_nd) # [batch, premise_len, 2]
  return premise_vecs



def accuracy(logits, labels):
  predictions = tf.argmax(  # shape [batch, num_classes]
      logits, axis=-1, output_type=tf.int32)
  return tf.metrics.accuracy(
      # Shape of model_outputs['labels'] is [batch_size, 1]
      # because T2T on TPU expects these tensors to have rank >= 2
      labels=tf.reshape(labels, [-1]),
      predictions=predictions)  # shape [batch, num_classes]


class BaseModel(t2t_model.T2TModel):
  """Base Image Model; subclass needs to implement body()."""

  def bottom(self, features):
    """Preprocess."""
    # reshape from [batch, 1, 1, 1] to [batch]
    if 'targets' not in features:
      return features

    targets = features['targets']
    n_target = targets.shape.as_list()[1]
    features.update(
        targets=tf.reshape(targets, [-1, n_target]),
    )
    return features

  def estimator_spec_eval(self, features, logits, labels, loss, losses_dict):
    """Constructs `tf.estimator.EstimatorSpec` for EVAL (evaluation) mode."""
    del losses_dict

    def eval_metrics_fn(theorem_logits, theorem_labels,
                        premise_logits, premise_labels):
      return dict(
          theorem_accuracy=accuracy(theorem_logits, theorem_labels),
          premise_accuracy=accuracy(premise_logits, premise_labels)
      )

    if t2t_model.common_layers.is_xla_compiled():
      # Note: important to call this before remove_summaries()
      if self.hparams.tpu_enable_host_call:
        host_call = self.create_eval_host_call()
      else:
        host_call = None
      t2t_model.remove_summaries()
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          eval_metrics=(eval_metrics_fn, logits),
          host_call=host_call,
          loss=loss)

    evaluation_hooks = []
    # Create a SummarySaverHook
    eval_dir = os.path.join(
        self.hparams.model_dir,
        self.hparams.get('eval_dir_name', 'eval'))
    eval_summary_hook = tf.train.SummarySaverHook(
        save_steps=1,
        output_dir=eval_dir,
        summary_op=tf.summary.merge_all())
    evaluation_hooks.append(eval_summary_hook)
    evaluation_hooks += self.hparams.problem.eval_hooks(
        features, logits, self.hparams)

    return tf.estimator.EstimatorSpec(
        tf.estimator.ModeKeys.EVAL,
        predictions=logits,
        eval_metric_ops=eval_metrics_fn(**logits),
        evaluation_hooks=evaluation_hooks,
        loss=loss)

  def optimize(self, loss, num_async_replicas=1, use_tpu=False, variables=None):
    """Return a training op minimizing loss."""
    lr = t2t_model.learning_rate.learning_rate_schedule(self.hparams)
    tf.summary.scalar('learning_rate', lr)
    if num_async_replicas > 1:
      log_info("Dividing learning rate by num_async_replicas: %d",
               num_async_replicas)
    lr /= math.sqrt(float(num_async_replicas))
    train_op = t2t_model.optimize.optimize(
        loss, lr, self.hparams, use_tpu=use_tpu, variables=variables)
    return train_op

  def initialize_from_ckpt(self, ckpt_path):
    get_scaffold_fn(ckpt_path)()


@registry.register_model
class GraphTransformer(BaseModel):
  """Conv and Att on sparser patches."""

  def build_encoder(self, features):
    hparams = self.hparams

    # Here we expect features to have 'sequence' and 'attention_mask'
    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
      # import pdb; pdb.set_trace()
      sequence = features['sequence']  # [batch, seq_len=128]
      # types of entity: Point, Line, Segment, Halfplane, etc.
      embedding_output, _ = modeling.embedding_lookup(
          input_ids=sequence,
          vocab_size=hparams.entity_num_type,
          embedding_size=hparams.hidden_size,
          initializer_range=hparams.initializer_range,
          word_embedding_name='entity_type_embedding',
      )  # [batch, seq_len, hid_size]

      # Next we add a "type" to indicate which
      # object in the sequence is of problem state, and
      # which is the goal object.
      encoder_input = modeling.embedding_postprocessor(
          input_tensor=embedding_output,
          sequence_ids=sequence,
          hparams=self.hparams
      )  # [batch, seq_len, hid_size]

    # Next we feed the sequence into encoder transformer
    # with the corresponding attention mask.
    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
      # [batch, seq_len, seq_len]
      attention_mask = dec_to_bin_att_mask(features['attention_mask'])
      all_encoder_layers = modeling.transformer_model(
          input_tensor=encoder_input,  # [batch, seq_len, hid_size]
          attention_mask=attention_mask,  # [batch, seq_len, seq_len]
          hidden_size=hparams.hidden_size,
          num_hidden_layers=hparams.num_encode_layers,
          num_attention_heads=hparams.num_attention_heads,
          intermediate_size=hparams.intermediate_size,
          intermediate_act_fn=modeling.get_activation(hparams.hidden_act),
          hidden_dropout_prob=hparams.dropout_prob,
          attention_probs_dropout_prob=hparams.dropout_prob,
          initializer_range=hparams.initializer_range,
          do_return_all_layers=True,
          attention_top_k=hparams.attention_top_k,
          densify_attention_mask=hparams.densify_attention_mask)

    sequence_output, attention_weights = all_encoder_layers[-1]  # [batch seq_len hid_size]
    cls_vector = sequence_output[:, 0:1, :]  # [batch 1 hid_size]

    return sequence_output, cls_vector, attention_weights

  def greedy_decode_8steps(self, 
                           cls_vector,  # batch, 1, hid_size
                           sequence_output):  # batch, seq_len, hid_size
    hparams = self.hparams

    # When features into self.body() doesn't have 'targets' and 'theorem'
    # then we are in predict/infer mode. Since there is only a small
    # number of unrolling steps for the output, (1 for predicting theorem
    # and another 7 for the theorem premise), we build a static graph
    # to do greedy decode.

    # Here we cache the activations during decoding.
    # for each layer of the decoding transformer, we store
    # a tensor of size [batch, current_length, hidden_dim]
    # at first current_length = 0:
    cached_layers = [
        tf.zeros_like(cls_vector[:, :0, :])  # [batch, 0, hid_size]
        for _ in range(hparams.num_decode_layers)
    ]

    # We also store all the premise prediction into a tensor
    # of shape [batch, current_length]
    premises = tf.zeros_like(cls_vector[:, :0, 0],  # [batch, 0]
                             dtype=tf.int32)

    # The first token to be processed is the CLS vector.
    decoder_input = cls_vector

    # Now we build the static unrolling of 8-step decoding,
    # each step update a new value for decoder_input
    for count in range(8):
      current_lengths = [layer.shape.as_list()[1]
                         for layer in cached_layers]
      assert current_lengths[1:] == current_lengths[:-1]
      current_length = current_lengths[0]
      with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        # cached_layers will be updated inside this method.
        # Feed this single token into the decoder transformer.
        output_vector = self.one_column_cached_transformer(
            decoder_input,  # batch, 1, hid_size
            # list of num_hid_layers tensors, each of shape
            # [batch, current_length, hidden_size]
            cached_layers
        )  # [batch, 1, hid_size]

      # After this step, all tensors in cached_layers 
      # increased 1 in length:
      assert cached_layers[0].shape.as_list()[1] == current_length + 1

      # Next the output vector is used to predict theorem
      # if we are at step 0, otherwise predict premise.
      with tf.variable_scope('prediction', reuse=tf.AUTO_REUSE):
        if count == 0:
          theorem_logits = tf.keras.layers.Dense(  # [batch, 1, num_theorems]
              name='theorem',
              units=hparams.num_theorems,
              use_bias=True,
              kernel_initializer=modeling.create_initializer(
                  hparams.initializer_range))(output_vector)
          theorem = tf.argmax(  # [batch, 1]
              theorem_logits,   # [batch, 1, num_theorems]
              axis=-1, output_type=tf.int32)
        else:
          premise_logits = tf.matmul(  # batch, 1, seq_len
              a=output_vector,  # [batch, 1, hid_size]
              b=sequence_output,  # [batch, sequence_len, hid_size]
              transpose_b=True,  
          )  # [batch, 1, sequence_len]
          premise = tf.argmax(  # [batch, 1]
              premise_logits,   # [batch, 1, seq_len]
              axis=-1, output_type=tf.int32)

          # [batch, current_len + 1]
          premises = tf.concat([premises, premise], axis=1)

          # [batch, 1, hid_size]
          decoder_input = premise_gather_nd(sequence_output, premise)
          continue

      # For theorem prediction, we need to go back to variable scope
      # decoder/embedding to get the new decoder_input
      with tf.variable_scope('decoder/embeddings', reuse=tf.AUTO_REUSE):
        # [batch, 1, hid_size] and [num_theorems, hid_size]
        # from the theorem_embedding lookup table.
        decoder_input, _ = modeling.embedding_lookup(
            input_ids=theorem,  # [batch, 1]
            vocab_size=hparams.num_theorems,
            embedding_size=hparams.hidden_size, 
            initializer_range=hparams.initializer_range,
            word_embedding_name='theorem_embedding',
        )

    logits = dict(theorem=theorem,  # [batch, 1]
                  premises=premises)  # [batch, 7]
    losses = dict(training=tf.constant(0.0))
    return logits, losses

  def one_column_cached_transformer(self, 
                                    decoder_input, 
                                    cached_layers):
    hparams = self.hparams
    current_len = cached_layers[0].shape.as_list()[1]

    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
      # Add positional embedding of shape [1, hid_size]
      pos_embedding, _ = modeling.embedding_lookup(
          input_ids=tf.constant([current_len]),  # [1]
          vocab_size=hparams.max_premise,  # >= premise_len
          embedding_size=hparams.hidden_size, 
          initializer_range=hparams.initializer_range,
          word_embedding_name='positional_embedding',
      )
      pos_embedding = tf.reshape(
          pos_embedding, [1, 1, hparams.hidden_size])

      decoder_input = modeling.layer_norm_and_dropout(
          decoder_input +  # [batch, 1, hid_size]
          pos_embedding,   # [1,     1, hid_size]
          hparams.dropout_prob)  # [batch, 1, hid_size]

    with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
      # In this decoding transformer layer, our tensor can
      # attend to everything computed so far, including itself
      # => attention mask of shape: [batch, 1, current_len + 1]
      batch_size = tf.shape(decoder_input)[0]
      causal_attention_mask = tf.ones([batch_size, 1, current_len+1])

      all_decoder_layers = modeling.cached_transformer_model(
          input_vector=decoder_input,
          cached_layers=cached_layers,
          attention_mask=causal_attention_mask,
          hidden_size=hparams.hidden_size,
          num_hidden_layers=hparams.num_decode_layers,
          num_attention_heads=hparams.num_attention_heads,
          intermediate_size=hparams.intermediate_size,
          intermediate_act_fn=modeling.get_activation(hparams.hidden_act),
          hidden_dropout_prob=hparams.dropout_prob,
          attention_probs_dropout_prob=hparams.dropout_prob,
          initializer_range=hparams.initializer_range,
          do_return_all_layers=True,
          attention_top_k=hparams.attention_top_k,
          densify_attention_mask=hparams.densify_attention_mask)

      decoder_output = all_decoder_layers[-1]  # [batch, 1, hid_size]
    return decoder_output

  def body(self, features):
    hparams = self.hparams
    if not self.is_training:
      hparams.dropout_prob = 0.0

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
      # attention_weights: [batch, n_head, from_len, to_len]
      sequence_output, cls_vector, attention_weights = self.build_encoder(features)

    if 'targets' not in features:
      assert self.hparams.dropout_prob == 0.0
      logits, losses = self.greedy_decode_8steps(cls_vector, sequence_output)
      logits.update(attention_weights=attention_weights[:, :, 0, :])
      return logits, losses

    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
      with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
        premise = features['targets']  # [batch, premise_len=8] -bad naming:(
        # [batch, premise_len, hid_size]
        premise_vecs = premise_gather_nd(sequence_output, premise)

        batch_size = tf.shape(premise)[0]
        premise_len = premise.shape.as_list()[-1]
        theorem = features['theorem']  # batch, 1

        # [batch, 1, hid_size] and [num_theorems, hid_size]
        theorem_vec, theorem_emb_table = modeling.embedding_lookup(
            input_ids=theorem,  # [batch, 1]
            vocab_size=hparams.num_theorems,
            embedding_size=hparams.hidden_size, 
            initializer_range=hparams.initializer_range,
            word_embedding_name='theorem_embedding',
        )
        depth = features['depth']  # batch, 1

        decoder_input = tf.concat([
            cls_vector,  # [batch, 1, hid_size]
            theorem_vec, # [batch, 1, hid_size]
            premise_vecs[:, :-1, :]  # [batch, premise_len-1, hid_size]
        ], axis=1)  # [batch, premise_len + 1, hid_size]
        decode_length = decoder_input.shape.as_list()[1]
        assert decode_length == premise_len + 1

        # [decode_length, hid_size]
        pos_embedding, _ = modeling.embedding_lookup(
            input_ids=tf.range(decode_length),  # [decode_length]
            vocab_size=hparams.max_premise,  # >= premise_len
            embedding_size=hparams.hidden_size, 
            initializer_range=hparams.initializer_range,
            word_embedding_name='positional_embedding',
        )
        pos_embedding = tf.reshape(
            pos_embedding, [1, decode_length, hparams.hidden_size])

        decoder_input = modeling.layer_norm_and_dropout(
            decoder_input +  # [batch, decode_length, hid_size]
            pos_embedding,   # [1,     decode_length, hid_size]
            hparams.dropout_prob)  # [batch, decode_length, hid_size]

      with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
        causal_attention_mask = t2t_model.common_layers.ones_matrix_band_part(
            rows=decode_length,
            cols=decode_length,
            num_lower=-1,  # attend to everything before
            num_upper=0,  # attend to nothing after
            out_shape=[1, decode_length, decode_length]
        )  # 1, decode_length, decode_length

        # [batch, decode_length, decode_length]
        causal_attention_mask = tf.tile(
            causal_attention_mask, [batch_size, 1, 1])

        all_decoder_layers = modeling.transformer_model(
            input_tensor=decoder_input,
            attention_mask=causal_attention_mask,
            hidden_size=hparams.hidden_size,
            num_hidden_layers=hparams.num_decode_layers,
            num_attention_heads=hparams.num_attention_heads,
            intermediate_size=hparams.intermediate_size,
            intermediate_act_fn=modeling.get_activation(hparams.hidden_act),
            hidden_dropout_prob=hparams.dropout_prob,
            attention_probs_dropout_prob=hparams.dropout_prob,
            initializer_range=hparams.initializer_range,
            do_return_all_layers=True,
            attention_top_k=hparams.attention_top_k)

        decoder_output, _ = all_decoder_layers[-1]  # [batch, dec_len, hid_size]
        theorem_feature = decoder_output[:, 0, :]  # [batch, hid_size]
        premise_feature = decoder_output[:, 1:, :]  # [batch, tar_len, hid_size]

    with tf.variable_scope('prediction', reuse=tf.AUTO_REUSE):
      theorem_logits = tf.keras.layers.Dense(  # [batch, num_theorems]
          name='theorem',
          units=hparams.num_theorems,
          use_bias=True,
          kernel_initializer=modeling.create_initializer(
              hparams.initializer_range))(theorem_feature)

      premise_logits = tf.matmul(
          a=premise_feature,  # [batch, premise_len, hid_size]
          b=sequence_output,  # [batch, sequence_len, hid_size]
          transpose_b=True,  
      )  # [batch, premise_len, sequence_len]

      # [batch * premise_len, sequence_len]
      seq_len = premise_logits.shape.as_list()[-1]
      premise_logits = tf.reshape(premise_logits, [-1, seq_len]) 

      premise_weights = tf.cast(premise > 0, tf.float32)  # [batch, prem_len]
      premise_weights = tf.reshape(premise_weights, [-1])  # [batch * prem_len]
      premise = tf.reshape(premise, [-1, 1])  # [batch * prem_len, 1]

      theorem_loss = tf.losses.sparse_softmax_cross_entropy(
          labels=theorem,  # [batch, 1]
          logits=theorem_logits  # [batch, num_theorems]
      )
      premise_loss = tf.losses.sparse_softmax_cross_entropy(
          labels=premise,  # [batch * premise_len, 1]
          logits=premise_logits,  # [batch * premise_len, sequence_len]
          weights=premise_weights  # [batch * premise_len]
      )

      logits = dict(theorem_logits=theorem_logits,
                    theorem_labels=theorem,
                    premise_logits=premise_logits,
                    premise_labels=premise)

      losses = dict(training=theorem_loss + premise_loss,
                    theorem_loss=theorem_loss,
                    premise_loss=premise_loss)

    return logits, losses


def update_hparams(original_hparams, **kwargs):
  """Update hparams with new key-values **kwargs."""
  original_values = original_hparams.values()
  original_values.update(**kwargs)
  return tf.contrib.training.HParams(**original_values)


@registry.register_hparams
def graph_transformer():
  return update_hparams(
      common_hparams.basic_params1(),
      learning_rate=0.1,

      dropout_prob=0.1,
      hidden_size=512,
      intermediate_size=1024,

      initializer_range=0.02,
      hidden_act='gelu',

      num_attention_heads=8,
      num_encode_layers=8,
      num_decode_layers=8,
      attention_top_k=-1,
      densify_attention_mask=False,

      entity_num_type=16,  # Point, Line, Segment, etc
      num_theorems=32,
      max_premise=16,
      state_vocab_size=4,  # state/goal etc
  )


@registry.register_hparams
def graph_transformer_base():
  return update_hparams(
      graph_transformer(),
      learning_rate=0.05,
      num_encode_layers=12,
      num_decode_layers=12,
  )



@registry.register_hparams
def graph_transformer_local():
  return update_hparams(
      graph_transformer(),
      batch_shuffle_size=8,
      batch_size=2,
  )


@registry.register_hparams
def graph_transformer_base_local():
  return update_hparams(
      graph_transformer_base(),
      batch_shuffle_size=8,
      batch_size=2,
  )