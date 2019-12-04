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
import model

# pylint: disable=logging-format-interpolation


@registry.register_model
class GraphTransformer2(model.BaseModel):
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
      attention_mask = model.dec_to_bin_att_mask(features['attention_mask'])
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
          do_return_all_layers=True)

    sequence_output = all_encoder_layers[-1]  # [batch seq_len hid_size]
    cls_vector = sequence_output[:, 0:1, :]  # [batch 1 hid_size]

    return sequence_output, cls_vector

  def greedy_decode_8steps(self, 
                           cls_vector,  # batch, 1, hid_size
                           sequence_output):  # batch, seq_len, hid_size
    hparams = self.hparams

    # When features into self.body() doesn't have 'targets' and 'theorem'
    # then we are in predict/infer mode. Since there is only a small
    # number of unrolling steps for the output, (1 for predicting theorem
    # and another 7 for the theorem premise), we build a static graph
    # to do greedy decode.
    with tf.variable_scope('prediction', reuse=tf.AUTO_REUSE):
      theorem_logits = tf.keras.layers.Dense(  # [batch, 1, num_theorems]
          name='theorem',
          units=hparams.num_theorems,
          use_bias=True,
          kernel_initializer=modeling.create_initializer(
              hparams.initializer_range))(cls_vector)
      theorem = tf.argmax(  # [batch, 1]
          theorem_logits,   # [batch, 1, num_theorems]
          axis=-1, output_type=tf.int32)

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

    # Here we cache the activations during decoding.
    # for each layer of the decoding transformer, we store
    # a tensor of size [batch, current_length, hidden_dim]
    # at first current_length = 0:
    sequence_length = sequence_output.shape.as_list()[1]
    cached_layers = [
        sequence_output  # [batch, sequence_length, hid_size]
    ]

    # We also store all the premise prediction into a tensor
    # of shape [batch, current_length]
    premises = tf.zeros_like(cls_vector[:, :0, 0],  # [batch, 0]
                             dtype=tf.int32)

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
            cached_layers,
            sequence_length,
        )  # [batch, 1, hid_size]

      # After this step, all tensors in cached_layers 
      # increased 1 in length:
      assert cached_layers[0].shape.as_list()[1] == current_length + 1

      # Next the output vector is used to predict theorem
      # if we are at step 0, otherwise predict premise.
      with tf.variable_scope('prediction', reuse=tf.AUTO_REUSE):
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
        decoder_input = model.premise_gather_nd(sequence_output, premise)

    logits = dict(theorem=theorem,  # [batch, 1]
                  premises=premises)  # [batch, 7]
    losses = dict(training=tf.constant(0.0))
    return logits, losses

  def one_column_cached_transformer(self, 
                                    decoder_input, 
                                    cached_layers,
                                    sequence_length):
    hparams = self.hparams
    current_len = cached_layers[0].shape.as_list()[1]

    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
      # Add positional embedding of shape [1, hid_size]
      pos_embedding, _ = modeling.embedding_lookup(
          input_ids=tf.constant([current_len-sequence_length]),  # [1]
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
          num_hidden_layers=1,
          num_attention_heads=hparams.num_attention_heads,
          intermediate_size=hparams.intermediate_size,
          intermediate_act_fn=modeling.get_activation(hparams.hidden_act),
          hidden_dropout_prob=hparams.dropout_prob,
          attention_probs_dropout_prob=hparams.dropout_prob,
          initializer_range=hparams.initializer_range,
          do_return_all_layers=True)

      decoder_output = all_decoder_layers[-1]  # [batch, 1, hid_size]
    return decoder_output

  def body(self, features):
    hparams = self.hparams
    if not self.is_training:
      hparams.dropout_prob = 0.0

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
      sequence_output, cls_vector = self.build_encoder(features)

    if 'targets' not in features:
      assert self.hparams.dropout_prob == 0.0
      return self.greedy_decode_8steps(cls_vector, sequence_output)

    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
      with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
        premise = features['targets']  # [batch, premise_len=8] -bad naming:(
        # [batch, premise_len, hid_size]
        premise_vecs = model.premise_gather_nd(sequence_output, premise)

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
            theorem_vec, # [batch, 1, hid_size]
            premise_vecs[:, :-1, :]  # [batch, premise_len-1, hid_size]
        ], axis=1)  # [batch, premise_len, hid_size]
        decode_length = decoder_input.shape.as_list()[1]
        assert decode_length == premise_len

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

        sequence_len = sequence_output.shape.as_list()[1]
        causal_attention_mask = tf.concat([
            tf.ones([1, decode_length, sequence_len], dtype=tf.float32),
            causal_attention_mask
        ], axis=-1)  # [1, decode_length, decode_length + sequence_len]

        # [batch, decode_length, decode_length + sequence_len]
        causal_attention_mask = tf.tile(
            causal_attention_mask, [batch_size, 1, 1])

        all_decoder_layers = modeling.cached_transformer_model(
            input_vector=decoder_input,
            cached_layers=[sequence_output],
            attention_mask=causal_attention_mask,
            hidden_size=hparams.hidden_size,
            num_hidden_layers=1,
            num_attention_heads=hparams.num_attention_heads,
            intermediate_size=hparams.intermediate_size,
            intermediate_act_fn=modeling.get_activation(hparams.hidden_act),
            hidden_dropout_prob=hparams.dropout_prob,
            attention_probs_dropout_prob=hparams.dropout_prob,
            initializer_range=hparams.initializer_range,
            do_return_all_layers=True)

        premise_feature = all_decoder_layers[-1]  # [batch, dec_len, hid_size]

    with tf.variable_scope('prediction', reuse=tf.AUTO_REUSE):
      theorem_logits = tf.keras.layers.Dense(  # [batch, num_theorems]
          name='theorem',
          units=hparams.num_theorems,
          use_bias=True,
          kernel_initializer=modeling.create_initializer(
              hparams.initializer_range))(cls_vector[:, 0, :])

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


@registry.register_hparams
def graph_transformer2_base():
  return model.update_hparams(
      model.graph_transformer_base(),
      num_encode_layers=24,
  )


@registry.register_hparams
def graph_transformer2_base_local():
  return model.update_hparams(
      graph_transformer2_base(),
      batch_shuffle_size=8,
      batch_size=2,
  )