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

# pylint: disable=logging-format-interpolation


def dec2bin32(dec):
  if dec.dtype is not tf.int32:
    raise ValueError('Only support int32.')
  return tf.mod(
    tf.bitwise.right_shift(
        tf.expand_dims(dec, -1), 
        tf.range(32, dtype=tf.int32)), 2)


def dec_to_att_mask(dec, seq_len=128):
  return tf.reshape(dec2bin32(dec), [-1, seq_len, seq_len])


class BaseModel(t2t_model.T2TModel):
  """Base Image Model; subclass needs to implement body()."""

  def bottom(self, features):
    """Preprocess."""
    # reshape from [batch, 1, 1, 1] to [batch]
    features.update(
        targets=tf.reshape(features['targets'], [-1]),
    )

    return features

  def get_eval_metrics(self, model_outputs):
    predictions = tf.argmax(  # shape [batch, num_classes]
        model_outputs['logits'], axis=-1, output_type=tf.int32)
    accuracy = tf.metrics.accuracy(
        # Shape of model_outputs['labels'] is [batch_size, 1]
        # because T2T on TPU expects these tensors to have rank >= 2
        labels=tf.reshape(model_outputs['labels'], [-1]),
        predictions=predictions)  # shape [batch, num_classes]
    eval_metrics = dict(accuracy=accuracy)
    return eval_metrics

  def estimator_spec_eval(self, features, logits, labels, loss, losses_dict):
    """Constructs `tf.estimator.EstimatorSpec` for EVAL (evaluation) mode."""
    del losses_dict
    # hparams = self.hparams

    assert not t2t_model.common_layers.is_xla_compiled(), (
        'Currently not supported eval on TPU.')
    # The returned logits dict from self.body() is what expected by
    # bert_pretraining.metric_fn.
    eval_metrics = self.get_eval_metrics(logits)
    predictions = logits

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
        predictions=predictions,
        eval_metric_ops=eval_metrics,
        evaluation_hooks=evaluation_hooks,
        loss=loss)

  # def optimize(self, loss, num_async_replicas=1, use_tpu=False):
  #   lr = t2t_model.learning_rate.learning_rate_schedule(self.hparams)
  #   if num_async_replicas > 1:
  #     lr /= math.sqrt(float(num_async_replicas))
  #   # if is_running_locally():
  #   #   lr = tf.Print(lr, [lr], message='learning rate')
  #   tf.logging.info('Use T2T resnet optimizer: lr={}'.format(lr))
  #   return t2t_model.optimize.optimize(
  #       loss, lr, self.hparams, use_tpu=use_tpu)


@registry.register_model
class GraphTransformer(BaseModel):
  """Conv and Att on sparser patches."""

  def body(self, features):
    hparams = self.hparams
    # hparams.attention_probs_dropout_prob = hparams.hidden_dropout_prob
    # if not self.is_training:
    #   hparams.hidden_dropout_prob = 0.0
    #   hparams.attention_probs_dropout_prob = 0.0

    self.features = features

    cls_embedding = tf.get_variable(
        name='cls_embedding',
        shape=[1, 1, hparams.hidden_size])

    loss = tf.reduce_sum(cls_embedding)

    sequence = features['sequence']
    targets = features['targets']
    attention_mask = features['attention_mask']
    theorem = features['theorem']
    depth = features['depth']

    attention_mask = dec_to_att_mask(attention_mask)
    loss = tf.Print(loss, [attention_mask,
                           targets,
                           sequence,
                           theorem,
                           depth])

    # loss = tf.reduce_mean(loss)
    # losses = dict(training=loss)
    # logits = dict(logits=tf.reshape(logits, [-1, logits.shape[1]]),
    #               # logits on TPU needs to be of rank > 1
    #               labels=tf.reshape(targets, [-1, 1]))

    logits = tf.one_hot(features['targets'], 128)
    logits = dict(logits=logits,
                  labels=features['targets'])
    losses = dict(training=loss)

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
      # use_presoftmax_ln=True,
      # learning_rate=4e-1,
      # train_softmax_only=False,
      # num_last_layer_to_use=1,
      # num_presoftmax_projections=0,
      # weight_decay=0.0,
      # training_subset_percentage=1.0,
      # max_num_conv_block=100,
      # resnet_hparams='resnet_50',
      # resnet_hparams_str='',
      # optimizer_use_t2t_resnet='resnet_50',
      # t2topt_hparams_str='',
      # learning_rate_warmup_steps=int(10e3),
      # cosine_warmup_steps=100
  )


@registry.register_hparams
def graph_transformer_local():
  return update_hparams(
      graph_transformer(),
      batch_shuffle_size=8
  )