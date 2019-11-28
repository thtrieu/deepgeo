
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import tensorflow as tf
import os
import numpy as np
import time

import geometry
import theorems
from theorems_utils import State
from geometry import Point, Line, Segment, HalfPlane, Angle
from geometry import SegmentLength, AngleMeasure, LineDirection


def bin2dec_v3(bool_array):
  """Turn bool array a[N, n_bit] into signed integer.

  In other words, inverse of the following:
  tf.mod(
    tf.bitwise.right_shift(
        tf.expand_dims(a_dec, 1), 
        tf.range(n_bit, dtype=tf.int64)), 2)
  """
  bool_array = bool_array[:, ::-1]
  # sign = first bit AND (OR [other bits])
  sign = bool_array[:, :1] * np.logical_or.reduce(
      bool_array[:, 1:], -1, keepdims=True)

  bool_array = bool_array.astype(int)
  sign = sign.astype(int)
  
  bool_array = sign - bool_array  # flip bit if negative, negate if positive
  dec = bool_array.dot(1 << np.arange(bool_array.shape[-1] - 1, -1, -1))
  return -(dec + sign.flatten())  # negate back, add -1 if negative


def get_examples_from_depth(
      tmp_dir, depth, max_seq_len=128, max_target=8):
  files = tf.io.gfile.glob(
      os.path.join(tmp_dir, '*.depth.{:02}.*'.format(depth)))

  for count, f in enumerate(sorted(files)):
    start_time = time.time()
    with np.load(f) as loaded:
    # loaded = dict(np.load(f))
      for i in range(1000):  #np.random.permutation(1000):
        sequence = list(loaded['arr_' + str(i * 3)])
        attention_mask = loaded['arr_' + str(i * 3 + 1)]
        theorem_target = loaded['arr_' + str(i * 3 + 2)]
        theorem = theorem_target[0]
        target = list(theorem_target[1:])

        seq_len = len(sequence)
        num_pad = max_seq_len - seq_len

        attention_mask_full = np.zeros((max_seq_len, max_seq_len), dtype=bool)
        attention_mask_full[:seq_len, :seq_len] = attention_mask
        sequence += [0] * num_pad
        target += [0] * (max_target - len(target))

        assert max_seq_len % 32 == 0  # Turning into int32
        attention_mask = attention_mask_full.reshape([-1, 32])
        attention_mask = list(bin2dec_v3(attention_mask))
        # import pdb; pdb.set_trace()

        yield dict(sequence=sequence, 
                   attention_mask=attention_mask,
                   theorem=[theorem],
                   targets=target,
                   depth=[depth])

    tf.logging.info('Depth {}, file {}/{}: {} done in {}s'.format(
        depth, count+1, len(files), f, time.time() - start_time))


@registry.register_problem
class GeoUpto5(text_problems.Text2TextProblem):

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.stop_at_eos = int(True)
    p.modality = {'target': text_problems.modalities.ModalityType.SYMBOL}

  def get_hparams(self, model_hparams=None):
    """Returns problem_hparams."""
    if self._hparams is not None:
      return self._hparams

    if model_hparams is None:
      model_hparams = default_model_hparams()

    hp = problem._default_hparams()
    if self._was_reversed:
      problem._reverse_problem_hparams(hp)
    if self._was_copy:
      problem._copy_problem_hparams(hp)

    self._hparams = hp
    return self._hparams

  @property
  def max_seq_len(self):
    return 128

  @property
  def max_target(self):
    return 8

  @property
  def is_generate_per_split(self):
    return False

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    """A generator that generates samples that are encoded.
    Args:
      data_dir: data directory
      tmp_dir: temp directory
      dataset_split: dataset split
    Yields:
      A dict.
    """
    generators = [get_examples_from_depth(
                      tmp_dir, i, self.max_seq_len, self.max_target)
                  for i in [1, 2, 3, 4, 5]]

    for gen in generators:
      for sample in gen:
        yield sample

  def example_reading_spec(self):
    """Specify fields to read from TFrecords."""
    data_items_to_decoders = None
    data_fields = dict(
        sequence=tf.FixedLenFeature([self.max_seq_len], tf.int64),
        attention_mask=tf.FixedLenFeature([self.max_seq_len * self.max_seq_len // 32], 
                                          tf.int64),
        theorem=tf.FixedLenFeature([1], tf.int64),
        targets=tf.FixedLenFeature([self.max_target], tf.int64),
        depth=tf.FixedLenFeature([1], tf.int64)
    )
    return (data_fields, data_items_to_decoders)

  @property
  def batch_size_means_tokens(self):
    """Batch size does *not* mean token, obviously, duh."""
    return False