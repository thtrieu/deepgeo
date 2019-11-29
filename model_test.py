from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import tensorflow as tf

import modeling
import model
import numpy as np


def test_premise_gather_nd():
  sequence_output = np.array(
      [[[ 0,  1], [ 2,  3], [ 4,  5], [ 6,  7]],

       [[ 8,  9], [10, 11], [12, 13], [14, 15]],

       [[16, 17], [18, 19], [20, 21], [22, 23]]])

  premise = [[0, 1], 
             [3, 1],
             [2, 3]]

  sequence_output = tf.constant(sequence_output)
  premise = tf.constant(premise)

  premise_nd = model.premise_gather_nd(sequence_output, premise)

  with tf.Session() as sess:
    result = sess.run(premise_nd)
    expected_result = np.array(
        [[[ 0,  1], [ 2,  3]],

         [[14, 15], [10, 11]],

         [[20, 21], [22, 23]]]
    )
    assert np.array_equal(result, expected_result)


if __name__ == '__main__':
  test_premise_gather_nd()
  print('OK')