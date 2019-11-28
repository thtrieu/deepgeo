"""
python t2t_datagen.py \
--problem=geo_upto5 \
--tmp_dir=data_small \
--data_dir=data \
--alsologtostderr

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_datagen

import problem
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)



if __name__ == '__main__':
  t2t_datagen.main(None)