"""
python t2t_datagen.py \
--problem=geo_all20 \
--tmp_dir=data_np \
--data_dir=data_all20 \
--alsologtostderr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_datagen

import problem
import tensorflow as tf


if __name__ == '__main__':
  t2t_datagen.main(None)