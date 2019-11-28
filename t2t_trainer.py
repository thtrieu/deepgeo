"""
python t2t_trainer.py \
--model=graph_transformer \
--hparams_set=graph_transformer_local \
--problem=geo_upto5 \
--data_dir=data \
--output_dir=local_ckpt \
--alsologtostderr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer

import model
import problem


if __name__ == '__main__':
  t2t_trainer.main(None)