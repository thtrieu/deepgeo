"""
python t2t_trainer.py \
--model=graph_transformer \
--hparams_set=graph_transformer_base_local \
--problem=geo_upto5 \
--data_dir=data \
--warm_start_from=/Users/thtrieu/deepgeo/gs_ckpt/enc12dec12_lr0d05/model.ckpt-500000 \
--output_dir=local_ckpt \
--alsologtostderr

python t2t_trainer.py \
--model=graph_transformer \
--hparams_set=graph_transformer_base_local \
--problem=geo_upto_depth6 \
--data_dir=data6 \
--warm_start_from=/Users/thtrieu/deepgeo/gs_ckpt/enc12dec12_lr0d05/model.ckpt-500000 \
--output_dir=local_ckpt \
--alsologtostderr

python t2t_trainer.py \
--model=graph_transformer2 \
--hparams_set=graph_transformer2_base_local \
--problem=geo_upto_depth6 \
--data_dir=data6 \
--output_dir=local_ckpt \
--alsologtostderr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer

import model
import model2
import problem


if __name__ == '__main__':
  t2t_trainer.main(None)