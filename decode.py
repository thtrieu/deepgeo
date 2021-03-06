r"""Decode (parallely and beam search).

Model trained upto 5 averaged
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/enc12dec12_lr0d05/model.ckpt-500000 \
--hparams=num_encode_layers=12,num_decode_layers=12 \
--problem=geo_upto5

Model trained upto 6, 300k steps not average

python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all6_300k/model.ckpt-300000 \
--problem=geo_all6

Model trained upto 6, 300k steps averaged

python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all6_300k/avg/model.ckpt-300000 \
--problem=geo_all6


Model trained upto 7, 250k steps not average

python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all7_250k/model.ckpt-250000 \
--problem=geo_all7

Model trained upto 7, 250k steps averaged

python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all7_250k/avg/model.ckpt-250000 \
--problem=geo_all7

Model trained upto 7, 350k steps not average

python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all7_350k/model.ckpt-350000 \
--problem=geo_all7

Model trained upto 7, 350k steps averaged
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all7_350k/avg/model.ckpt-350000 \
--problem=geo_all7

Model trained upto 12, 300k steps, not averaged.
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all12_300k/model.ckpt-300000 \
--problem=geo_all12

Model trained upto 12, 300k steps, averaged.
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all12_300k/avg/model.ckpt-300000 \
--problem=geo_all12


Model trained upto 20, 350k steps, NOT averaged.
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all20_350k/model.ckpt-350000 \
--problem=geo_all20

Model trained upto 20, 350k steps, averaged.
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/modelv1_all20_350k/avg/model.ckpt-350000 \
--problem=geo_all20


Model finetuned from trained upto 5 on depth6. =================================
python decode.py \
--alsologtostderr \
--model=graph_transformer \
--hparams_set=graph_transformer_base \
--data_dir=data6 \
--checkpoint_path=/Users/thtrieu/deepgeo/gs_ckpt/enc12dec12_depth6_lr0d01/model.ckpt-100000 \
--problem=geo_upto_depth6

python decode.py \
--alsologtostderr \
--model=graph_transformer2 \
--hparams_set=graph_transformer2_base \
--data_dir=data6 \
--checkpoint_path=/Users/thtrieu/deepgeo/local_ckpt/model.ckpt-0 \
--problem=geo_upto_depth6
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import getpass
import os
import readline  # pylint: disable=unused-import

from absl import flags
import numpy as np
from tensor2tensor.bin import t2t_decoder
import tensorflow as tf

from collections import defaultdict as ddict

import theorems
import problem
import model
import model2
import time
import action_chain_lib
import traceback
import data_gen_lib

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import PointCentersCircle
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint

try:
  from matplotlib import pyplot as plt
  from matplotlib import colors as mcolors
except:
  pass


flags.DEFINE_integer('max_seq_len', 128, '')
flags.DEFINE_bool('load_checkpoint', True, '')
FLAGS = flags.FLAGS




colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
color_names = ['blue', 'green', 'orange', 'indigo', 'firebrick', 'royalblue', 'lightseagreen', 'darkorchid']
COLORS = [colors[name] for name in color_names]


def get_model_fn(model_name, hparams, init_checkpoint):
  """Get model fn."""
  model_cls = t2t_decoder.registry.model(model_name)

  def model_fn(features, labels, mode, params=None, config=None):
    """Model fn."""
    _, _ = params, labels
    hparams_ = copy.deepcopy(hparams)

    # Instantiate model
    data_parallelism = None
    if not FLAGS.use_tpu and config:
      data_parallelism = config.data_parallelism
    reuse = tf.get_variable_scope().reuse
    this_model = model_cls(
        hparams_,
        # Always build model with EVAL mode to turn off all dropouts.
        tf.estimator.ModeKeys.EVAL,
        data_parallelism=data_parallelism,
        decode_hparams=None,
        _reuse=reuse)

    predictions, _ = this_model(features)

    scaffold_fn = (model.get_scaffold_fn(init_checkpoint)
                   if FLAGS.load_checkpoint else None)

    if mode == tf.estimator.ModeKeys.TRAIN:
      # Dummy spec, only for caching checkpoint purpose
      return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.0),
          train_op=tf.no_op())

    if FLAGS.use_tpu:
      predict_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=predictions,
          scaffold_fn=scaffold_fn)
    else:
      scaffold_fn()
      predict_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions)
    return predict_spec

  return model_fn


def create_estimator(model_name, hparams, init_checkpoint):
  """Create a T2T Estimator."""
  model_fn = get_model_fn(model_name, hparams, init_checkpoint)
  run_config = t2t_decoder.t2t_trainer.create_run_config(hparams)
  if FLAGS.use_tpu:
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=1,
        eval_batch_size=1,
        predict_batch_size=1
    )
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
    )

  return estimator


def estimator_input_fn(multi_step_features):
  """Turn beam features generator into Estimator input fn."""

  def proof_features_generator_wrapper():
    """Estimator expects no argument to the generator."""
    for features in multi_step_features:
      yield features

  def predict_input_fn(params):
    _ = params
    return tf.data.Dataset.from_generator(
        proof_features_generator_wrapper,
        output_types=dict(
            sequence=tf.int32,
            attention_mask=tf.int32),
        output_shapes=dict(
            sequence=[128],
            attention_mask=[512]
        )).batch(1)

  return predict_input_fn


ACTION_VOCAB = {
    'mid': 0,  # 0.000365972518921
    'mirror': 1,
    'seg_line': 2,
    'parallel': 3,
    'line': 4,
    'eq': 5,  # 1.73088312149
    'sas': 6,  # 0.251692056656
    'asa': 7,  # 2.26002907753 3.96637487411
    '.parallel': 8,
    '.parallel2': 9,
}

ACTION_VOCAB = {y: theorems.all_theorems[x] 
                for x, y in ACTION_VOCAB.items()}


class CLS(object):
  pass


VOCAB = {
    'PAD': 0,
    CLS: 1,
    Point: 2,
    Segment: 3,
    Line: 4,
    HalfPlane: 5,
    Angle: 6,
    Circle: 7,
    SegmentLength: 8,
    AngleMeasure: 9,
    LineDirection: 10
}


def execute_user_steps(steps, new_obj_names, state, canvas, verbose=False):
  action_chain = []

  for i, ((theorem, command), new_obj_name) in enumerate(zip(steps, new_obj_names)):
    name_maps = [c.split('=') for c in command.split()]

    mapping = dict(
        (theorem.names[a], action_chain_lib._find(state, b))
        if a in theorem.names
        else (action_chain_lib._find_premise(theorem.premise_objects, a), 
              action_chain_lib._find(state, b))
        for a, b in name_maps)
    action_gen = theorem.match_from_input_mapping(state, mapping, randomize=False)

    try:
      action = action_gen.next()
    except StopIteration:
      raise ValueError('Matching not found {} {}'.format(theorem, command))

    action.set_chain_position(i)
    action_chain.append(action)

    new_obj = action.theorem.for_drawing[0]
    state_obj = action.mapping[new_obj]
    state_obj.name = new_obj_name
    # print(i + 1, action.to_str())

    if verbose:
      print('\tAdd : {}'.format([obj.name for obj in action.new_objects]))
    state.add_relations(action.new_objects)
    line2pointgroups = action.draw(canvas)
    state.add_spatial_relations(line2pointgroups)


def get_or_add_line(state, canvas, line_name, p1_name, p2_name):
  p1 = state.name2obj[p1_name]
  p2 = state.name2obj[p2_name]

  line2points = ddict(lambda: [])
  for rel in state.type2rel[LineContainsPoint]:
    line, point = rel.init_list
    line2points[line].append(point)

  for line, points in line2points.items():
    if p1 in points and p2 in points:
      return line

  # print('Create new line {}'.format(line_name))
  line = Line(line_name)
  state.add_one(LineContainsPoint(line, p1))
  state.add_one(LineContainsPoint(line, p2))
  state.add_spatial_relations(canvas.add_line(line, p1, p2))
  return line


def get_or_add_segment(state, canvas, segment_name, p1_name, p2_name):
  line = get_or_add_line(
      state, canvas, segment_name.lower(), p1_name, p2_name)
  p1 = state.name2obj[p1_name]
  p2 = state.name2obj[p2_name]

  segment2points = ddict(lambda: [])
  for rel in state.type2rel[PointEndsSegment]:
    point, segment = rel.init_list
    segment2points[segment].append(point)

  for segment, points in segment2points.items():
    if p1 in points and p2 in points:
      return line, segment

  # print('Create new segment {}'.format(segment_name))
  segment = Segment(segment_name)
  state.add_one(PointEndsSegment(p1, segment))
  state.add_one(PointEndsSegment(p2, segment))
  return line, segment


def get_or_add_angle(state, canvas, angle_name, head, leg1, leg2):
  head = state.name2obj[head]
  leg1 = state.name2obj[leg1]
  leg2 = state.name2obj[leg2]

  seg1_name = head + leg1
  seg2_name = head + leg2
  line1, _ = get_or_add_segment(state, canvas, seg1_name, head, leg1)
  line2, _ = get_or_add_segment(state, canvas, seg2_name, head, leg2)

  angle2hps = ddict(lambda: [])
  for rel in state.type2rel[HalfplaneCoversAngle]:
    hp, angle = rel.init_list
    angle2hps[angle].append(hp)

  hp11, hp12 = state.line2hps[line1]
  hp21, hp22 = state.line2hps[line2]
  hp1 = hp11 if leg2 in state.hp2points[hp11] else hp12
  hp2 = hp21 if leg1 in state.hp2points[hp21] else hp22

  for angle, hps in angle2hps.items():
    if hp1 in hps and hp2 in hps:
      return angle

  # print('Create new angle {}'.format(angle_name))
  angle = Angle(angle_name)
  state.add_one(HalfplaneCoversAngle(hp1, angle))
  state.add_one(HalfplaneCoversAngle(hp2, angle))
  return angle


def goal_obj_name_to_obj(state, canvas, name):
  if name == name.lower():
    # obj is Line
    assert len(name) == 2
    p1, p2 = name
    line, _ = get_or_add_segment(
        state, canvas, name.upper(), p1.upper(), p2.upper())
    return line

  elif len(name) == 2:
    # obj is segment
    p1, p2 = name
    _, segment = get_or_add_segment(
        state, canvas, name, p1, p2)
    return segment

  elif len(name == 3):
    # obj is angle
    leg1, head, leg2 = name
    return get_or_add_angle(
        state, canvas, name, head, leg1, leg2)

  else:
    raise ValueError('Cannot infer type of "{}"'.format(name))


def plt_show_canvas_chain(canvas_chain, state, size=5):
  num_canvas = len(canvas_chain)
  fig, axes = plt.subplots(nrows=num_canvas, ncols=3,
                           figsize=(3 * size, 
                                    num_canvas * size))
  fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(size, size))

  file_name = raw_input('Save sketch to file name: ')
  if not file_name:
    return

  if not os.path.exists(file_name):
    os.makedirs(file_name)
  print('Saving to {0}/{0}.png'.format(file_name))
  fig.tight_layout()

  for step, axes_row in enumerate(axes):
    canvas, old_state, action, attention_highlights, premise_highlights = canvas_chain.pop(0)

    state_highlights = []

    if old_state is not None:
      colors = list(COLORS)
      for val, rels in old_state.val2valrel.items():
        if isinstance(val, LineDirection):
          continue
        color = colors.pop(0)
        if colors == []:
          colors = list(COLORS)

        objs = [rel.init_list[0] for rel in rels]
        state_highlights.extend(
            [(obj, color, 0.5) for obj in objs])

    highlights = (state_highlights, attention_highlights, premise_highlights)
    for i, ax in enumerate(axes_row):
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      ax2.clear()
      ax2.set_xlim(-2, 4)
      ax2.set_ylim(-2, 4)
      ax2.get_xaxis().set_visible(False)
      ax2.get_yaxis().set_visible(False)

      highlight = highlights[i]
      if highlight is None:
        continue

      if i == 1:
        canvas.plt_show(ax, state, highlights[0], mark_segment=True)
        canvas.plt_show(ax2, state, highlights[0], mark_segment=True)

      canvas.plt_show(ax, state, highlight, mark_segment=(i==0))
      canvas.plt_show(ax2, state, highlight, mark_segment=(i==0))

      if i == 0:
        title = 'Problem Figure:'
      elif i == 1:
        title = 'Attending ..'
      elif i == 2:
        title = action.theorem.name

      ax.set_title(title, fontweight='bold')
      ax2.set_title(title, fontweight='bold')
      fig2.savefig('{}/{:02}_{}.png'.format(file_name, step, i), dpi=500)

  fig.savefig('{0}/{0}.png'.format(file_name), dpi=500)
  plt.clf()



# def plt_show_canvas_chain(canvas_chain, state, size=5):
#   num_canvas = len(canvas_chain)

#   for canvas, attention, premise in canvas_chain:
#     fig, axes = plt.subplots(nrows=1, ncols=3,
#                              figsize=(3 * size, 
#                                       num_canvas * size))

#     canvas, attention_highlights, premise_highlights = canvas_chain.pop(0)
#     highlights = ([], attention_highlights, premise_highlights)
#     for i, ax in enumerate(axes_row):
#       ax.get_xaxis().set_visible(False)
#       ax.get_yaxis().set_visible(False)
#       highlight = highlights[i]

#       if highlight is None:
#         continue
#       canvas.plt_show(ax, state, highlight)

#     file_name = raw_input('Save sketch to file name: ')
#     if file_name:
#       print('Saving to {}.png'.format(file_name))
#       fig.tight_layout()
#       plt.savefig('{}.png'.format(file_name), dpi=500)
#     plt.clf()


class StepLoop(object):

  def __init__(self):
    self.pdb_dead = False
    self.canvas_chain = []  # for illustration

  def init_with_user_action_steps(self, action_steps, new_obj_names, goal_objs):
    state, canvas, _ = action_chain_lib.init_by_normal_triangle()
    execute_user_steps(
        action_steps, new_obj_names, state, canvas)

    self.canvas_chain = [(canvas, state, None, None, None)]
    self.state = state.copy()

    obj1_name, obj2_name = goal_objs
    goal_obj1 = goal_obj_name_to_obj(
        self.state, canvas, obj1_name)
    goal_obj2 = goal_obj_name_to_obj(
        self.state, canvas, obj2_name)
    assert isinstance(goal_obj1, type(goal_obj2))

    if isinstance(goal_obj1, Segment):
      val = SegmentLength('goal_length')
    elif isinstance(goal_obj1, Angle):
      val = AngleMeasure('goal_measure')
    elif isinstance(goal_obj1, Line):
      val = LineDirection('goal_direction')

    self.goal_objects = (val, goal_obj1, goal_obj2)

  def make_features_from_state_and_goal(self):
    seq, obj_list, obj2idx, attention_mask = data_gen_lib.serialize_state(self.state)
    self.state_object_list = obj_list

    val, obj1, obj2 = self.goal_objects
    val_idx = len(seq)

    global VOCAB
    seq.append(VOCAB[type(val)])
    obj_list.append(val)

    for obj in [obj1, obj2]:
      obj_idx = obj2idx[obj]
      attention_mask[val_idx, obj_idx] = True
      attention_mask[obj_idx, val_idx] = True

    seq_len = len(seq)
    num_pad = FLAGS.max_seq_len - seq_len

    attention_mask_full = np.zeros(
        (FLAGS.max_seq_len, FLAGS.max_seq_len), dtype=bool)
    attention_mask_full[:seq_len, :seq_len] = attention_mask
    seq += [0] * num_pad

    assert FLAGS.max_seq_len % 32 == 0  # Turning into int32
    attention_mask = attention_mask_full.reshape([-1, 32])
    attention_mask = list(problem.bin2dec_v3(attention_mask))

    return dict(sequence=seq, 
                attention_mask=attention_mask)

  def found_goal(self):
    _, goal_obj1, goal_obj2 = self.goal_objects
    if goal_obj1 not in self.state.obj2valrel:
      return False
    if goal_obj2 not in self.state.obj2valrel:
      return False
    val1 = self.state.obj2valrel[goal_obj1].init_list[1]
    val2 = self.state.obj2valrel[goal_obj2].init_list[1]
    return val1 == val2

  def multi_step_generator(self, user_action_steps_generator):
    for steps, obj_names, goal_objs in user_action_steps_generator:

      try:
        self.init_with_user_action_steps(steps, obj_names, goal_objs)
      except Exception as e:
        print(e)
        traceback.print_exc()
        continue

      print('\n Working on it ..')
      while True:
        yield self.make_features_from_state_and_goal()

        if self.found_goal():
          print('Found goal!')
          plt_show_canvas_chain(self.canvas_chain, self.state)
          break

        if self.pdb_dead:
          self.pdb_dead = False
          break

  def add_prediction_to_state(self,
                              theorem,  # [1]
                              premises,
                              attention_weights): # [7]
    global ACTION_VOCAB
    theorem = ACTION_VOCAB[theorem[0]]
    premise_names = sorted(theorem.names)
    premise_obj = [theorem.names[name] for name in premise_names]
    n_obj = len(premise_obj)

    state_obj = [self.state_object_list[idx] 
                 for idx in premises[:n_obj]]

    mapping = dict(zip(premise_obj, state_obj))
    action_gen = theorem.match_from_input_mapping(self.state, mapping)
    try:
      action = action_gen.next()
      print('Applied {}'.format(action.to_str()))
    except StopIteration:
      action = None

      for x in mapping:
        mapping_ = dict(mapping)
        mapping_.pop(x)
        action_gen = theorem.match_from_input_mapping(self.state, mapping_)
        try:
          action = action_gen.next()
          print(' * Applied {}'.format(action.to_str()))
          # break
        except:
          continue

      if action is None:
        match = [(x.name.split('_')[0], y.name) for x, y in mapping.items()]
        match = ' '.join(['{}={}'.format(x, y) for x, y in sorted(match)])
        print('Wrong: {} {}'.format(theorem.name, match))
        import pdb; pdb.set_trace()
        self.pdb_dead = True
        return

    attention_weights = summarize_heads(attention_weights)

    attention_highlights = [
        (obj, 'red', attention_weight)
        for obj, attention_weight in
        zip(self.state_object_list, attention_weights)[1:]]

    if isinstance(action.theorem, (theorems.ASA, 
                                   theorems.SAS, 
                                   theorems.ParallelBecauseInteriorAngles)):
      premise_val2objs = ddict(lambda: [])
      for rel in action.premise_objects:
        if not isinstance(rel, (LineHasDirection, SegmentHasLength, AngleHasMeasure)):
          continue
        obj, val = rel.init_list
        premise_val2objs[val].append(obj)

      premise_highlights = []
      colors = list(COLORS)
      for val, objs in premise_val2objs.items():
        color = colors.pop(0)
        for obj in objs:
          premise_highlights.append((obj, color, 0.5))
    else:
      premise_highlights = [
          (obj, 'green', 0.5)
          for obj in state_obj
      ]

    old_canvas, old_state = self.canvas_chain[-1][:2]
    self.canvas_chain[-1] = (
        old_canvas, old_state, action, attention_highlights, premise_highlights)

    new_canvas = old_canvas.copy()
    self.state.add_relations(action.new_objects)
    self.state.add_spatial_relations(action.draw(new_canvas))
    self.canvas_chain.append((new_canvas, self.state, None, None, None))
    self.state = self.state.copy()

  def predict(self, estimator, user_action_steps_generator):
    """Beam search."""
    multi_step_features = self.multi_step_generator(
        user_action_steps_generator)

    predictions = estimator.predict(
        input_fn=estimator_input_fn(multi_step_features),
        yield_single_examples=True)

    for prediction in predictions:
      self.add_prediction_to_state(**prediction)


def summarize_heads(attention_weights):  # [num_head, sequence_length]
  entropy = - (attention_weights * np.log(attention_weights + 1e-12)).sum(-1)
  weights = attention_weights[np.argmin(entropy), :]
  # weights = attention_weights.sum(0)
  maxw, minw = np.max(weights), np.min(weights)
  weights = (weights - minw) / (maxw - minw)
  # import pdb; pdb.set_trace()
  return weights * 0.5


def convert_text_inputs_to_action_steps(text_input):
  text_input = text_input.replace(' ', '')
  sentences = text_input.split('.')
  steps = []
  obj_names = []
  for sent in sentences[:-1]:
    obj_name, rhs = sent.split('=', 1)
    obj_names.append(obj_name)
    theorem_name, args = rhs.split(':')
    theorem = theorems.all_theorems[theorem_name]
    steps.append((theorem, args.replace(',', ' ')))

  goal = sentences[-1].replace(' ', '')
  obj1_name, obj2_name = goal.split('=')
  goal_objs = (obj1_name, obj2_name)

  return steps, obj_names, goal_objs


def interactive_text_inputs():
  while True:
    # Parrallelogram
    # l1=parallel: A=A, l=bc. l2=parallel: A=C, l=ab. D=line_line:l1=l1,l2=l2. DA=BC
    # l1=parallel: A=B, l=ca. l2=parallel: A=C, l=ab. D=line_line:l1=l1,l2=l2. DB=CA
    # l1=parallel: A=B, l=ca. l2=parallel: A=C, l=ab. D=line_line:l1=l1,l2=l2. DC=AB

    # midpoints:
    # M=mid:A=A,B=C. l=line:A=B,B=M. l1=parallel:A=A,l=l. l2=parallel:A=B,l=ca. D=line_line:l1=l1,l2=l2. DB=MC
    # M=mid:A=A,B=C. l=line:A=B,B=M. l1=parallel:A=A,l=l. l2=parallel:A=B,l=ca. D=line_line:l1=l1,l2=l2. DA=BM
    # M=mid:A=A,B=C. l=line:A=B,B=M. l1=parallel:A=A,l=l. l2=parallel:A=B,l=ca. D=line_line:l1=l1,l2=l2. DM=BC
    # lc=parallel:A=C,l=ab. la=parallel:A=A,l=bc. D=mid:A=B,B=C. E=mirror:A=A,B=B. F=line_line:l1=la,l2=lc. DF=DE
    # lc=parallel:A=C,l=ab. la=parallel:A=A,l=bc. E=mirror:A=A,B=B. F=line_line:l1=la,l2=lc. bf=ec

    # parallel:
    # M=mid:A=A,B=B. lm=parallel:A=M,l=ca. N=mid:A=A,C=C. ln=parallel:A=N,l=ab. P=line_line:l1=lm,l2=ln. pb=mn
    # M=mid:A=A,B=B. lm=parallel:A=M,l=ca. N=mid:A=A,C=C. ln=parallel:A=N,l=ab. P=line_line:l1=lm,l2=ln. PC=MN

    # center of mass:
    # M=mid:A=A,B=B. N=mid:A=A,B=C. l1=line:A=B,B=N. l2=line:A=C,B=M. D=line_line:l1=l1,l2=l2. l3=line:A=A,B=D. P=seg_line:l=l3,A=B,B=C. PB=PC

    # Thales:
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. AN=CN

    # Thales hint:
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. P=mirror:A=N,B=M. AN=CN
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. l1=parallel:A=C,l=ab. AN=CN
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. l1=parallel:A=N,l=ab. AN=CN
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. l1=parallel:A=B,l=ca. AN=CN

    # with all12
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. l1=line:A=M,B=C. AN=CN
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. l1=line:A=B,B=N. AN=CN
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. P=mid:A=B,B=C. AN=CN

    # Thales hint fail:
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. l1=parallel:A=M,l=ca. AN=CN
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. l1=line:A=B,B=N. AN=CN
    # M=mid:A=A,B=B. l=parallel:A=M,l=bc. N=seg_line:l=l,A=A,B=C. l1=parallel:A=B,l=ca. AN=CN

    user_input = raw_input('\n>>> Given triangle ABC. ')
    if user_input == 'q':
      break
    yield user_input


def main(_):
  t2t_decoder.trainer_lib.set_random_seed(FLAGS.random_seed)
  t2t_decoder.usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  hparams = t2t_decoder.create_hparams()
  t2t_decoder.trainer_lib.add_problem_hparams(hparams, FLAGS.problem)

  # latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

  estimator = create_estimator(
      FLAGS.model,
      hparams,
      FLAGS.checkpoint_path
  )

  def user_action_steps_generator():
    for text_input in interactive_text_inputs():
      try:
        yield convert_text_inputs_to_action_steps(text_input)
      except Exception as e:
        print(e)
        traceback.print_exc()
        continue

  StepLoop().predict(estimator, user_action_steps_generator())


if __name__ == '__main__':
  tf.app.run()