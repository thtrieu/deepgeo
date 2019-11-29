"""
python problem_test.py \
--model=graph_transformer \
--problem=geo_upto5 \
--hparams_set=graph_transformer_local \
--data_dir=data
"""
import problem
import model
import tensorflow as tf
import geometry
import theorems
import numpy as np

from tensor2tensor.bin import t2t_trainer

from theorems_utils import State
from geometry import Point, Line, Segment, HalfPlane, Angle
from geometry import SegmentLength, AngleMeasure, LineDirection

relations = {
    (Point, Segment): geometry.PointEndsSegment,
    (Line, Point): geometry.LineContainsPoint,
    (HalfPlane, Point): geometry.HalfPlaneContainsPoint,
    (Line, HalfPlane): geometry.LineBordersHalfplane,
    (HalfPlane, Angle): geometry.HalfplaneCoversAngle,
    (Segment, SegmentLength): geometry.SegmentHasLength,
    (Line, LineDirection): geometry.LineHasDirection,
    (Angle, AngleMeasure): geometry.AngleHasMeasure
}

action_vocab = {
    theorems.ConstructMidPoint: 0,  # 0.000365972518921
    theorems.ConstructMirrorPoint: 1,
    theorems.ConstructIntersectSegmentLine: 2,
    theorems.ConstructParallelLine: 3,
    theorems.ConstructThirdLine: 4,
    theorems.EqualAnglesBecauseParallel: 5,  # 1.73088312149
    theorems.SAS: 6,  # 0.251692056656
    theorems.ASA: 7,  # 2.26002907753 3.96637487411
    theorems.ParallelBecauseCorrespondingAngles: 8,
    theorems.ParallelBecauseInteriorAngles: 9,
}

action_vocab = {y: x() for x, y in action_vocab.items()}


class CLS(object):
  pass


vocab = {
    'PAD': 0,
    CLS: 1,
    geometry.Point: 2,
    geometry.Segment: 3,
    geometry.Line: 4,
    geometry.HalfPlane: 5,
    geometry.Angle: 6,
    geometry.Circle: 7,
    geometry.SegmentLength: 8,
    geometry.AngleMeasure: 9,
    geometry.LineDirection: 10
}

vocab = {y: x for x, y in vocab.items()}


def check(sequence, attention_mask, theorem, target, depth):
  for name, y in target.items():
    assert isinstance(theorem.names[name], sequence[y])
  
  assert np.array_equal(attention_mask, attention_mask.T)
  sequence = [t() for t in sequence if t != 'PAD' ]

  assert len(sequence == sum(attention_mask[0]))
  attention_mask = attention_mask[:len(sequence), :len(sequence)]

  frac = np.sum(attention_mask) * 1.0 / np.sum(np.ones_like(attention_mask))
  print(theorem.name, sorted(theorem.names.keys()), frac, depth)

  state = State()
  for i in range(1, len(sequence)-2):
    for j in range(i+1, len(sequence)-1):
      if not attention_mask[i, j]:
        continue
      obj1, obj2 = sequence[i], sequence[j]
      t1, t2 = type(obj1), type(obj2)
      if (t1, t2) in relations:
        state.add_one(relations[(t1, t2)](obj1, obj2))
      else:
        state.add_one(relations[(t2, t1)](obj2, obj1))
  
  mapping = {theorem.names[name]: sequence[y]
             for name, y in target.items()}
  action_gen = theorem.match_from_input_mapping(state, mapping)
  action = action_gen.next()

  if depth == 1:
    conclusion = attention_mask[-1, :]
    obj1, obj2 = [sequence[i] for i, x in enumerate(conclusion) 
                  if x == 1 and i > 0]
    state.add_relations(action.new_objects)
    found = False
    for val, rels in state.val2valrel.items():
      objs = [rel.init_list[0] for rel in rels]
      if obj1 in objs and obj2 in objs:
        found = True
        break
    assert found
    print('YAS')


data_fields = dict(
    sequence=tf.FixedLenFeature([128], tf.int64),
    attention_mask=tf.FixedLenFeature([128 * 4], tf.int64),
    theorem=tf.FixedLenFeature([1], tf.int64),
    targets=tf.FixedLenFeature([8], tf.int64),
    depth=tf.FixedLenFeature([1], tf.int64)
)


geo_problem = problem.GeoUpto5()

hparams = t2t_trainer.create_hparams()
t2t_trainer.trainer_lib.add_problem_hparams(hparams, tf.flags.FLAGS.problem)

hparams.batch_shuffle_size = 8

dataset = geo_problem.input_fn(
    tf.estimator.ModeKeys.TRAIN,
    hparams,
    data_dir='data',
    params=None,
    config=None,
    force_repeat=False,
    prevent_repeat=False,
    dataset_kwargs=dict(output_buffer_size=8, 
                        shuffle_buffer_size=8))

features = dataset.make_one_shot_iterator().get_next()[0]
features.update(attention_mask=model.dec_to_bin_att_mask(features['attention_mask']))

keys = ['sequence', 'theorem', 'attention_mask', 'targets', 'depth']
sess = tf.Session()


while True:
  f = sess.run(features)
  # separate each slice in batch dimension:
  for sequence, theorem, mask, targets, depth in zip(
      *[f[key] for key in keys]):

    targets = targets.flatten()
    theorem = action_vocab[theorem[0]]
    mask = mask.astype(bool)

    check(sequence=[vocab[x] for x in sequence if x > 0], 
          attention_mask=mask,
          theorem=theorem,
          target={x: y for x, y in zip(sorted(theorem.names.keys()), 
                                       targets)},
          depth=depth[0])



