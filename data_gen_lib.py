import numpy as np
import os

import theorems
import glob

import geometry
from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import LineBordersHalfplane, HalfPlaneContainsPoint

import traceback
import debugging


db = debugging.get_db()


class ProofReservoir(object):

  def __init__(self, depth, out_dir, worker_id, max_store=1000):
    self.depth = depth
    self.name = 'res.{:03}.depth.{:02}'.format(worker_id, depth)
    self.out_dir = out_dir
    self.store = []
    self.max_store = max_store
    self.size = 0

  def add(self, proof):
    self.store.append(proof)
    self.size += 1
    if len(self.store) == self.max_store:
      self.dump()

  def dump(self):
    files = os.path.join(self.out_dir, self.name)
    flush_count = len(glob.glob(files + '*'))

    filename = '{}.part.{:05}'.format(self.name, flush_count)
    all_arrays = sum([example.arrays for example in self.store], [])
    
    print('\n\t/!\\ Flushing {} ..\n'.format(filename))

    target_file = os.path.join(self.out_dir, filename)
    write_to = target_file

    with open(write_to, 'wb') as f:
      np.savez_compressed(f, *all_arrays)
    
    self.store = []


def build_theorem_premise(theorem_premise,
                          theorem_premise_constructions, 
                          action_chain,
                          full_state):
  # Add relevant relations to our theorem_premise
  for construction, action in zip(theorem_premise_constructions, 
                                  action_chain):
    if construction == []:
      continue
    if construction == True:  # Full action
      theorem_premise.add_relations(action.conclusion_objects)
    else:  # Only the relevant part of action is added
      all_constructions = sum(construction, [])
      all_constructions = list(set(all_constructions))
      theorem_premise.add_relations(all_constructions)

  # Next, we add spatial relations
  # First we copy the line-hp relations over
  items = list(theorem_premise.name2obj.items())
  for _, obj in items:
    if isinstance(obj, Line):
      hp1, hp2 = full_state.line2hps[obj]
      # problem_state.add_one(hp1)
      # problem_state.add_one(hp2)
      theorem_premise.add_one(LineBordersHalfplane(obj, hp1))
      theorem_premise.add_one(LineBordersHalfplane(obj, hp2))

  # Second we copy the hp-point relations over:
  for hp in theorem_premise.all_hps:
    if hp not in full_state.hp2points:
      # It is possible that some hp exists without 
      # containing any points, but only for specifying angles.
      continue
    for p in full_state.hp2points[hp]:
      if p in theorem_premise.all_points:
        theorem_premise.add_one(HalfPlaneContainsPoint(hp, p))


def serialize_state(state):
  state_obj_ids = [1]  # CLS
  obj_list = ['CLS']
  obj2idx = {}
  connections = []

  for relation in state.relations:
    for obj in relation.init_list:
      if obj not in obj2idx:
        obj2idx[obj] = len(state_obj_ids)
        # Halfpi got a separate embedding to other Angles:
        if obj == geometry.halfpi:
          state_obj_ids.append(vocab[obj])
        else:
          state_obj_ids.append(vocab[type(obj)])
        obj_list.append(obj)

    obj1, obj2 = relation.init_list
    connections.append((obj2idx[obj1], obj2idx[obj2]))
    connections.append((obj2idx[obj2], obj2idx[obj1]))

  attention_mask = np.zeros([len(state_obj_ids) + 1,  # +1 for goal val.
                             len(state_obj_ids) + 1], dtype=bool)
  for id1, id2 in connections:
    attention_mask[id1, id2] = True
  # CLS look at everything and vice versa.
  attention_mask[:, 0] = True
  attention_mask[0, :] = True

  return state_obj_ids, obj_list, obj2idx, attention_mask


def build_example(action, state_obj_ids, goal_objects, attention_mask, obj2idx):
  # Get the goal object triplet (val, obj1, obj2)
  val, rel1, rel2 = goal_objects
  obj1, obj2 = rel1.init_list[0], rel2.init_list[0]

  # Get their idx and appropriately add masks to them.
  val_idx = len(state_obj_ids)
  state_obj_ids.append(vocab[type(val)])

  for obj in [obj1, obj2]:
    try: 
      obj_idx = obj2idx[obj]
    except:
      traceback.print_exc()
      import pdb; pdb.set_trace()
    attention_mask[val_idx, obj_idx] = True
    attention_mask[obj_idx, val_idx] = True

  target = [obj2idx[action.mapping[obj]]
            for _, obj in sorted(action.theorem.names.items())]
  target = [action_vocab[type(action.theorem)]] + target

  return Example(
      np.array(state_obj_ids, np.int8), 
      attention_mask,
      np.array(target, np.int8))



class Example(object):

  def __init__(self, sequence, attention_mask, target):
    self.sequence = sequence
    self.attention_mask = attention_mask
    self.target = target
    self.arrays = [sequence, attention_mask, target]


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
    # theorems.ParallelBecauseInteriorAngles: 9,
    theorems.ConstructPerpendicularLineFromPointOn: 10,
    theorems.ConstructPerpendicularLineFromPointOut: 11,
    theorems.ConstructAngleBisector: 12
    # ============
    # Only add things after this point for backward compat
}

vocab = {
    'PAD': 0,
    'CLS': 1,
    Point: 2,
    Segment: 3,
    Line: 4,
    HalfPlane: 5,
    Angle: 6,
    Circle: 7,
    SegmentLength: 8,
    AngleMeasure: 9,
    LineDirection: 10,
    geometry.halfpi: 11
    # ============
    # Only add things after this point for backward compat
}
