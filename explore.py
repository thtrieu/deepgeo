"""Implement the environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theorems_utils
import geometry
import sketch
import theorems
import numpy as np
import time
import os
import trieu_graph_match
import glob
import pickle as pkl

from collections import defaultdict as ddict

from theorems_utils import collinear, concyclic, in_halfplane
from theorems_utils import divides_halfplanes, line_and_halfplanes
from theorems_utils import have_length, have_measure, have_direction
from theorems_utils import segment_def, angle_def
from theorems_utils import diff_side, same_side
from theorems_utils import Conclusion

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import LineDirectionPerpendicular, PointCentersCircle
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint

import tensorflow as tf

tf.compat.v1.flags.DEFINE_boolean('pdb', False, '')
tf.compat.v1.flags.DEFINE_string('out_dir', '/Users/thtrieu/deepgeo/data/', '')
tf.compat.v1.flags.DEFINE_integer('worker_id', 0, '')

FLAGS = tf.compat.v1.flags.FLAGS


non_relations = [
    Point, Line, Segment, Angle, HalfPlane, Circle,
    SegmentLength, AngleMeasure, LineDirection
]


def print_construction(constructions):
  for c in constructions:
    obj, rels = c[0], c[1:]
    if isinstance(obj, (Point, Segment, Line, HalfPlane, Angle, Circle,
                        SegmentLength, AngleMeasure, LineDirection)):
      print('Build {} {} such that'.format(type(obj).__name__, obj.name))
    else:
      print('Add Relation {}: {}'.format(type(obj).__name__, obj.name))
    for rel in rels:
      print('\t * {}'.format(rel.name))


class ExplorationBackoffDFS(object):

  def __init__(self, 
               init_state,
               init_canvas,
               out_dir,
               init_action_chain,
               max_construction=7,
               max_depth=45,
               max_line=6,
               max_point=8,
               max_circle=0):
    self.init_state = init_state
    self.init_canvas = init_canvas
    self.init_action_chain = init_action_chain
    self.proof_count = 0

    self.max_construction = max_construction
    self.max_depth = max_depth
    self.max_line = max_line
    self.max_point = max_point
    self.max_circle = max_circle

    self.proof_extractor = ProofExtractor(out_dir)
    self.construct_theorems = [
        theorems.all_theorems['mid'],
        theorems.all_theorems['mirror'],
        theorems.all_theorems['seg_line'],
        theorems.all_theorems['parallel'],
        theorems.all_theorems['line'],
    ]
    self.deduct_theorems = [
        theorems.all_theorems['eq'],
        theorems.all_theorems['.parallel'],
        theorems.all_theorems['.parallel2'],
        theorems.all_theorems['sas'],
        theorems.all_theorems['asa'],
    ]
    self.all_theorems = self.construct_theorems + self.deduct_theorems

  def interactive_choose_theorem(self, state, depth, canvas, mode='auto'):
    while True:
      print('\nDepth = {}'.format(depth))
      print(' '.join([name for name, obj in state.name2obj.items()
                      if isinstance(obj, Circle)] +
                     [name for name, obj in state.name2obj.items()
                      if isinstance(obj, Line)]
                    ))
      for i, theorem in enumerate(self.all_theorems):
        print('{}. {}'.format(i, type(theorem).__name__))

      choice = raw_input('[Q]uit [I]nspect [P]db >> ').lower()
      if choice == 'q':
        exit()
      elif choice == 'i':
        [print(r.name) for r in state.relations]
        continue
      elif choice == 'p':
        import pdb; pdb.set_trace()
        continue
      else:
        try:
          choice = int(choice)
        except ValueError:
          continue

      theorem = self.all_theorems[choice]

      action_generator = theorem.match_all(state)
      if mode == 'input':
        names = theorem.names
        prompt = ','.join(names)
        mapping = {}
        my_input = raw_input('{} = '.format(prompt)).split()

        not_found = False
        for premise_obj_name, state_obj_name in zip(names, my_input):
          state_obj_name = state_obj_name.strip()
          if state_obj_name not in state.name2obj:
            print('Not found {}'.format(state_obj_name))
            not_found = True
            break

          mapping.update({theorem.names[premise_obj_name]:
                          state.name2obj[state_obj_name]})

        if not_found:
          continue

        action_generator = theorem.match_from_input_mapping(
            state, mapping)

      start_time = time.time()
      for action in action_generator:
        duration = time.time() - start_time
        signature = ''
        for obj in action.theorem.premise_objects:
          if isinstance(obj, (Point, Line)):
            signature += '{}::{}  '.format(obj.name, action.mapping[obj].name)
        print('\t {} ({:.3}s)'.format(signature, duration))

        choice = raw_input('[Y]es [N]ext [E]scape [I]nspect [P]db >> ')
        choice = choice.lower()
        if choice == 'y':
          return action
        elif choice == 'e':
          break
        elif choice == 'i':
          [print(r.name) for r in state.relations]
          continue
        elif choice == 'p':
          import pdb; pdb.set_trace()
        start_time = time.time()

  def interactive_choose_action(self, state, depth, canvas):
    theorem_actions = [theorem.match_all(state)
                       for theorem in self.all_theorems]
    while True:
      print('\nDepth = {}'.format(depth))
      all_actions = []
      for i, theorem in enumerate(self.all_theorems):
        try:
          start_time = time.time()
          action = theorem_actions[i].next()
          all_actions.append(action)
        except StopIteration:
          continue
        duration = time.time() - start_time
        signature = '  '.join([
            '{}::{}'.format(obj.name, action.mapping[obj].name)
            for obj in action.theorem.premise_objects
            if isinstance(obj, Point)
        ])
        print('{}. {}'.format(len(all_actions) - 1, type(theorem).__name__))
        print('\t {} ({:.3}s)'.format(signature, duration))

      choice = raw_input('[N]ext [Q]uit [I]nspect [P]db >> ')
      choice = choice.lower()
      if choice == 'q':
        exit()
      elif choice == 'i':
        [print(r.name) for r in state.relations]
        continue
      elif choice == 'p':
        import pdb; pdb.set_trace()
      else:
        try:
          choice = int(choice)
        except ValueError:
          continue

      if choice < len(all_actions):
        return all_actions[choice]

  def explore_interactive(self, action_chain, state, canvas, 
                          depth=0, mode='theorem'):
    """DFS."""
    if depth > self.max_depth:
      return

    if mode == 'theorem':
      action = self.interactive_choose_theorem(state, depth, canvas,)
    elif mode == 'theorem_input':
      action = self.interactive_choose_theorem(state, depth, canvas, mode='input')
    elif mode == 'action':
      action = self.interactive_choose_action(state, depth, canvas)
    else:
      raise ValueError('Unrecognized Interactive Mode {}'.format(mode))
    print_construction(action.matched_conclusion.topological_list)

    # This is needed for whittling proof.
    action.set_chain_position(depth)

    # Branch the tree by copying state & canvas.
    action_chain.append(action)
    s = time.time()
    new_state = state.copy()
    new_canvas = canvas.copy()
    print(' * copy ' + str(time.time()-s))

    s = time.time()
    new_state.add_relations(action.new_objects)
    print(' * add rel ' + str(time.time()-s))
    # By drawing we add spatial relations (point in halfplanes)
    # to the state through inspection.
    # spatial_relations is dict(line: [a1, a2, ..], [b1, b2, ..])
    # where a1, a2 are points on the same halfplane and so are b1, b2..
    s = time.time()
    line2pointgroups = action.draw(new_canvas)
    print(' * draw ' + str(time.time()-s))
    s = time.time()
    new_state.add_spatial_relations(line2pointgroups)
    print(' * add spatial rel ' + str(time.time()-s))

    self.explore_interactive(
        action_chain, new_state, new_canvas, depth+1, mode)
    action_chain.pop(-1)

  def random_action(self, state, depth, canvas):
    if depth <= self.max_construction:
      all_theorems = list(self.construct_theorems)
      theorem_actions = [theorem.match_all(state)
                         for theorem in all_theorems]
    else:
      all_theorems = list(self.deduct_theorems)
      theorem_actions = [theorem.match_all(state)
                         for theorem in all_theorems]

    while all_theorems:
      i = np.random.randint(len(all_theorems))
      theorem = all_theorems[i]
      action_generator = theorem_actions[i]

      if (isinstance(theorem, 
                     (theorems.ConstructThirdLine,
                      theorems.ConstructParallelLine)) 
          and len(canvas.lines) >= self.max_line):
        theorem_actions.pop(i)
        all_theorems.pop(i)
        continue

      if isinstance(theorem, 
                    (theorems.ConstructMidPoint,
                     theorems.ConstructIntersectSegmentLine)
                    ) and len(canvas.lines) >= self.max_point:
        theorem_actions.pop(i)
        all_theorems.pop(i)
        continue

      try:
        s = time.time()
        action = action_generator.next()
        action.duration = time.time() - s
        yield action
      except StopIteration:
        # print('give up in ', time.time() - s, ', left ', len(all_theorems))
        theorem_actions.pop(i)
        all_theorems.pop(i)
        continue

  def explore(self, do_pdb=False):
    self._recursive_explore(
        list(self.init_action_chain), 
        self.init_state, 
        self.init_canvas,
        do_pdb=do_pdb)

  def explore_steps(self, steps, do_pdb=False):
    self._recursive_explore(
        list(self.init_action_chain), 
        self.init_state, 
        self.init_canvas,
        steps=steps,
        do_pdb=do_pdb)

  def _recursive_explore(self, action_chain, state, canvas, 
                         steps=None, do_pdb=False, depth=None):
    """DFS."""
    if depth is None:
      depth = len(action_chain) + 1

    if depth > self.max_depth:
      depth0 = len(self.init_action_chain) + 1
      x = np.arange(depth0, self.max_construction)
      backoff = np.random.choice(x, p=x[::-1]*1.0/np.sum(x))
      print('Reach max depth ', depth, ' backoff = ', backoff)
      return backoff

    if steps is None:
      actions = self.random_action(state, depth, canvas)
    elif steps == []:
      return 0
    else:
      def action_gen(steps):
        while True:
          step = steps.pop(0)
          theorem, command_str = step
          name_maps = [c.split('=') for c in command_str.split()]
          mapping = {theorem.names[a]: _find(state, b) for a, b in name_maps}
          action_gen = theorem.match_from_input_mapping(state, mapping)
          try:
            action = action_gen.next()
          except StopIteration:
            raise ValueError('Matching not found {} {} {}'.format(
                depth, theorem.name, command_str))
          yield action

      actions = action_gen(steps)

    for action in actions:
      print(' ' * depth, depth, type(action.theorem).__name__, action.duration)
      # This is needed for whittling proof.
      action.set_chain_position(depth - 1)
      action_chain.append(action)

      # Branch the tree by copying state & canvas.
      # s = time.time()
      new_state = state.copy()
      new_canvas = canvas.copy()
      # print(' * copy ', time.time() - s)

      # s = time.time()
      try:
        new_state.add_relations(action.new_objects)
      except ValueError:
        # Not happening, but if it does, back to 1.
        # import pdb; pdb.set_trace()
        return 1
      # print(' * add ', time.time() - s)

      # By drawing we add spatial relations (point in halfplanes)
      # to the state through inspection.
      # spatial_relations is dict(line: [a1, a2, ..], [b1, b2, ..])
      # where a1, a2 are points on the same halfplane and so are b1, b2..
      # s = time.time()
      line2pointgroups = action.draw(new_canvas)
      # print(' * draw ', time.time() - s)

      # s = time.time()
      new_state.add_spatial_relations(line2pointgroups)
      # print(' * spatial ', time.time() - s)

      s = time.time()
      self.proof_extractor.collect_proof(
          action_chain, self.init_state, self.init_canvas, 
          new_state, new_canvas, do_pdb)
      print(' '.join(['{}: {}'.format(key, reservoir.size)
                      for key, reservoir in sorted(
                      self.proof_extractor.reservoirs.items()
                      ) if reservoir.size > 0]))

      backoff = self._recursive_explore(
          action_chain, new_state, new_canvas,
          steps=steps, do_pdb=do_pdb, depth=depth+1)
      action_chain.pop(-1)

      if backoff < depth:
        return backoff

    if depth == 1:
      # Out of option at depth = 1, do it again.
      print('Out of option at depth 1, start a new Backoff DFS.')
      self.explore()
    else:
      depth0 = len(self.init_action_chain) + 1
      x = np.arange(depth0, self.max_construction)
      backoff = np.random.choice(x, p=x[::-1]*1.0/np.sum(x))
      print('Out of option at depth ', depth, ' backoff = ', backoff)
      # import pdb; pdb.set_trace()
      return backoff


def get_state_and_proof_objects(last_action, state):
  # Collect new value to rel in conclusion:
  new_objs, val2objs = [], {}
  for rel in last_action.new_objects:
    if isinstance(rel, (SegmentHasLength, LineHasDirection, AngleHasMeasure)):
      obj, val = rel.init_list
      new_objs.append(obj)
      val2objs[val] = state.val2valrel[val]

  # Loop through values correspond to new objects
  for val, rels in val2objs.items():
    # if there are < 2 objs associated with this val
    # then we move on
    if len(rels) < 2:
      continue

    # Loop through all distinct pair of rels
    for i, rel1 in enumerate(rels[:-1]):
      for rel2 in rels[i+1:]:
        # both objects are not new, move on.
        obj1, obj2 = rel1.init_list[0], rel2.init_list[0]
        if obj1 not in new_objs and obj2 not in new_objs:
          continue
        # Else yield the state and proof queues
        problem_queue = [obj1, obj2]
        proof_queue = [(val, rel1, rel2)]
        yield problem_queue, proof_queue


class ProofReservoir(object):

  def __init__(self, depth, out_dir, max_store=1000):
    self.depth = depth
    self.name = 'res.{:03}.depth.{:02}'.format(FLAGS.worker_id, depth)
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
    with tf.io.gfile.GFile(os.path.join(self.out_dir, filename), 'w') as f:
      f.write(pkl.dumps(all_arrays, protocol=pkl.HIGHEST_PROTOCOL))
    
    self.store = []


class ProofExtractor(object):

  def __init__(self, out_dir, max_state_size=127):
    self.max_state_size = max_state_size
    self.reservoirs = {depth: ProofReservoir(depth, out_dir)
                       for depth in range(100)}
    self.opposite_angle_check = theorems.OppositeAnglesCheck()
    self.thales_check = theorems.ThalesCheck()
    # self.checks = [theorems.ThalesCheck(), 
    #                theorems.OppositeAnglesCheck()]

  def collect_proof(self, action_chain, init_state, init_canvas, 
                    full_state, full_canvas, do_pdb=False):
    new_action = action_chain[-1]
    if not isinstance(new_action.theorem, 
                      (theorems.ASA, theorems.SAS, theorems.SSS, 
                       theorems.ParallelBecauseCorrespondingAngles,
                       theorems.ParallelBecauseInteriorAngles)):
      return []

    all_lengths = []
    last_action = action_chain[-1]
    all_discoveries = get_state_and_proof_objects(last_action, full_state)

    for problem_queue, proof_queue in all_discoveries:    
      problem_constructions = whittle_from(
          list(problem_queue), action_chain)
      proof_steps = whittle_from(
          list(proof_queue), action_chain, 
          problem_queue, problem_constructions)

      # Transfer all partial steps into problem formulation
      for i, step in enumerate(proof_steps):
        if not (step == [] or step == True):
          if problem_constructions[i] != True:
            problem_constructions[i] += step
          proof_steps[i] = []
      # So now all proof steps are True or []

      length = sum([1 for x in proof_steps if x != []])
      all_lengths.append(length)

      # if length >= 5:
      print()
      print(action_chain[-1].theorem.name, length)
      for i, (action, s) in enumerate(zip(action_chain, problem_constructions)):
        duration = action.duration
        if s == True:
          print(i + 1, action.to_str(), duration)
        elif s:
          print(i + 1, action.theorem.name, [r.name for r in sum(s, [])], duration)
      print('----------', [r.name for r in proof_queue[0]])
      for i, (action, s) in enumerate(zip(action_chain, proof_steps)):
        duration = action.duration
        if s == True:
          print(i + 1, action.to_str(), duration)
        elif s:
          print(i + 1, action.theorem.name, [r.name for r in sum(s, [])], duration)

      self.create_training_examples(
          init_state,
          full_state,
          action_chain,
          problem_constructions,
          proof_queue[0],
          proof_steps)

  def create_training_examples(
      self, 
      init_state, 
      full_state,
      action_chain,
      problem_constructions,
      goal_objects,
      proof_steps):

    # Copy the init state
    problem_state = init_state.copy()

    redundant_actions = []  # for data aug
    for action, construction, step in zip(action_chain,
                                          problem_constructions,
                                          proof_steps):
      if construction == [] and step == []:
        redundant_actions.append(action)

    # Add relation to state
    for construction, action in zip(problem_constructions, action_chain):
      if construction == []:
        continue
      if construction == True:
        problem_state.add_relations(action.conclusion_objects)
      else:
        all_constructions = sum(construction, [])
        all_constructions = list(set(all_constructions))
        problem_state.add_relations(all_constructions)

    # Add spatial relations
    for name, obj in problem_state.name2obj.items():
      if isinstance(obj, Line):
        hp1, hp2 = full_state.line2hps[obj]
        # problem_state.add_one(hp1)
        # problem_state.add_one(hp2)
        problem_state.add_one(LineBordersHalfplane(obj, hp1))
        problem_state.add_one(LineBordersHalfplane(obj, hp2))

    for hp in problem_state.all_hps:
      for p in full_state.hp2points[hp]:
        if p in problem_state.all_points:
          problem_state.add_one(HalfPlaneContainsPoint(hp, p))

    if self.opposite_angle_check.found(problem_state, goal_objects):
      return  # Nothing to look at here.

    found_thales = self.thales_check.found(problem_state, goal_objects)

    # Now we loop through actions in the proof steps
    proof_actions = [action for action, step in 
                     zip(action_chain, proof_steps)
                     if step != []]

    for i, action in enumerate(proof_actions):
      # print('Try {}'.format(action.to_str()))
      # The current graph is too big (size = # nodes)
      graph_size = len(problem_state.name2obj) + 1
      if graph_size > self.max_state_size:
        break

      # Now we randomly add various amount of redundant actions
      # into the problem state to augment the training examples
      size_maxed = True
      for example in self.augment_and_serialize(
          problem_state, goal_objects, action, full_state, redundant_actions):
        size_maxed = False

        # We isolate the first step of thales proof into reservoir 0
        if i == 0 and found_thales:
          reservoir_id = 0
        else:
          reservoir_id = len(proof_actions) - i  # always >= 1
        self.reservoirs[reservoir_id].add(example)

      if size_maxed:
        break

      # Now we add the action into the problem state
      add_action(problem_state, action, full_state)

  def augment_and_serialize(
      self, state, goal_objects, action, full_state, redundant_actions):
    state = state.copy()
    np.random.shuffle(redundant_actions)

    # Yield the non augmented version.
    if len(state.name2obj) + 1 > self.max_state_size:
      return  # No example for this one.
    example = self.serialize(state, goal_objects, action)
    if example is None:
      # [r.name for r in action.premise_objects if r not in state.relations and r.name not in state.name2obj]
      # {l.name: (h1.name, h2.name) for l, (h1, h2) in state.line2hps.items()}
      # {hp.name: [p.name for p in ps] for hp, ps in state.hp2points.items()}
      # {l.name: ([p.name for p in p1], [p.name for p in p2]) for l, (p1, p2) in canvas.line2hps.items()}
      # import pdb; pdb.set_trace()
      # raise ValueError()
      import pdb; pdb.set_trace()
      raise ValueError('Cannot apply action {}'.format(action.to_str()))

    yield example

    # Add redundant action until reach full size
    for red_action in redundant_actions:
      # print(' * Try ', red_action.to_str())
      add_action(state, red_action, full_state)
      if len(state.name2obj) + 1 > self.max_state_size:
        break
      example = self.serialize(state, goal_objects, action)
      if example:
        yield example

  def serialize(self, state, goal_objects, action):
    assert len(state.name2obj) + 1 <= self.max_state_size
    match = {y: action.mapping[y]
             for x, y in action.theorem.names.items()}
    action_gen = action.theorem.match_from_input_mapping(
      state, match)
    try:
      new_action = action_gen.next()
    except StopIteration:
      match = {x: action.mapping[y].name
               for x, y in action.theorem.names.items()}
      print(match)
      print('Failed matching {} {}'.format(action.theorem.name, match))
      # [r.name for r in action.premise_objects if r not in state.relations]
      # {l.name: (h1.name, h2.name) for l, (h1, h2) in state.line2hps.items()}
      # {hp.name: [p.name for p in ps] for hp, ps in state.hp2points.items()}
      # {l.name: ([p.name for p in p1], [p.name for p in p2]) for l, (p1, p2) in canvas.line2hps.items()}
      # import pdb; pdb.set_trace()
      # raise ValueError()
      return None

    seq = [1]  # CLS
    obj2idx = {}
    connections = []

    for relation in state.relations:
      for obj in relation.init_list:
        if obj not in obj2idx:
          obj2idx[obj] = len(seq)
          seq.append(vocab[type(obj)])

      obj1, obj2 = relation.init_list
      connections.append((obj2idx[obj1], obj2idx[obj2]))
      connections.append((obj2idx[obj2], obj2idx[obj1]))

    val, rel1, rel2 = goal_objects
    obj1, obj2 = rel1.init_list[0], rel2.init_list[0]

    val_idx = len(seq)
    seq.append(vocab[type(val)])
    connections.append((val_idx, obj2idx[obj1]))
    connections.append((val_idx, obj2idx[obj2]))
    connections.append((obj2idx[obj1], val_idx))
    connections.append((obj2idx[obj2], val_idx))

    # if len(obj2idx) != len(state.name2obj):
    #   import pdb; pdb.set_trace()

    attention_mask = np.zeros([len(seq), len(seq)], dtype=bool)
    for id1, id2 in connections:
      attention_mask[id1, id2] = True
    # CLS look at everything and vice versa.
    attention_mask[:, 0] = True
    attention_mask[0, :] = True

    target = [obj2idx[action.mapping[obj]]
              for _, obj in sorted(action.theorem.names.items())]
    target = [action_vocab[type(action.theorem)]] + target

    return Example(np.array(seq, np.int8), 
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
    theorems.ParallelBecauseInteriorAngles: 9,
}

vocab = {
    # PAD: 0
    # CLS: 1
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

def add_action(state, action, full_state):
  for obj in action.conclusion_objects:
    # if obj.name in state.name2obj:
    #   continue
    state.add_one(obj)
    if isinstance(obj, Line):
      # canvas.update_line(obj, full_canvas.lines[obj])
      hp1, hp2 = full_state.line2hps[obj]
      # state.add_one(hp1)
      # state.add_one(hp2)
      state.add_one(LineBordersHalfplane(obj, hp1))
      state.add_one(LineBordersHalfplane(obj, hp2))

  for hp in state.all_hps:
    for p in full_state.hp2points[hp]:
      if p in state.all_points:
        state.add_one(HalfPlaneContainsPoint(hp, p))


value_entity = (
    AngleMeasure, SegmentLength, LineDirection
)

value_rels = (
    AngleHasMeasure, SegmentHasLength, LineHasDirection
)


def print_name(dependents):
  s = []
  for d in dependents:
    if isinstance(d, tuple):
      i, a, b = d
      s.append('{} {} {}'.format(i.name, a.name, b.name))
    elif isinstance(d, int):
      s.append(str(d))
    else:
      s.append(d.name)
  return ', '.join(s)


def whittle_from(queue, action_chain, 
                 goal_objects=None, whittled_state=None):
  # Whittled info will be put into here:
  whittled = [[] for _ in range(len(action_chain))]
  # We expect empty [] if the corresponding action in action_chain
  # is not relevant, True, if the whole action is needed
  # and a list of constructions, if action is not needed but only
  # part of its conclusion
  # Keep track of the head of the queue 
  # (we don't pop things from queue)
  i = 0
  non_critical_count = 0  # count when the whole action is not needed.

  while i < len(queue):
    query = queue[i]
    i += 1

    if isinstance(query, tuple):
      val, rel1, rel2 = query
      obj1, obj2 = rel1.init_list[0], rel2.init_list[0]
      dependents = val.dependency_path(obj1, obj2)
      if not all([d is not None for d in dependents]):
        import pdb; pdb.set_trace()
        raise ValueError('Path not found between {} and {} in {}'.format(
            obj1.name, obj2.name,
            {x.name: {a.name: b for a, b in y.items()} for x, y in val.edges.items()}))
      # dependents = [pos for pos in positions if pos is not None]
      # if obj1.name == '^23' and obj2.name == '^17':
      #   import pdb; pdb.set_trace()
      dependents += [obj1, obj2]
      queue.extend(dependents)
      # {x.name: {a.name: b for a, b in y.items()} for x, y in val.edges.items()}
      # import pdb; pdb.set_trace()
      # print('{} {} <= {}'.format(rel1.name, rel2.name, print_name(dependents)))
      # import pdb; pdb.set_trace()
      continue

    if isinstance(query, int):
      critical = True
      pos = query
    else:
      pos = query.chain_position  # at init state already
      if pos is None:
        continue
      critical = query.critical

    # the whole action and its premise is visited.
    if (whittled_state and whittled_state[pos] == True 
        or whittled[pos] == True): 
      # if isinstance(query, int):
      #   print(' X Skip {}'.format(query))
      # else:
      #   print(' X Skip {} because {} fulfilled'.format(query.name, pos))
      continue

    action = action_chain[pos]
    # When the whole action is not needed and there is still
    # critical query, we defer this query to the end
    # This optimizes running time because it maximizes
    # the hit `if whittled[pos] == True` above.
    if not critical and len(queue) - (i-1) > non_critical_count:
      non_critical_count += 1
      queue.append(query)
      continue
    elif critical:
      # The whole action is needed.
      whittled[pos] = True
      # Unless it is the goal, state doesnt have to create
      # the objs if proof already covers them
      # if whittled_state:
      #   whittled_state[pos] = [construct 
      #                          for construct in whittled_state[pos]
      #                          if construct[0] in goal_objects]
      dependents = []
      valrels = {}
      for obj in action.premise_objects:
        if not isinstance(obj, value_entity + value_rels):
          dependents.append(obj)
        elif isinstance(obj, value_rels):
          val = obj.init_list[1]
          if val not in valrels:
            valrels[val] = []
          valrels[val].append(obj)
      dependents += [(val, rel1, rel2) if rel1 != rel2 else rel1
                      for val, (rel1, rel2) in valrels.items()]
      # print('*', pos, action.theorem.name, '<=', print_name(dependents))
    else:
      found = action.matched_conclusion.topological_list[
          query.conclusion_position]
      whittled[pos].append(found)
      # Here we ignore the relations in `found` themselves
      # because we know that they are created at chain_pos = pos
      # there is no need to go further. Only init_list are concerned.
      dependents = sum([c.init_list for c in found
                        if hasattr(c, 'init_list')], tuple())
      non_critical_count -= 1
      # print(query.name, '<=', print_name(dependents))

    # Push dependents into queue.
    for dep in dependents:
      if dep not in queue:
        queue.append(dep)
      if hasattr(dep, 'init_list'):
        a, b = dep.init_list
        if a not in queue:
          queue.append(a)
        if b not in queue:
          queue.append(b)

  return whittled


def _is_numeric(string):
  try:
    _ = int(string)
    return True
  except:
    return False


def _find(state, name):
  if name in state.name2obj:
    return state.name2obj[name]

  names = [n.split('_') for n in state.name2obj.keys()]
  names = [n for n in names if (
              len(n) == 2 and
              n[0] == name and
              _is_numeric(n[1]) 
           )]
  if len(names) != 1:
    raise ValueError('Failed looking for {}'.format(name))

  name = '_'.join(names[0])
  return state.name2obj[name]


def _find_premise(premise_objects, name):
  for obj in premise_objects:
    if obj.name.startswith(name):
      return obj
  return None


def execute_steps(steps, state, canvas, verbose=False):
  action_chain = []

  for i, (theorem, command) in enumerate(steps):
    print(i + 1, ' ', type(theorem).__name__, command)
    name_maps = [c.split('=') for c in command.split()]
    mapping = dict(
        (theorem.names[a], _find(state, b))
        if a in theorem.names
        else (_find_premise(theorem.premise_objects, a), _find(state, b))
        for a, b in name_maps)
    action_gen = theorem.match_from_input_mapping(state, mapping, randomize=False)

    try:
      action = action_gen.next()
    except StopIteration:
      raise ValueError('Matching not found {} {}'.format(theorem, command))

    # print(' '.join(['{}::{}'.format(x.name, y.name)
    #                 for x, y in action.mapping.items()
    #                 if isinstance(x, AngleHasMeasure)]))
    action.set_chain_position(i)
    action_chain.append(action)

    if verbose:
      print('\tAdd : {}'.format([obj.name for obj in action.new_objects]))
    state.add_relations(action.new_objects)
    line2pointgroups = action.draw(canvas)
    state.add_spatial_relations(line2pointgroups)

  return state, canvas, action_chain


def init_by_normal_triangle():
  geometry.reset()
  canvas = sketch.Canvas()
  state = theorems_utils.State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, BC, CA = map(Segment, 'AB BC CA'.split())

  state.add_relations(
      # [A, B, C, ab, bc, ca, AB, BC, CA] +
      segment_def(AB, A, B) +
      segment_def(BC, B, C) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A)
  )

  state.add_spatial_relations(canvas.add_triangle(A, B, C, ab, bc, ca))
  return state, canvas, []


def init_by_thales():
  geometry.reset()
  init_state, init_canvas, _ = init_by_normal_triangle()
  state, canvas = init_state.copy(), init_canvas.copy()

  steps = [
      (theorems.all_theorems['mid'], 'A=A B=B'),  # P1
      (theorems.all_theorems['parallel'], 'A=P1 l=bc'),  # l1
      (theorems.all_theorems['seg_line'], 'l=l1 A=A B=C'),  # P1
      (theorems.all_theorems['parallel'], 'A=C l=ab'),  # l2
      (theorems.all_theorems['line'], 'A=P1 B=C'),  # l3
  ]

  state, canvas, action_chain = execute_steps(steps, state, canvas)
  return state, canvas, action_chain



if __name__ == '__main__':
  np.random.seed(int(time.time() % 42949671) * 100 + FLAGS.worker_id)
  state, canvas, action_chain = init_by_normal_triangle()
  # state, canvas, action_chain = init_by_thales()
  explorer = ExplorationBackoffDFS(state, canvas, FLAGS.out_dir, action_chain)
  explorer.explore(FLAGS.pdb)
  # explorer.explore_interactive([], state, canvas, mode='theorem_input')

