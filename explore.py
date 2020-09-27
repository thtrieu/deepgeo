"""Implement the environment.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os
import sys
import glob
import pickle as pkl
import traceback
import profiling

import theorems_utils
import geometry
import sketch
import theorems
import trieu_graph_match
import whittling
import data_gen_lib
import action_chain_lib

from IPython.display import clear_output

from profiling import Timer
from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection

from matplotlib import pyplot as plt

import tensorflow as tf
tf.compat.v1.flags.DEFINE_boolean('pdb', False, '')
tf.compat.v1.flags.DEFINE_boolean('verbose', False, '')
tf.compat.v1.flags.DEFINE_boolean('enable_profiling', False, '')

tf.compat.v1.flags.DEFINE_string('mode', 'datagen', 'datagen or interactive')
tf.compat.v1.flags.DEFINE_string('out_dir', '/Users/thtrieu/deepgeo/data_small/', '')

tf.compat.v1.flags.DEFINE_integer('max_construction', 7, '')
tf.compat.v1.flags.DEFINE_integer('max_depth', 45, '')
tf.compat.v1.flags.DEFINE_integer('max_line', 6, '')
tf.compat.v1.flags.DEFINE_integer('max_point', 8, '')
tf.compat.v1.flags.DEFINE_integer('explore_worker_id', 0, '')


FLAGS = tf.compat.v1.flags.FLAGS


non_relations = [
    Point, Line, Segment, Angle, HalfPlane, Circle,
    SegmentLength, AngleMeasure, LineDirection
]


class DebugObjects(object):
  """Saves the explore chain from root upto current point."""

  def __init__(self):
    self.state = [None] * 100
    self.canvas = [None] * 100
    self.action = [None] * 100
    self.depth = None
  
  def update(self, depth, obj=None):
    self.depth = depth
    if isinstance(obj, sketch.Canvas):
      self.canvas[depth] = obj
    elif isinstance(obj, theorems_utils.State):
      self.state[depth] = obj
    elif isinstance(obj, theorems.Action):
      self.action[depth] = obj

  def save_chain(self, filename):
    # geometry.reset()
    # init_state, init_canvas, _ = init_by_normal_triangle()
    # state, canvas = init_state.copy(), init_canvas.copy()

    # steps = [
    #     (theorems.all_theorems['mid'], 'A=A B=B'),  # P1
    #     (theorems.all_theorems['parallel'], 'A=P1 l=bc'),  # l1
    #     (theorems.all_theorems['seg_line'], 'l=l1 A=A B=C'),  # P1
    #     (theorems.all_theorems['parallel'], 'A=C l=ab'),  # l2
    #     (theorems.all_theorems['line'], 'A=P1 B=C'),  # l3

    i = 0
    while self.state[i] is None:
      i += 1
    
    assert self.canvas[i] is not None
    init_state = self.state[i]
    init_canvas = self.canvas[i]
    import pdb; pdb.set_trace()


  def report(self):
    canvas_state = []
    print('\nDebug:')
    for i, (state, canvas, action) in enumerate(
        zip(self.state, self.canvas, self.action)):

      if (i == self.depth):
        break

      assert (state is None) == (canvas is None)
      if state is None:
        continue

      action_str = action.to_str() if action else 'None'
      print(i, action_str)
      canvas_state.append((canvas, state))
  
    n = len(canvas_state)
    nrows = int(np.ceil(n/3.))
    _, axes = plt.subplots(
        nrows=nrows, ncols=3, figsize=(nrows * 5, 3 * 5))
    for i, (canvas, state) in enumerate(canvas_state):
      ax = axes[i//3, i%3]
      canvas.plt_show(ax, state, [], mark_segment=0)
    plt.show()
      

db = DebugObjects()
                                

class ExplorationBackoffDFSBase(object):

  def __init__(self, 
               init_state,
               init_canvas,
               out_dir,
               predefined_steps=[],
               max_construction=10,
               max_depth=45,
               max_line=6,
               max_point=8,
               max_circle=0):
    self.init_state = init_state
    self.init_canvas = init_canvas
    self.predefined_steps = predefined_steps
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
        # theorems.all_theorems['perp_on'],
        # theorems.all_theorems['perp_out'],
        theorems.all_theorems['line'],
        theorems.all_theorems['bisect'],
    ]
    self.deduct_theorems = [
        theorems.all_theorems['eq'],
        theorems.all_theorems['.parallel'],
        theorems.all_theorems['.parallel2'],
        theorems.all_theorems['sas'],
        theorems.all_theorems['asa'],
    ]
    self.all_theorems = self.construct_theorems + self.deduct_theorems

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

  def explore(self, predefined_steps=None, do_pdb=False):
    self._recursive_explore(
        [], 
        self.init_state, 
        self.init_canvas,
        predefined_steps=self.predefined_steps,
        do_pdb=do_pdb)

  def get_actions(self, state, depth, canvas,
                  predefined_steps):
    if predefined_steps is None or predefined_steps == []:
      # Just do the normal random action sampling
      return self.random_action(state, depth, canvas)
    
    # if predefined_steps == []:
    #   return 0

    step = predefined_steps.pop(0)
    theorem, command_str = step
    name_maps = [c.split('=') for c in command_str.split()]
    mapping = {theorem.names[a]: action_chain_lib._find(state, b) 
               for a, b in name_maps}
    action_gen = theorem.match_from_input_mapping(state, mapping)
    try:
      action = action_gen.next()
    except StopIteration:
      raise ValueError('Matching not found {} {} {}'.format(
          depth, theorem.name, command_str))

    def action_gen_wrapper():
      yield action
    return action_gen_wrapper()

  def print_stats(self):
    if not FLAGS.verbose:
      os.system('clear')
    # Profiling different parts of pipeline
    if FLAGS.enable_profiling:
      profiling.print_records()
    # How many proof collected?
    self.proof_extractor.print_sizes()

  def _recursive_explore(self, action_chain, state, canvas, 
                         predefined_steps=None, 
                         do_pdb=False, 
                         depth=None):
    """Random Back-off DFS.

    Recursively call itself to explore the space in backoff DFS manner.
    If run out of eligible action or reach max_depth, randomly
    sample a backoff point in the current chain and go back there.
    """
    
    if depth is None:
      depth = len(action_chain) + 1

    db.update(depth, state)
    db.update(depth, canvas)

    if depth > self.max_depth:
      # Now we randomly sample a backoff point
      # with probability biased towards the beginning of the chain.
      # depth0 = len(self.init_action_chain) + 1
      depth0 = 1
      x = np.arange(depth0, self.max_construction)
      backoff = np.random.choice(x, p=x[::-1]*1.0/np.sum(x))
      if FLAGS.verbose:
        print('Reach max depth ', depth, ' backoff = ', backoff)
      self.print_stats()
      return backoff

    tab = ' ' * depth

    # Timing how long does it take to find 01 eligible action.
    timer = Timer('action', start=True)
    actions = self.get_actions(state, depth, canvas, predefined_steps)

    for action in actions:
      timer.stop()

      if FLAGS.verbose:
        print(tab, depth, type(action.theorem).__name__, action.duration)

      db.update(depth, action)
      # This is needed for whittling proof.
      action.set_chain_position(depth - 1)
      action_chain.append(action)

      # Branch the tree by copying state & canvas.
      with Timer('copy'):
        new_state = state.copy()
        new_canvas = canvas.copy()

      with Timer('add'):
        try:
          new_state.add_relations(action.new_objects)
        except ValueError:
          # Not happening, but if it does, back to 1.
          traceback.print_exc()
          db.report()
          import pdb; pdb.set_trace()
          exit()

      # By drawing we add spatial relations (point in halfplanes)
      # to the state through inspection.
      # spatial_relations is dict(line: [a1, a2, ..], [b1, b2, ..])
      # where a1, a2 are points on the same halfplane and so are b1, b2..
      with Timer('draw'):
        line2pointgroups = action.draw(new_canvas)

      with Timer('spatial'):
        new_state.add_spatial_relations(line2pointgroups)
        new_canvas.update_hps(new_state.line2hps)

      db.update(depth+1, new_state)
      db.update(depth+1, new_canvas)

      with Timer('proof'):
        try:
          self.proof_extractor.collect_proof(
              action_chain, self.init_state, self.init_canvas, 
              new_state, do_pdb)
        except:
          traceback.print_exc()
          db.report()
          db.save_chain()
          import pdb; pdb.set_trace()
          exit()

      backoff = self._recursive_explore(
          action_chain, new_state, new_canvas,
          predefined_steps=predefined_steps, 
          do_pdb=do_pdb, depth=depth+1)
      action_chain.pop(-1)

      if backoff < depth:
        return backoff

      # Timing next action.
      timer = Timer('action', start=True)
      # end loop

    # At this point, we are out of eligible action to pick:
    if depth > 1:
      # Random backoff
      # depth0 = len(self.init_action_chain) + 1
      depth0 = 1
      x = np.arange(depth0, self.max_construction)
      backoff = np.random.choice(x, p=x[::-1]*1.0/np.sum(x))
      if FLAGS.verbose:
        print('Out of option at depth ', depth, ' backoff = ', backoff)
      self.print_stats()
      return backoff

    # Out of option at depth = 1, do it again.
    if FLAGS.verbose:
      print('Out of option at depth 1, start a new Backoff DFS.')
    self.print_stats()
    self.explore()


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


class ExplorationBackoffDFS(ExplorationBackoffDFSBase):
  """With debugging support: interactive explore & predefined explore chains."""

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
        prompt = ' '.join(names)
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
                          depth=0, mode='theorem_input'):
    """DFS."""
    if depth > self.max_depth:
      return

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
    new_canvas.update_hps(new_state.line2hps)
    print(' * add spatial rel ' + str(time.time()-s))

    # self.proof_extractor.collect_proof(
    #     action_chain, self.init_state, self.init_canvas, 
    #     new_state, new_canvas, do_pdb)

    self.explore_interactive(
        action_chain, new_state, new_canvas, depth+1, mode)
    action_chain.pop(-1)


class ProofExtractor(object):

  def __init__(self, out_dir, max_state_size=127):
    self.max_state_size = max_state_size
    self.reservoirs = {
        depth: data_gen_lib.ProofReservoir(
            depth, out_dir, FLAGS.explore_worker_id)
        for depth in range(100)}
    self.opposite_angle_check = theorems.OppositeAnglesCheck()
    self.thales_check = theorems.ThalesCheck()
    # self.checks = [theorems.ThalesCheck(), 
    #                theorems.OppositeAnglesCheck()]

  def print_sizes(self):
    size_str = '\n'.join(['Size {:>3}: {}'.format(key, reservoir.size)
                         for key, reservoir in sorted(
                         self.reservoirs.items()
                        ) if reservoir.size > 0])
    if size_str:
      print('\nProof Reservoirs Sizes:')
      print(size_str)

  def collect_proof(self, action_chain, init_state, init_canvas, 
                    full_state, do_pdb=False):
    last_action = action_chain[-1]
    if not isinstance(last_action.theorem, 
                      (theorems.ASA, theorems.SAS, theorems.SSS, 
                       theorems.ParallelBecauseCorrespondingAngles,
                       theorems.ParallelBecauseInteriorAngles)):
      return []

    # With this new last_action, what are all possible new discoveries?
    all_discoveries = whittling.get_state_and_proof_objects(
        last_action, full_state)

    for equal_obj1_obj2, val_rel1_rel2 in all_discoveries:
      # Here:
      # equal_obj1_obj2 = [obj1, obj2]
      # val_rel1_rel2 = (val, rel1, rel2)
      # Where at least obj1 or obj2 is a new obj created by last_action.

      # theorem_premise_constructions is the constructions
      # on top of init_state to produce the theorem premise.
      with Timer('proof/whittle'):
        theorem_premise_constructions = whittling.whittle_from(
            list(equal_obj1_obj2),  # make a copy
            action_chain)

        # proof_steps are the actions on top of theorem_premise
        # to drive to the conclusion.
        proof_steps = whittling.whittle_from(
            [val_rel1_rel2],  # make a copy
            action_chain, 
            equal_obj1_obj2, 
            theorem_premise_constructions)

        # Transfer all partial steps
        # from proof_steps to theorem_premise_constructions
        for i, step in enumerate(proof_steps):
          if not (step == [] or step == True):
            if theorem_premise_constructions[i] != True:
              theorem_premise_constructions[i] += step
            proof_steps[i] = []

      with Timer('proof/serialize'):
        self.create_training_examples(
            init_state,
            full_state,
            action_chain,
            theorem_premise_constructions,
            goal_objects=val_rel1_rel2,
            proof_steps=proof_steps)

  def create_training_examples(
      self, 
      init_state, 
      full_state,
      action_chain,
      theorem_premise_constructions,
      goal_objects,
      proof_steps):

    # Copy the init state
    theorem_premise = init_state.copy()

    # Get the actions so far that is not contributing to this extracted
    # proof, so that we can add them later as augmentations
    redundant_actions = []  # for data aug
    for action, construction, step in zip(action_chain,
                                          theorem_premise_constructions,
                                          proof_steps):
      if construction == [] and step == []:
        redundant_actions.append(action)

    # Add theorem_premise_constructions into theorem_premise
    data_gen_lib.build_theorem_premise(
        theorem_premise,
        theorem_premise_constructions, 
        action_chain,
        full_state)

    # Detect if the extracted proof is the one we care about:
    if self.opposite_angle_check.found(theorem_premise, goal_objects):
      return  # Nothing to look at here.
    found_thales = self.thales_check.found(theorem_premise, goal_objects)

    # Now we loop through actions in the proof steps
    proof_actions = [action for action, step in 
                     zip(action_chain, proof_steps)
                     if step != []]

    # Here we loop through each action in the proof steps
    # for each of them, generate one training example (and augmented ones)
    # and then add the action to the state and move on to the next action.
    for i, action in enumerate(proof_actions):
      # print('Try {}'.format(action.to_str()))
      # The current graph is too big (size = # nodes)
      graph_size = len(theorem_premise.name2obj) + 1
      if graph_size > self.max_state_size:
        break

      # Now we randomly add various amount of redundant actions
      # into the problem state to augment the training examples
      size_maxed = True
      for example in self.augment_and_serialize(
          theorem_premise, goal_objects, action, full_state, 
          redundant_actions):
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
      action_chain_lib.add_action(theorem_premise, action, full_state)
      # Then move on to next action,

  def augment_and_serialize(
      self, state, goal_objects, action, full_state, 
      redundant_actions):
    state = state.copy()

    # Yield the non augmented version.
    if len(state.name2obj) + 1 > self.max_state_size:
      return  # No example for this one.
    example = self.serialize(state, goal_objects, action)

    if example is None:
      # [r.name for r in action.premise_objects if r not in state.relations and r.name not in state.name2obj]
      # {l.name: (h1.name, h2.name) for l, (h1, h2) in state.line2hps.items()}
      # {hp.name: [p.name for p in ps] for hp, ps in state.hp2points.items()}
      # {l.name: ([p.name for p in p1], [p.name for p in p2]) for l, (p1, p2) in canvas.line2hps.items()}
      print('Cannot apply action {}'.format(action.to_str()))
      print('State: {}'.format(state.to_str(join='\n')))
      raise ValueError('Cannot apply action {}'.format(action.to_str()))

    yield example

    # Add redundant action until reach full size
    np.random.shuffle(redundant_actions)
    for red_action in redundant_actions:
      # print(' * Try ', red_action.to_str())
      action_chain_lib.add_action(state, red_action, full_state)
      if len(state.name2obj) + 1 > self.max_state_size:
        break
      example = self.serialize(state, goal_objects, action)
      if example:
        yield example

  def serialize(self, state, goal_objects, action):
    """Turn state, goal, action into np arrays."""
    assert len(state.name2obj) + 1 <= self.max_state_size
    match = {y: action.mapping[y]
             for x, y in action.theorem.names.items()}
    action_gen = action.theorem.match_from_input_mapping(
      state, match)

    try:
      action_gen.next()
    except StopIteration:
      match = {x: action.mapping[y].name
               for x, y in action.theorem.names.items()}
      # It is possible that adding redundant actions breaks the
      # applicability of the proof.
      if FLAGS.verbose:
        print(match)
        print('Failed matching {} {}'.format(action.theorem.name, match))
      return None

    state_obj_ids, _, obj2idx, attention_mask = data_gen_lib.serialize_state(state)
    return data_gen_lib.build_example(
        action, state_obj_ids, goal_objects, attention_mask, obj2idx)


if __name__ == '__main__':
  np.random.seed(int(time.time() % 42949671) * 100 + FLAGS.explore_worker_id)
  state, canvas, predefined_steps = action_chain_lib.init_by_normal_triangle()
  # state, canvas, predefined_steps = action_chain_lib.init_by_thales()
  explorer = ExplorationBackoffDFS(
      state, canvas, FLAGS.out_dir, predefined_steps)

  if FLAGS.enable_profiling:
    profiling.enable()
  else:
    profiling.disable()

  if FLAGS.mode == 'datagen':
    explorer.explore(FLAGS.pdb)
  if FLAGS.mode == 'interactive':
    explorer.explore_interactive([], state, canvas, mode='theorem_input')