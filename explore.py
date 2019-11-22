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
import trieu_graph_match

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

FLAGS = tf.compat.v1.flags.FLAGS


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
               init_action_chain,
               max_construction=5,
               max_depth=30,
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

    self.proof_extractor = ProofExtractor()
    self.construct_theorems = [
        theorems.all_theorems['mid'],
        theorems.all_theorems['seg_line'],
        theorems.all_theorems['parallel'],
        theorems.all_theorems['line'],
    ]
    self.deduct_theorems = [
        theorems.all_theorems['eq'],
        theorems.all_theorems['.parallel'],
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

  def _recursive_explore(self, action_chain, state, canvas, 
                         do_pdb=False, depth=None):
    """DFS."""
    if depth is None:
      depth = len(action_chain) + 1

    if depth > self.max_depth:
      depth0 = len(self.init_action_chain) + 1
      x = np.arange(depth0, depth)
      backoff = np.random.choice(x, p=x[::-1]*1.0/np.sum(x))
      print('Reach max depth ', depth, ' backoff = ', backoff)
      # import pdb; pdb.set_trace()
      return backoff

    for action in self.random_action(state, depth, canvas):
      print(' ' * depth, depth, 
            type(action.theorem).__name__, action.duration)
      # print(' * find ', time.time() - start_time)
      # if time.time() - start_time > self.timeout:
      #   x = np.arange(1, depth)
      #   backoff = np.random.choice(x, p=x[::-1]*1.0/np.sum(x))
      #   return backoff
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

      lengths = self.proof_extractor.collect_proof(
          action_chain, self.init_state, self.init_canvas, 
          new_canvas, do_pdb)
      if sum(lengths):
        self.proof_count += sum(lengths)
        print('\n >>> {}\n'.format(self.proof_count))

      backoff = self._recursive_explore(
          action_chain, new_state, new_canvas, do_pdb, depth+1)
      action_chain.pop(-1)

      if backoff < depth:
        return backoff

    if depth == 1:
      # Out of option at depth = 1, do it again.
      print('Out of option at depth 1, start a new Backoff DFS.')
      self.explore()
    else:
      depth0 = len(self.init_action_chain) + 1
      x = np.arange(depth0, depth)
      backoff = np.random.choice(x, p=x[::-1]*1.0/np.sum(x))
      print('Out of option at depth ', depth, ' backoff = ', backoff)
      # import pdb; pdb.set_trace()
      return backoff


class ProofReservoir(object):

  def __init__(self, max_store=1000):
    self._store = []
    self._max_store = max_store

  def add(self, proof):
    self._store.append(proof)
    if len(self._store) == self._max_store:
      self.dump()

  def dump(self):
    pass


def is_quantitative_construction(construction):
  if len(construction) < 3:
    return False

  if not isinstance(construction[0],
                    (LineDirection, SegmentLength)):
    return False
  return True



class ProofExtractor(object):

  def __init__(self):
    self._reservoirs = {}

  def collect_proof(self, action_chain, init_state, init_canvas, 
                    canvas, do_pdb=False):
    new_action = action_chain[-1]
    if not isinstance(new_action.theorem, 
                      (theorems.ASA, theorems.SAS, theorems.SSS, 
                       theorems.ParallelBecauseCorrespondingAngles)):
      return []

    # Extract all value -> value_rel
    val2rels = {}
    for obj in action_chain[-1].new_objects:
      if isinstance(obj, value_rels):
        val = obj.init_list[1]
        if val not in val2rels:
          val2rels[val] = []
        val2rels[val].append(obj)

    all_lengths = []
    for val in val2rels:
      rels = val2rels[val]
      if len(rels) != 2:
        continue

      problem_queue = [r.init_list[0] for r in rels]
      proof_queue = [(val,) + tuple(rels)]

      problem_constructions = whittle_from(
          list(problem_queue), action_chain)
      proof_whittled = whittle_from(
          list(proof_queue), action_chain, 
          problem_queue, problem_constructions)

      for i, p in enumerate(proof_whittled):
        if not (p == [] or p == True):
          if problem_constructions[i] != True:
            problem_constructions[i] += p
          proof_whittled[i] = []

      problem_state = init_state.copy()
      problem_canvas = init_canvas.copy()

      for step, action in zip(problem_constructions, action_chain):
        if step == []:
          continue
        if step == True:
          problem_state.add_relations(action.conclusion_objects)
        else:
          all_constructions = sum(step, [])
          problem_state.add_relations(all_constructions)

      info = {}
      for name, obj in problem_state.name2obj.items():
        if isinstance(obj, Point):
          problem_canvas.update_point(obj, canvas.points[obj])
        elif isinstance(obj, Line):
          problem_canvas.update_line(obj, canvas.lines[obj])
        elif isinstance(obj, Circle):
          problem_canvas.circles[obj] = canvas.circles[obj]
      problem_state.add_spatial_relations(problem_canvas.line2hps)

      length = sum([1 for x in proof_whittled if x != []])
      all_lengths.append(length)

      if length >= 1:
        print()
        print(action.theorem.name, length)
        for i, (action, s) in enumerate(zip(action_chain, problem_constructions)):
          duration = action.duration
          if s == True:
            print(i + 1, action.to_str(), duration)
          elif s:
            print(i + 1, action.theorem.name, [r.name for r in sum(s, [])], duration)
        print('----------', [r.name for r in rels])
        for i, (action, s) in enumerate(zip(action_chain, proof_whittled)):
          duration = action.duration
          if s == True:
            print(i + 1, action.to_str(), duration)
          elif s:
            print(i + 1, action.theorem.name, [r.name for r in sum(s, [])], duration)
        if do_pdb:
          import pdb; pdb.set_trace()

    return all_lengths


value_entity = (
    AngleMeasure, SegmentLength, LineDirection
)

value_rels = (
    AngleHasMeasure, SegmentHasLength, LineHasDirection
)


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
      positions = val.dependency_path(obj1, obj2)
      dependents = [pos for pos in positions if pos is not None]
      # if dependents == []:
      #   import pdb; pdb.set_trace()
      dependents += [obj1, obj2]
      queue.extend(dependents)
      # {x.name: {a.name: b for a, b in y.items()} for x, y in val.edges.items()}
      # print('{} {} <= {}'.format(rel1.name, rel2.name, dependents))
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


class Proof(object):

  def __init__(self, whittled_state_pos, full_state_actions, 
               proof_actions, goal_objects):
    self.whittled_state_pos = whittled_state_pos
    self.full_state_actions = full_state_actions
    self.proof_actions = proof_actions
    self.goal_objects = goal_objects
    self.length = len(proof_actions)

  def create_state(self, action_chain):
    pass


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


def execute_steps(steps, state, canvas):
  action_chain = []

  for i, (theorem, command) in enumerate(steps):
    print(i + 1, ' ', type(theorem).__name__, command)
    name_maps = [c.split('=') for c in command.split()]
    mapping = {theorem.names[a]: _find(state, b) for a, b in name_maps}
    action_gen = theorem.match_from_input_mapping(state, mapping)

    try:
      action = action_gen.next()
    except StopIteration:
      raise ValueError('Matching not found {} {}'.format(theorem, command))

    action.set_chain_position(i)
    action_chain.append(action)

    # print('\tAdd : {}'.format([obj.name for obj in action.new_objects]))
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
      [A, B, C, ab, bc, ca, AB, BC, CA] +
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
  state, canvas, action_chain = init_by_normal_triangle()
  # state, canvas, action_chain = init_by_thales()
  explorer = ExplorationBackoffDFS(state, canvas, action_chain)
  explorer.explore(FLAGS.pdb)
  # explorer.explore_interactive([], state, canvas, mode='theorem')

