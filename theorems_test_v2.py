from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict

import time
import numpy as np

import theorems_utils
import trieu_graph_match
import geometry
import whittling
import action_chain_lib
import theorems
import sketch

import profiling

from theorems_utils import segment_def, collinear, fangle_def
from theorems_utils import have_direction, have_measure, have_length
from theorems_utils import divides_halfplanes, distinct

from state import State, Conclusion

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle, TransitiveRelation
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, LineHasDirection
from geometry import PointEndsSegment, LineBordersHalfplane
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


all_theorems = theorems.theorem_from_short_name


def test_sss_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, CA = map(Segment, 'AB CA'.split())

  init_state.add_relations(
      segment_def(AB, A, B) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(ca, C, A) +
      have_length('1m', AB, CA)
  )

  init_state.add_spatial_relations(
      init_canvas.add_isosceles_triangle(A, B, C, ab, bc, ca))

  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = ['SSS: B A C C A B']

  print('\nRunning SSS isosceles test:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)


def test_asa_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  ab_hp, bc_hp, ca_hp = map(HalfPlane, 'ab_hp bc_hp ca_hp'.split())
  d_ab, d_bc, d_ca = LineDirection('d_ab'), LineDirection('d_bc'), LineDirection('d_ca')

  B_xx, B_xo, B_def = fangle_def(d_ab, d_bc)
  C_xx, C_xo, C_def = fangle_def(d_ca, d_bc)

  init_state.add_relations(
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      divides_halfplanes(ab, ab_hp, p1=C) +
      divides_halfplanes(ca, ca_hp, p1=B) +
      divides_halfplanes(bc, bc_hp, p1=A) +
      have_direction(d_ab, ab) +
      have_direction(d_ca, ca) +
      have_direction(d_bc, bc) +
      B_def + C_def
  )

  line2points = init_canvas.add_isosceles_triangle(A, B, C, ab, bc, ca)
  init_state.add_spatial_relations(line2points)
  init_canvas.update_hps(init_state.line2hps)

  B_angle = B_xx if ab_hp.sign == bc_hp.sign else B_xo
  C_angle = C_xx if ca_hp.sign == bc_hp.sign else C_xo

  init_state.add_relations(
      have_measure('m1', B_angle, C_angle))

  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = ['ASA: C B A B C ab ca']

  print('\nRunning ASA isosceles test:')
  action_chain_lib.execute_steps(steps, state, canvas)


def test_sas_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, CA = map(Segment, 'AB CA'.split())

  init_state.add_relations(
      segment_def(AB, A, B) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(ca, C, A) +
      have_length('1m', AB, CA)
  )

  line2points = init_canvas.add_isosceles_triangle(A, B, C, ab, bc, ca)
  init_state.add_spatial_relations(line2points)
  init_canvas.update_hps(init_state.line2hps)

  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = ['SAS: A B C A C B']

  print('\nRunning SAS isosceles test:')
  action_chain_lib.execute_steps(steps, state, canvas)


def test_angle_bisect_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, CA = map(Segment, 'AB CA'.split())

  ab_hp, ca_hp = map(HalfPlane, 'ab_hp ca_hp'.split())

  init_state.add_relations(
      distinct(A, B, C) +
      distinct(ab, ca) + 
      segment_def(AB, A, B) +
      segment_def(CA, C, A) +
      divides_halfplanes(ab, ab_hp, p1=C) +
      divides_halfplanes(ca, ca_hp, p1=B) +
      collinear(ab, A, B) +
      collinear(ca, C, A) +
      have_length('1m', AB, CA)
  )

  line2points = init_canvas.add_isosceles_triangle(A, B, C, ab, bc, ca)
  init_state.add_spatial_relations(line2points)
  init_canvas.update_hps(init_state.line2hps)

  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = [
      'angle_bisect: ab_hp ca_hp',
      'lineXsegment: l1 B C',
      'SAS: A B P1 A C P1'
  ]

  print('\nRunning test_angle_bisect_isosceles:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)


def test_base_bisect_sss_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, CA = map(Segment, 'AB CA'.split())

  init_state.add_relations(
      # [A, B, C, AB, BC, CA, ab, bc, ca] +
      distinct(A, B, C) +
      segment_def(AB, A, B) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      have_length('1m', AB, CA)
  )

  line2points = init_canvas.add_isosceles_triangle(A, B, C, ab, bc, ca)
  init_state.add_spatial_relations(line2points)
  init_canvas.update_hps(init_state.line2hps)

  init_canvas.add_free_points(bc, B, C)

  state, canvas = init_state.copy(), init_canvas.copy()

  steps = [
      'midp: B C',  # -> P1
      'line: A P1',
      'SSS: B A P1 C A P1'
  ]

  print('\nRunning test_base_bisect_sss_isosceles:')
  action_chain_lib.execute_steps(steps, state, canvas)


def test_ang_isos_outer_bisect_parallel_to_base():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  steps = [
      'ang_isos:',
      'angle_bisect: l1_hp hp3'
  ]

  print('\nRunning test_ang_isos_outer_bisect_parallel_to_base:')
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  assert len([v for v in state.val2valrel 
              if isinstance(v, LineDirection)]) == 3
  assert len([v for v in state.val2valrel 
              if isinstance(v, AngleMeasure)]) == 2


def rewind(action_chain, n=1):
  if n < 1:
    raise ValueError('n must > 0')
  action_chain = list(action_chain)
  state, canvas = None, None
  for _ in range(n):
    action = action_chain.pop(-1)
    state, canvas = action.state, action.canvas
  return state, canvas, action_chain


def test_ang_isos_bisect_is_perp():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3'  # -> l4
  ]

  print('\nRunning test_ang_isos_bisect_is_perp:')
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  assert len([v for v in state.val2valrel 
              if isinstance(v, LineDirection)]) == 4

  assert len([v for v in state.val2valrel 
              if isinstance(v, AngleMeasure)]) == 5
  
  # test that halfpi is amongst the list of angles created.
  halfpi, _ = geometry.get_constant_angles(0.5)
  assert any([halfpi in state.obj2valrel])
  mhalfpi = state.obj2valrel[halfpi].init_list[1]
  assert len(state.val2valrel[mhalfpi]) == 3

  # Test that we cannot create perp line from P1 to l2
  # because l4 has already been figured out that it is perp to l2.
  theorem, mapping = action_chain_lib.extract_theorem_mapping('perp: P1 l2', state)
  action_gen = theorem.match_from_input_mapping(
      state, mapping, randomize=False, canvas=canvas)
  assert len(list(action_gen)) == 0

  steps = [
      'lineXlineA: l4 l2',
      'ASA: P1 P5'
  ]

  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, state, canvas, init_action_chain=action_chain)

  # import pdb; pdb.set_trace()
  assert len([v for v in state.val2valrel 
              if isinstance(v, SegmentLength)]) == 2
  
  # undo asa and lineXline and bisector
  state, canvas, action_chain = rewind(action_chain, 3)
  steps = [
      'perp: P1 l2',  # l5 P6
      'ASA: P1 None None None P6 None None'
  ]

  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, state, canvas, init_action_chain=action_chain)

  assert len([v for v in state.val2valrel 
              if isinstance(v, LineDirection)]) == 4
  assert len([v for v in state.val2valrel 
              if isinstance(v, AngleMeasure)]) == 5
  assert len([v for v in state.val2valrel 
              if isinstance(v, SegmentLength)]) == 2

  # test that halfpi is amongst the list of angles created.
  halfpi, _ = geometry.get_constant_angles(0.5)
  assert any([halfpi in state.obj2valrel])
  mhalfpi = state.obj2valrel[halfpi].init_list[1]
  assert len(state.val2valrel[mhalfpi]) == 3


def test_ang_isos_perp_base_is_bisect():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  steps = [
      'ang_isos:',
      'midp: P2 P3',  # -> P4
      'perp: P4 l2',  # l4
  ]

  print('Running test_ang_isos_perp_base_is_bisect')
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)
  
  measures = [v for v in state.val2valrel 
              if isinstance(v, AngleMeasure)]
  assert len(measures) == 5

  l14xx, l14xo = state.angle_between('l1', 'l4')
  l34xx, l34xo = state.angle_between('l3', 'l4')
  assert state.is_equal(l14xx, l34xx) and state.is_equal(l14xo, l34xo)


def test_isos_merge_lines():
  geometry.reset()

  canvas = sketch.Canvas()
  state = State()

  print('\nRunning Isos Merge test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXlineA: l4 l2',  # -> P4
      'midp: P2 P3',  # -> P5
      'perp: P5 l2',  # -> l5
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, state, canvas)

  halfpi, _ = geometry.get_constant_angles(0.5)
  assert any([halfpi in state.obj2valrel])
  mhalfpi = state.obj2valrel[halfpi].init_list[1]

  state.print_all_equal_angles()
  measure2angles = state.all_equal_angles()
  state.print_all_equal_angles()
  assert len(measure2angles[mhalfpi]) == 9, len(measure2angles[mhalfpi])
  # exit()

  d4 = state.name2obj['d4']
  assert len(state.val2valrel[d4]) == 2

  l5 = state.name2obj['l5']
  P1 = state.name2obj['P1']
  assert not state.has_relation(LineContainsPoint(l5, P1))

  steps = [
      'ASA: P4 P1',
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, state, canvas, init_action_chain=action_chain)
  
  assert len(state.val2valrel[d4]) == 1

  measure2angles = state.all_equal_angles()
  assert len(measure2angles[mhalfpi]) == 5

  assert state.has_relation(LineContainsPoint(l5, P1))


def copy_points_and_lines(constructions, new_canvas, final_canvas):
  new_lines = []
  for c in constructions:
    if isinstance(c, Point):
      new_canvas.update_point(c, final_canvas.points[c])
    elif isinstance(c, Line):
      new_canvas.update_line(c, final_canvas.lines[c])
      new_canvas.line2hps[c] = final_canvas.line2hps[c]
      new_lines.append(c)
    elif isinstance(c, Circle):
      new_canvas.circles[c] = final_canvas.circles[c]

  return {l: [list(n), list(p)] 
          for l, (n, p) in new_canvas.line2points.items()
          if l in new_lines}


def whittle(final_state, state_queue, proof_queue, action_chain, 
            init_state, init_canvas, canvas, verbose=True):
  # Basically shave off any excess from action_chain
  # and crystallize what is relevant as premise & conclusion
  # of a discovered theorem.

  whittled_state = whittling.whittle_from(
      final_state, list(state_queue), action_chain)

  # import pdb; pdb.set_trace()

  proof_whittled = whittling.whittle_from(
      final_state, list(proof_queue), action_chain, 
      goal_objects=state_queue, whittled_state=whittled_state)

  # Transfer stuff from proof_whittled to whittled_state.
  for i, p in enumerate(proof_whittled):
    if not (p == [] or p == True):
      if whittled_state[i] != True:
        whittled_state[i] += p
      proof_whittled[i] = []
  
  if proof_whittled[0] == True:
    whittled_state[0] = True
    proof_whittled[0] = []

  new_state = init_state.copy()
  new_canvas = init_canvas.copy()

  def add_to_state(state, objs_to_add):
    state = state.copy()
    state.add_relations(objs_to_add)

    l2ps = copy_points_and_lines(
        objs_to_add, new_canvas, canvas)
    state.add_spatial_relations(l2ps)
    action.action_eliminate_angle(state, new_canvas, None)
    action.action_eliminate_distance(state, new_canvas, None)
    return state

  all_states = [action.state for action in action_chain] + [final_state]

  if verbose:
    print('\nWhittled state: ')
  for i, (step, action) in enumerate(zip(whittled_state, action_chain)):
    if step == []:
      continue
    if step == True:
      if verbose:
        print(i, action.to_str())
      new_state = add_to_state(new_state, all_states[i+1].inc)
    else:
      # all_constructions = sum(step, [])
      all_constructions = set()
      for x in step:
        all_constructions.update(x)
      all_constructions = list(all_constructions)
      if verbose:
        print('{}. {} : {}'.format(
            i, action.theorem.name,
            [r.name for r in all_constructions]))
      new_state = add_to_state(new_state, all_constructions)

  proof_steps = []
  for i, (step, action) in enumerate(zip(proof_whittled, action_chain)):
    if step == []:
      action = None
    else:
      proof_steps.append(i)

  if verbose:
    print()
    if isinstance(proof_queue[0], (list, tuple)):
      print('Whittled Proof {}'.format(final_state.name_map(proof_queue[0])))
    else:
      print('Whittled Proof {}'.format(final_state.name_map(proof_queue[0])))

    for i, (step, action) in enumerate(zip(proof_whittled, action_chain)):
      if step == []:
        continue
      if step == True:
        print('Apply {}. {}'.format(i, action.to_str()))
      else:
        all_constructions = sum(step, [])
        print('{}. {}'.format(i, [r.name for r in all_constructions]))
    print()
  return new_state, new_canvas, proof_steps


def extract_name2goals(proof_goals, state, prev_state):
  # Check if all the goals are here:
  name2goals = {}
  for state_queue, proof_queue in proof_goals:
    if isinstance(proof_queue[0], tuple):
      repr = proof_queue[0][0]
      if isinstance(repr, geometry.CausalValue):
        rel1, rel2 = proof_queue[0][1:]
        obj1 = rel1.init_list[0]
        obj2 = rel2.init_list[0]
        name = state.name_map(obj1) + ' == ' + state.name_map(obj2)
      else:
        name = prev_state.name_map(proof_queue[0][0])
    else:
      name = prev_state.name_map(proof_queue[0])
    name2goals[name] = state_queue, proof_queue
  return name2goals


def test_isos_merge_whittle_goal1():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning Isos Merge Whittle goal1 test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXlineA: l4 l2',  # -> P4
      'midp: P2 P3',  # -> P5
      'perp: P5 l2',  # -> l5
      'ASA: P4 P1',  # -> Now l5 contains P1
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  # state.print_all_equal_segments()
  prev_state = action_chain[-1].state

  l5 = state.name2obj['l4']
  P1 = state.name2obj['P1']
  assert state.has_relation(LineContainsPoint(l5, P1))

  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = extract_name2goals(proof_goals, state, prev_state)

  all_target_goals = [
      ('l5{P4}', [5]), 
      ('P4[P5P3', [5]), 
      ('P4[P5P2', [5]), 
      ('l4{P5}', [2, 5]), 
      ('l5/l4_hp1', [2, 5]), 
      ('l5/l4_hp2', [2, 5]), 
      ('l5{P1}', [1, 2, 5]), 
      ('l4/l5_hp2', [2, 5]), 
      ('l4/l5_hp1', [2, 5])
  ]
  for goal, correct_proof_steps in all_target_goals:
    assert goal in name2goals, goal
    state_queue, proof_queue = name2goals[goal]
    _, _, proof_steps = whittle(
        state, state_queue, proof_queue, action_chain,
        init_state, init_canvas, canvas, verbose=False)
    print('check whittle({}) = {}'.format(goal, correct_proof_steps))
    assert correct_proof_steps == proof_steps
  
  state_queue, proof_queue = name2goals['l5{P1}']
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)

  assert proof_steps == [1, 2, 5]
  steps = [
      'angle_bisect: hp1 hp3',  # -> l6
      'lineXlineA: l6 l2',  # -> P6
      'ASA: P6 P1',  # -> Now l6 contains P1
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  l5 = proved_problem.name2obj['l5']
  P1 = proved_problem.name2obj['P1']
  assert proved_problem.has_relation(LineContainsPoint(l5, P1))

  last_state = action_chain[-1].state
  l6 = last_state.name2obj['l6']
  assert l6 in l5.merge_graph[proved_problem]['equivalents']


def test_isos_merge_whittle_goal2():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning Isos Merge whittle goal2 test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXlineA: l4 l2',  # -> P4
      'midp: P2 P3',  # -> P5
      'perp: P5 l2',  # -> l5
      'ASA: P4 P1',  # -> Now l5 contains P1, l4 contains P5
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  # state.print_all_equal_segments()
  prev_state = action_chain[-1].state

  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = extract_name2goals(proof_goals, state, prev_state)

  all_target_goals = [
      'l5{P4}', 'P4[P5P3', 'P4[P5P2', 
      'l4{P5}', 'l5/l4_hp1', 'l5/l4_hp2', 
      'l5{P1}', 'l4/l5_hp2', 'l4/l5_hp1']
  for goal in all_target_goals:
    assert goal in name2goals, goal
  
  state_queue, proof_queue = name2goals['l4{P5}']
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)

  assert proof_steps == [2, 5]

  steps = [
      'lineXlineA: l4 l2',  # -> P6
      'ASA: P6 P1',  # -> Now l6 contains P1
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  l4 = proved_problem.name2obj['l4']
  P5 = proved_problem.name2obj['P5']
  assert proved_problem.has_relation(LineContainsPoint(l4, P5))

  last_state = action_chain[-1].state
  P6 = last_state.name2obj['P6']
  assert P6 in P5.merge_graph[proved_problem]['equivalents']


def test_isos_merge_whittle_v2():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning Isos Merge whittle v2 test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXlineA: l4 l2',  # -> P4
      'midp: P2 P3',  # -> P5
      'perp: P5 l2',  # -> l5
      'lineXlineA: l5 l1',  # -> P6
      'ASA: P4 P1',  # -> Now l5 contains P1, l4 contains P5
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  # state.print_all_equal_segments()
  prev_state = action_chain[-1].state

  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = extract_name2goals(proof_goals, state, prev_state)

  all_target_goals = [
      'l4{P6}', 'l3{P6}',
      'l5{P4}', 'P4[P5P3', 'P4[P5P2', 
      'l4{P5}', 'l5/l4_hp1', 'l5/l4_hp2', 
      'l5{P1}', 'l4/l5_hp2', 'l4/l5_hp1']
  for goal in all_target_goals:
    assert goal in name2goals, goal
    state_queue, proof_queue = name2goals[goal]
    problem, problem_canvas, proof_steps = whittle(
        state, state_queue, proof_queue, action_chain,
        init_state, init_canvas, canvas, verbose=False)
  
  state_queue, proof_queue = name2goals['l3{P6}']
  # there will be fragments of 1. construct angle bisector
  # in the whittled problem, but that's okay
  # what we care is the aggregated problem, not its construction
  # on the other hand, proof construction is what we really
  # have to care about.
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)

  l3 = problem.name2obj['l3']
  P6 = problem.name2obj['P6']
  assert not problem.has_relation(LineContainsPoint(l3, P6))

  assert proof_steps == [1, 2, 6], proof_steps

  steps = [
      'angle_bisect: hp1 hp3',  # -> l6
      'lineXlineA: l6 l2',  # -> P7
      'ASA: P7 P1',  # -> Now l5 contains P1, l4 contains P5
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert proved_problem.has_relation(LineContainsPoint(l3, P6))


def test_isos_merge_whittle_v3():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning Isos Merge whittle v3 test:')

  steps = [
      'ang_isos:',
      'midp: P2 P3',  # -> P4
      'perp: P4 l2',  # -> l4
      'lineXlineA: l4 l1',  # -> P5
  ]
  state3, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)
  steps = [
      'ASA: P2 P4',  # -> Now l5 contains P1, l4 contains P5
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, state3, canvas, init_action_chain=action_chain)

  prev_state = action_chain[-1].state
  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = extract_name2goals(proof_goals, state, prev_state)

  l3 = state.name2obj['l3']
  P5 = state.name2obj['P5']
  assert state.has_relation(LineContainsPoint(l3, P5))

  all_target_goals = ['l3{P5}', '4.P2P5 == 4.P3P5', 'l4{P1}']
  for goal in all_target_goals:
    assert goal in name2goals, goal
    state_queue, proof_queue = name2goals[goal]
    problem, problem_canvas, proof_steps = whittle(
        state, state_queue, proof_queue, action_chain,
        init_state, init_canvas, canvas, verbose=False)
  
  state_queue, proof_queue = name2goals['l3{P5}']
  # there will be fragments of 1. construct angle bisector
  # in the whittled problem, but that's okay
  # what we care is the aggregated problem, not its construction
  # on the other hand, proof construction is what we really
  # have to care about.
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)

  assert not problem.has_relation(LineContainsPoint(l3, P5))
  assert proof_steps == [4], proof_steps

  steps = [
      'ASA: P4',
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  P5_equivs = P5.merge_graph[proved_problem]['equivalents']
  P5_equivs_name = map(lambda x: x.name, P5_equivs)
  assert set(P5_equivs_name) == {'P1', 'P7'}
  assert proved_problem.has_relation(LineContainsPoint(l3, P5))


def test_thales():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning test_thales:')

  steps = [
      'triangle:',  # P1 P2 P3
      'midp: P1 P2',  # -> P4
      'parallel: P4 l2',  # -> l4
      'lineXlineD: l4 l3',  # -> P5
      'parallel: P3 l1',  # -> l5
      'line: P4 P3',  # -> l6
      'ASA: P4 P3 P2 P3 P4',  # -> P6
      'ASA: P1 P4 P5 P3 P6'  # P7 == P5
  ]

  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  prev_state = action_chain[-1].state
  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = extract_name2goals(proof_goals, state, prev_state)
  all_target_goals = ['7.P5P4 == 7.P5P6', '7.P1P5 == 7.P3P5']
  for goal in all_target_goals:
    assert goal in name2goals, goal
    state_queue, proof_queue = name2goals[goal]
    problem, problem_canvas, proof_steps = whittle(
        state, state_queue, proof_queue, action_chain,
        init_state, init_canvas, canvas, verbose=False)

  state_queue, proof_queue = name2goals['7.P1P5 == 7.P3P5']
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  
  assert proof_steps == [4, 5, 6, 7], proof_steps
  P1P5 = problem.segment_between('P1', 'P5')
  P3P5 = problem.segment_between('P3', 'P5')
  assert not problem.is_equal(P1P5, P3P5)

  # assert not problem.has_relation(LineContainsPoint(l3, P5))
  # assert proof_steps == [4], proof_steps

  steps = [
      'parallel: P3 l1',  # --> l7
      'line: P4 P3',
      'ASA: P4 P3 P2 P3 P4',  # --> P7
      'ASA: P1 P4 P5 P3 P7',
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)
  assert proved_problem.is_equal(P1P5, P3P5)


def test_thales_noise_shuffle():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning test_thales:')

  steps = [
      'triangle:',  # P1 P2 P3
      'parallel: P1 l2',  # -> l4
      'parallel: P3 l1',  # -> l5
      'midp: P1 P2',  # -> P4
      'line: P4 P3',  # -> l6
      'parallel: P4 l2',  # -> l7
      'midp: P2 P3',  # -> P5
      'lineXlineD: l7 l3',  # -> P6
      'ASA: P4 P3 P2 P3 P4',  # -> P7
      'ASA: P1 P4 P6 P3 P7'  # P7 == P5
  ]

  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  prev_state = action_chain[-1].state
  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = extract_name2goals(proof_goals, state, prev_state)
  all_target_goals = ['9.P6P4 == 9.P6P7', '9.P1P6 == 9.P3P6']
  for goal in all_target_goals:
    assert goal in name2goals, goal
    state_queue, proof_queue = name2goals[goal]
    problem, problem_canvas, proof_steps = whittle(
        state, state_queue, proof_queue, action_chain,
        init_state, init_canvas, canvas, verbose=False)

  state_queue, proof_queue = name2goals['9.P1P6 == 9.P3P6']
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  
  assert proof_steps == [2, 4, 8, 9]
  P1P6 = problem.segment_between('P1', 'P6')
  P3P6 = problem.segment_between('P3', 'P6')
  assert not problem.is_equal(P1P6, P3P6)

  steps = [
      'parallel: P3 l1',  # --> l8
      'line: P4 P3',  # --> l9
      'ASA: P4 P3 P2 P3 P4',  # --> P8
      'ASA: P1 P4 P6 P3 P8',
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)
  assert proved_problem.is_equal(P1P6, P3P6)


def test_thales_noise_shuffle_merge_goal1():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning test_thales:')

  steps = [
      'triangle:',  # P1 P2 P3
      'parallel: P1 l2',  # -> l4
      'parallel: P3 l1',  # -> l5
      'midp: P1 P2',  # -> P4
      'line: P4 P3',  # -> l6
      'parallel: P4 l2',  # -> l7
      'midp: P2 P3',  # -> P5
      'lineXlineD: l7 l3',  # -> P6
      'ASA: P4 P3 P2 P3 P4 l5 l7',  # -> P7
      'midp: P1 P3',  # -> P8
      'parallel: P8 l2',  # l8
      'ASA: P1 P4 P6 P3 P7 l3 l7',  # P7 == P5
  ]

  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  prev_state = action_chain[-1].state
  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = extract_name2goals(proof_goals, state, prev_state)
  print(name2goals.keys())
  # exit()

  all_target_goals = [
      ('11.P1P6 == 11.P3P6', [2, 4, 8, 11]), 
      ('11.P6P4 == 11.P6P7', [11]),
      ('l7{P8}', [2, 4, 7, 8, 11]), 
      ('l8{P4}', [2, 4, 5, 7, 8, 11]), 
      ('l8{P6}', [2, 4, 8, 11]), 
      ('l8{P7}', [7, 11])
  ]
  for goal, target_proof_steps in all_target_goals:
    assert goal in name2goals, goal
    state_queue, proof_queue = name2goals[goal]
    problem, problem_canvas, proof_steps = whittle(
        state, state_queue, proof_queue, action_chain,
        init_state, init_canvas, canvas, verbose=False)
    assert target_proof_steps == proof_steps

  state_queue, proof_queue = name2goals['11.P1P6 == 11.P3P6']
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  
  assert proof_steps == [2, 4, 8, 11]
  P1P6 = problem.segment_between('P1', 'P6')
  P3P6 = problem.segment_between('P3', 'P6')
  assert not problem.is_equal(P1P6, P3P6)

  steps = [
      'parallel: P3 l1',  # --> l9
      'line: P4 P3',  # --> l10
      'ASA: P4 P3 P2 P3 P4',  # --> P9
      'ASA: P1 P4 P6 P3 P9',
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)
  assert proved_problem.is_equal(P1P6, P3P6)


def test_thales_noise_shuffle_merge_goal2():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning test_thales:')

  steps = [
      'triangle:',  # P1 P2 P3
      'parallel: P1 l2',  # -> l4
      'parallel: P3 l1',  # -> l5
      'midp: P1 P2',  # -> P4
      'line: P4 P3',  # -> l6
      'parallel: P4 l2',  # -> l7
      'midp: P2 P3',  # -> P5
      'lineXlineD: l7 l3',  # -> P6
      'ASA: P4 P3 P2 P3 P4 l5 l7',  # -> P7
      'midp: P1 P3',  # -> P8
      'parallel: P8 l2',  # l8
      'ASA: P1 P4 P6 P3 P7 l3 l7',  # P7 == P5
  ]

  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  prev_state = action_chain[-1].state
  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = extract_name2goals(proof_goals, state, prev_state)
  
  state_queue, proof_queue = name2goals['l8{P4}']
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  
  assert proof_steps == [2, 4, 5, 7, 8, 11]
  l8 = problem.name2obj['l8']
  P4 = problem.name2obj['P4']
  assert not problem.has_relation(LineContainsPoint(l8, P4))

  steps = [
      'parallel: P3 l1',  # --> l9
      'line: P4 P3',  # --> l10
      'parallel: P4 l2',  # l11
      'lineXlineD: l11 l3',  # P9
      'ASA: P4 P3 P2 P3 P4',  # --> P10
      'ASA: P1 P4 P9 P3 P10',
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)
  assert proved_problem.has_relation(LineContainsPoint(l8, P4))


if __name__ == '__main__':
  np.random.seed(int(time.time()))

  profiling.enable()

  t = time.time()
  # Self-congruences:
  test_sss_isosceles()
  test_asa_isosceles()
  test_sas_isosceles()
  
  # Aux point/lines
  test_angle_bisect_isosceles()
  test_base_bisect_sss_isosceles()

  # Test uniqueness of direction.
  test_ang_isos_outer_bisect_parallel_to_base()
  
  # Test gaussian elimination engine.
  test_ang_isos_bisect_is_perp()
  test_ang_isos_perp_base_is_bisect()
  test_isos_merge_lines()

  # Test Whittling with merges
  test_isos_merge_whittle_goal1()
  test_isos_merge_whittle_goal2()
  test_isos_merge_whittle_v2()
  test_isos_merge_whittle_v3()

  # Test thales theorems & proof whittling
  test_thales()
  test_thales_noise_shuffle()
  test_thales_noise_shuffle_merge_goal1()
  test_thales_noise_shuffle_merge_goal2()

  # TODO(thtrieu): think about reapplying facts across symmetries in the graph.

  # TODO(thtrieu): IMO 2018 SL G5
  # for all tricky concepts in their full glory: 
  # length/angle ratios, repeated symmetry.

  # TODO(thtrieu): IMO 2018 SL G7

  # TODO(thtrieu): IMO 2017 SL G1

  print('\n [OK!]')
  profiling.print_records()
  print(time.time()-t)