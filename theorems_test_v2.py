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
import explore
import debugging
import state

from theorems_utils import *
from theorems import *

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

  print('\nRunning bisector isosceles test:')
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

  print('\nRunning bisector SSS isosceles test:')
  action_chain_lib.execute_steps(steps, state, canvas)


def test_ang_isos_outer_bisect_parallel_to_base():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  steps = [
      'ang_isos:',
      'angle_bisect: l1_hp hp3'
  ]

  print('\nRunning Angle Isosceles test:')
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

  print('\nRunning Angle Isosceles test:')
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
      'lineXline: l4 l2',
      'ASA: P1 P5'
  ]

  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, state, canvas, init_action_chain=action_chain)

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


def test_isos_merge_lines():
  geometry.reset()

  canvas = sketch.Canvas()
  state = State()

  print('\nRunning Isos Merge test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXline: l4 l2',  # -> P4
      'midp: P2 P3',  # -> P5
      'perp: P5 l2',  # -> l5
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, state, canvas)

  halfpi, _ = geometry.get_constant_angles(0.5)
  assert any([halfpi in state.obj2valrel])
  mhalfpi = state.obj2valrel[halfpi].init_list[1]

  measure2angles = state.all_equal_angles()
  assert len(measure2angles[mhalfpi]) == 9

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
  proof_whittled = whittling.whittle_from(
      final_state, list(proof_queue), action_chain, 
      goal_objects=state_queue, whittled_state=whittled_state)

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

  if verbose:
    print('\nWhittled state: ')
  for i, (step, action) in enumerate(zip(whittled_state, action_chain)):
    if step == []:
      continue
    if step == True:
      if verbose:
        print(i, action.to_str())
      new_state = add_to_state(new_state, action.conclusion_objects)
    else:
      all_constructions = sum(step, [])
      # all_constructions = set()
      # for x in step:
      #   all_constructions.update(x)
      # all_constructions = list(all_constructions)
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
      print('Whittled Proof {}'.format([r.name for r in proof_queue[0]]))
    else:
      print('Whittled Proof {}'.format(proof_queue[0].name))

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


def test_isos_merge_whittle():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning Isos Merge test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXline: l4 l2',  # -> P4
      'midp: P2 P3',  # -> P5
      'perp: P5 l2',  # -> l5
      'ASA: P4 P1',  # -> Now l5 contains P1
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  # state.print_all_equal_segments()
  prev_state = action_chain[-1].state

  l5 = state.name2obj['l5']
  P1 = state.name2obj['P1']
  assert state.has_relation(LineContainsPoint(l5, P1))

  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = {}
  for state_queue, proof_queue in proof_goals:
    if isinstance(proof_queue[0], tuple):
      name = prev_state.name_map(proof_queue[0][0])
    else:
      name = prev_state.name_map(proof_queue[0])
    name2goals[name] = state_queue, proof_queue

  all_target_goals = [
      'l5{P4}', 'P4[P5P3', 'P4[P5P2', 
      'l4{P5}', 'l5/l4_hp1', 'l5/l4_hp2', 
      'l5{P1}', 'l4/l5_hp2', 'l4/l5_hp1']
  for goal in all_target_goals:
    assert goal in name2goals, goal
  
  state_queue, proof_queue = name2goals['l5{P1}']
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)

  assert proof_steps == [1, 2, 5]
  steps = [
      'angle_bisect: hp1 hp3',  # -> l6
      'lineXline: l6 l2',  # -> P6
      'ASA: P6 P1',  # -> Now l6 contains P1
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  l6 = proved_problem.name2obj['l6']
  P1 = proved_problem.name2obj['P1']
  assert proved_problem.has_relation(LineContainsPoint(l6, P1))

  last_state = action_chain[-1].state
  l5 = last_state.name2obj['l5']
  assert l5 in l6.merge_graph[proved_problem]['equivalents']


def test_isos_merge_whittle_v2():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning Isos Merge test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXline: l4 l2',  # -> P4
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
  name2goals = {}
  for state_queue, proof_queue in proof_goals:
    if isinstance(proof_queue[0], tuple):
      name = prev_state.name_map(proof_queue[0][0])
    else:
      name = prev_state.name_map(proof_queue[0])
    name2goals[name] = state_queue, proof_queue

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
      'lineXline: l4 l2',  # -> P6
      'ASA: P6 P1',  # -> Now l6 contains P1
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  l4 = proved_problem.name2obj['l4']
  P6 = proved_problem.name2obj['P6']
  assert proved_problem.has_relation(LineContainsPoint(l4, P6))

  last_state = action_chain[-1].state
  P5 = last_state.name2obj['P5']
  assert P5 in P6.merge_graph[proved_problem]['equivalents']


def test_isos_merge_whittle_v3():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning Isos Merge test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXline: l4 l2',  # -> P4
      'midp: P2 P3',  # -> P5
      'perp: P5 l2',  # -> l5
      'lineXline: l5 l1',  # -> P6
      'ASA: P4 P1',  # -> Now l5 contains P1, l4 contains P5
  ]
  state, canvas, action_chain = action_chain_lib.execute_steps(
      steps, init_state, init_canvas)

  # state.print_all_equal_segments()
  prev_state = action_chain[-1].state

  proof_goals = list(whittling.extract_all_proof_goals(action_chain, state))
  
  # Check if all the goals are here:
  name2goals = {}
  for state_queue, proof_queue in proof_goals:
    if isinstance(proof_queue[0], tuple):
      name = prev_state.name_map(proof_queue[0][0])
    else:
      name = prev_state.name_map(proof_queue[0])
    name2goals[name] = state_queue, proof_queue
  
  print(name2goals.keys())

  all_target_goals = [
      'l4{P6}', 'l3{P6}',
      'l5{P4}', 'P4[P5P3', 'P4[P5P2', 
      'l4{P5}', 'l5/l4_hp1', 'l5/l4_hp2', 
      'l5{P1}', 'l4/l5_hp2', 'l4/l5_hp1']
  for goal in all_target_goals:
    assert goal in name2goals, goal
  
  state_queue, proof_queue = name2goals['l3{P6}']
  problem, problem_canvas, proof_steps = whittle(
      state, state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)

  assert proof_steps == [1, 2, 6], proof_steps

  steps = [
      'lineXline: l4 l2',  # -> P6
      'ASA: P6 P1',  # -> Now l6 contains P1
  ]
  print('Proof execution:')
  proved_problem, _, action_chain = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  l4 = proved_problem.name2obj['l4']
  P6 = proved_problem.name2obj['P6']
  assert proved_problem.has_relation(LineContainsPoint(l4, P6))

  last_state = action_chain[-1].state
  P5 = last_state.name2obj['P5']
  assert P5 in P6.merge_graph[proved_problem]['equivalents']


def test_isos_merge_whittle_all():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  print('\nRunning Isos Merge test:')

  steps = [
      'ang_isos:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXline: l4 l2',  # -> P4
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
  name2goals = {}
  for state_queue, proof_queue in proof_goals:
    if isinstance(proof_queue[0], tuple):
      name = prev_state.name_map(proof_queue[0][0])
    else:
      name = prev_state.name_map(proof_queue[0])
    name2goals[name] = state_queue, proof_queue

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


def test_thales():
  steps = [
      'triangle:',
      'angle_bisect: hp1 hp3',  # -> l4
      'lineXline: l4 l2',  # -> P4
      'midp: P2 P3',  # -> P5
      'perp: P5 l2',  # -> l5
      'ASA: P4 P1',  # -> Now l5 contains P1, l4 contains P5
  ]


if __name__ == '__main__':
  np.random.seed(1234)
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
  test_isos_merge_lines()

  # Test Whittling with merges
  test_isos_merge_whittle()
  test_isos_merge_whittle_v2()
  test_isos_merge_whittle_all()
  # test_isos_merge_whittle_v3()

  # TODO(thtrieu): test thales theorems & proof whittling

  # TODO(thtrieu): think about reapplying facts in the current tree.

  print('\n [OK!]')
  print('Total time = {}'.format(time.time()-t))