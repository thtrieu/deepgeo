from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from theorems_utils import *
from theorems import *

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


def print_match(match, types=None, filter_fn=None):
  s = []
  for premise_obj, state_obj in sorted(match.items()):
    if types and type(premise_obj) not in types:
      continue
    if filter_fn and not filter_fn(premise_obj):
      continue
    s_ = '{}::{}, '.format(premise_obj.name, state_obj.name)
    s += [s_]
  print('   '.join(sorted(s)))


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


def state_merge_and_copy():
  AB, BC, CA, XY = map(Segment, ['AB', 'BC', 'CA', 'XY'])
  s = time.time()
  state = State()
  state.add_relations(
      have_length('l1', AB, BC, CA) +
      have_length('l2', BC, XY)
  )
  print(time.time() - s)

  # [print(r.name) for r in state.relations]
  all_vals = [r.init_list[1] for r in state.relations]
  assert all_vals[1:] == all_vals[:-1]

  s = time.time()
  state = State()
  state.add_relations(
      have_length('l1', AB, BC) +
      have_length('l2', CA, XY) +
      have_length('l3', AB, XY)
  )
  print(time.time() - s)

  # [print(r.name) for r in state.relations]
  all_vals = [r.init_list[1] for r in state.relations]
  assert all_vals[1:] == all_vals[:-1]

  state = State()
  state.add_relations(
      have_length('l1', AB, BC) +
      have_length('l2', CA, XY)
  )

  state2 = state.copy()
  state2.add_relations(
      have_length('l3', AB, XY)
  )
  # [print(r.name) for r in state2.relations]
  all_vals = [r.init_list[1] for r in state2.relations]
  assert all_vals[1:] == all_vals[:-1]

  # [print(r.name) for r in state.relations]
  all_vals = [r.init_list[1] for r in state.relations]
  assert not all_vals[1:] == all_vals[:-1]


def time_sas():
  state = State()

  A, B, C = map(Point, 'ABC')
  AB, BC, CA = map(Segment, 'AC BC CA'.split())
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  ab_hp, bc_hp, ca_hp = map(HalfPlane, 'ab_hp bc_hp ca_hp'.split())
  ABC, BCA, CAB = map(Angle, 'ABC BCA CAB'.split())

  state.add_relations(
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      segment_def(AB, A, B) +
      segment_def(BC, B, C) +
      segment_def(CA, C, A) +
      have_length('1m', AB, CA) +
      have_length('2m', BC) +
      divides_halfplanes(ab, ab_hp, p1=C) +
      divides_halfplanes(bc, bc_hp, p1=A) +
      divides_halfplanes(ca, ca_hp, p1=B) +
      angle_def(ABC, ab_hp, bc_hp) +
      angle_def(BCA, bc_hp, ca_hp) +
      angle_def(CAB, ca_hp, ab_hp) +
      # have_measure('1"', ABC, BCA) +
      have_measure('2"', CAB)
  )
  
  theorem = theorems.SAS()
  premise = theorem.premise
  conclusion = theorem.conclusion

  matches = trieu_graph_match.match_relations(
      premise, state.relations,
      conclusion=theorem.conclusion, 
      randomize=False, distinct=theorem.distinct,
      match_all=True)
  start = time.time()
  matches = list(matches)
  print(time.time() - start)
  for _, m in matches:
    print_match(m, [Point])
  assert len(matches) == 2, len(matches)


def sas():
  state = State()
  A, B, C, D = map(Point, 'ABCD')
  AC, BC, CD = map(Segment, 'AC BC CD'.split())
  l1, l2 = map(Line, 'l1 l2'.split())
  l1_hp1, l2_hp1, l2_hp2 = map(HalfPlane, 'l1_hp1 l2_hp1 l2_hp2'.split())
  ACD, BCD = map(Angle, 'ACD BCD'.split())

  state.add_relations(
      collinear(l1, A, B, C) +
      collinear(l2, C, D) +
      segment_def(AC, A, C) +
      segment_def(BC, B, C) +
      segment_def(CD, C, D) +
      have_length('1m', AC, BC) +
      have_length('2m', CD) +
      divides_halfplanes(l1, l1_hp1, p1=D) +
      divides_halfplanes(l2, l2_hp1, l2_hp2, A, B) +
      angle_def(ACD, l1_hp1, l2_hp1) +
      angle_def(BCD, l1_hp1, l2_hp2) +
      have_measure('1"', ACD, BCD)
  )
  
  theorem = theorems.SAS()
  premise = theorem.premise
  conclusion = theorem.conclusion

  start = time.time()
  matches = trieu_graph_match.match_relations(
      premise, state.relations,
      conclusion=theorem.conclusion, 
      randomize=False, 
      distinct=theorem.distinct,
      match_all=True)
  matches = list(matches)
  print(time.time() - start)
  for _, m in matches:
    print_match(m, [Point])
  assert len(matches) == 2, len(matches)

  state_conclusion, match = matches[0]
  state.add_relations(
      sum(state_conclusion.topological_list, [])
  )

  start = time.time()
  matches = trieu_graph_match.match_relations(
      premise, state.relations,
      conclusion=theorem.conclusion, 
      randomize=False, distinct=theorem.distinct)
  matches = list(matches)
  print(time.time() - start)
  assert len(matches) == 0


def conclusion_match():
  A, B = Point('A'), Point('B')
  l = Line('ab')
  AB = Segment('AB')


  state_candidates = {
      SegmentHasLength: [],
      PointEndsSegment: [],
      LineContainsPoint: [LineContainsPoint(l, A),
                          LineContainsPoint(l, B)]
  }

  state_relations = sum([y for x, y in state_candidates.items()], [])

  X, Y, Z, T = map(Point, 'XYZT')
  XY = Segment('XY')
  ZT = Segment('ZT')

  premise_match = {X: A, Y: B, Z: B, T: A}

  conclusion = Conclusion()
  conclusion.add(*segment_def(XY, X, Y))
  conclusion.add(*segment_def(ZT, Z, T))
  conclusion.add(*have_length('1m', XY, ZT))
  conclusion.gather_val2objs()

  state_conclusion, match = trieu_graph_match.match_conclusions(
      conclusion, 
      dict(state_candidates), 
      dict(premise_match),
      state_relations, 
      distinct=[(X, Y)])

  constructions = state_conclusion.topological_list
  assert len(constructions) == 2
  _, [length, haslength1, haslength2] = constructions
  assert haslength1.init_list == haslength2.init_list
  print_construction(constructions)

  state_candidates.update({
      SegmentHasLength: have_length('2m', AB),
      PointEndsSegment: segment_def(AB, A, B)})

  conclusion, match = trieu_graph_match.match_conclusions(
      conclusion, 
      dict(state_candidates), 
      dict(premise_match), 
      state_relations,
      distinct=[(X, Y)])
  assert conclusion.topological_list == []
  print_construction(conclusion.topological_list)


def def_to_state(state_def):
  name2obj = {}
  state = State()
  for rel_type, obj1_type, obj1_name, obj2_type, obj2_name in state_def:
    if obj1_name in name2obj:
      obj1 = name2obj[obj1_name]
    else:
      obj1 = obj1_type(obj1_name)
      name2obj[obj1_name] = obj1

    if obj2_name in name2obj:
      obj2 = name2obj[obj2_name]
    else:
      obj2 = obj2_type(obj2_name)
      name2obj[obj2_name] = obj2

    state.add_one(rel_type(obj1, obj2))
  return state, name2obj


def sas_hp():
  A, B, C, D = map(Point, 'ABCD')
  ab, da, dc = map(Line, 'ab da dc'.split())
  BA, BC, DA, DC = map(Segment, 'BA BC DA DC'.split())
  DAB, DCB = Angle('DAB'), Angle('DCB')

  ab_hp, da_hp, dc_hp = map(HalfPlane, 'ab_hp da_hp dc_hp'.split())

  state = State()
  state.add_relations(
      collinear(ab, A, B, C) +
      collinear(da, D, A) +
      collinear(dc, D, C) +
      divides_halfplanes(ab, ab_hp, p1=D) +
      divides_halfplanes(da, da_hp, p1=B) +
      divides_halfplanes(dc, dc_hp, p1=B) +
      segment_def(BA, B, A) +
      segment_def(BC, B, C) +
      have_length('1m', BA, BC) +
      segment_def(DA, D, A) +
      segment_def(DC, D, C) +
      have_length('2m', DA, DC) +
      angle_def(DAB, ab_hp, da_hp) +
      angle_def(DCB, ab_hp, dc_hp) +
      have_measure('1"', DAB, DCB)
  )

  theorem = theorems.SAS()
  premise = theorem.premise
  conclusion = theorem.conclusion

  matches = trieu_graph_match.match_relations(
      premise_relations=theorem.premise, 
      state_relations=state.relations,
      conclusion=theorem.conclusion,
      randomize=True,
      distinct=theorem.distinct)

  c, match = matches.next()
  state.add_relations(sum(c.topological_list, []))


def triangle_seed():
  init_canvas = sketch.Canvas()
  init_state = theorems_utils.State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, BC, CA = map(Segment, 'AB BC CA'.split())

  init_state.add_relations(
      # [A, B, C, AB, BC, CA, ab, bc, ca] +
      segment_def(AB, A, B) +
      segment_def(BC, B, C) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A)
  )

  init_state.add_spatial_relations(
      init_canvas.add_triangle(A, B, C, ab, bc, ca))

  return init_state, init_canvas


used_theorems = theorems.all_theorems


def test_thales():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

  print('Running thales:')
  steps = [
      (used_theorems['mid'], 'A=A B=B'),  # P1
      (used_theorems['parallel'], 'A=P1 l=bc'),  # l1
      (used_theorems['seg_line'], 'l=l1 A=A B=C'),  # P1
      # -------------------------------------------------------
      (used_theorems['parallel'], 'A=C l=ab'),  # l2
      (used_theorems['line'], 'A=P1 B=C'),  # l3
      # -------------------------------------------------------
      (used_theorems['eq'], 'l=l3 l1=ab l2=l2'),
      (used_theorems['eq'], 'l=l3 l1=l1 l2=bc'),
      (used_theorems['asa'], 'A=P1 B=B C=C D=C F=P1 de=l2 ef=l1'),  # P3
      (used_theorems['eq'], 'l=ca l1=ab l2=l2'),
      (used_theorems['eq'], 'l=l1 l1=ab l2=l2'),
      (used_theorems['asa'], 'A=A B=P2 C=P1 D=C F=P3 de=ca ef=l1'),
  ]

  s = time.time()
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  print('thales exec time ', time.time()-s)
  assert len(state.name2obj) == 70, len(state.name2obj)
  assert len(canvas.points) == 6, len(canvas.points)
  assert len(canvas.lines) == 6, len(canvas.lines)
  assert len(canvas.circles) == 0, len(canvas.circles)
  # print([r.name for r in action_chain[-1].new_objects])
  # s = time.time()
  # action = used_theorems['asa'].match_one_random(state)
  # print(time.time() - s)
  # assert action is None


def test_intersect_line_line():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

  print('Running thales:')
  steps = [
      (used_theorems['parallel'], 'A=A l=bc'),  # l1
      (used_theorems['parallel'], 'A=C l=ab'),  # l2
      (used_theorems['line_line'], 'l1=l1 l2=l2'),  # l3
  ]

  s = time.time()
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  print('line line exec time ', time.time()-s)
  assert len(state.name2obj) == 24, len(state.name2obj)
  assert len(canvas.points) == 4, len(canvas.points)
  assert len(canvas.lines) == 5, len(canvas.lines)
  assert len(canvas.circles) == 0, len(canvas.circles)


def test_intersect_line_line2():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

  print('Running thales:')
  steps = [
      (used_theorems['parallel'], 'A=A l=bc'),  # l1
      (used_theorems['line_line'], 'l1=l1 l2=bc'),  # l3
  ]

  s = time.time()
  try:
    state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  except sketch.InvalidLineIntersect:
    print('Invalid ok.')
    return

  assert False


def get_state_and_proof_objects(last_action, val_name):
  val2rels = {}
  for rel in last_action.conclusion_objects:
    if isinstance(rel, (SegmentHasLength, LineHasDirection, AngleHasMeasure)):
      obj, val = rel.init_list
      if val not in val2rels:
        val2rels[val] = []
      val2rels[val].append(rel)

  new_val2rels = {}
  for rel in last_action.new_objects:
    if isinstance(rel, (SegmentHasLength, LineHasDirection, AngleHasMeasure)):
      _, val = rel.init_list
      new_val2rels[val.name] = [val] + val2rels[val]

  queue = new_val2rels[val_name]
  state_queue = [r.init_list[0] for r in queue[1:]]
  proof_queue = [tuple(queue)]
  return state_queue, proof_queue


def test_thales_whittle1():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = [
      (used_theorems['mid'], 'A=A B=B'),  # P1
      (used_theorems['parallel'], 'A=A l=bc'),  # l1  noise
      (used_theorems['parallel'], 'A=P1 l=bc'),  # l2
      (used_theorems['seg_line'], 'l=l2 A=A B=C'),  # P1
      # -------------------------------------------------------
      (used_theorems['parallel'], 'A=C l=ab'),  # l3
      (used_theorems['line'], 'A=P1 B=C'),  # l4
      (used_theorems['parallel'], 'A=B l=ca'),  # l5  noise
      # -------------------------------------------------------
      (used_theorems['eq'], 'l=l4 l1=ab l2=l3'),
      (used_theorems['eq'], 'l=l4 l1=l2 l2=bc'),
      (used_theorems['eq'], 'l=ab l1=l1 l2=bc'),  # noise
      (used_theorems['asa'], 'A=P1 B=B C=C D=C F=P1 de=l3 ef=l2'),  # P3
      (used_theorems['eq'], 'l=ca l1=ab l2=l3'),
      (used_theorems['eq'], 'l=bc l1=ca l2=l5'),  # noise
      (used_theorems['eq'], 'l=l2 l1=ab l2=l3'),
      (used_theorems['asa'], 'A=A B=P2 C=P1 D=C F=P3 de=ca ef=l2'),
  ]

  print('\nRunning thales redundant 1:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)

  # Extract state queue & proof queue that prove P2 is mid AC
  conclusion = action_chain[-1].matched_conclusion
  # queue = list(conclusion.topological_list[-6])
  # queue += conclusion.topological_list[-5]

  state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], '10m')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('thales whittle time ', time.time()-s)

  # Test if we are having the correct problem statement
  assert len(problem_canvas.points) == 5
  assert len(problem_canvas.lines) == 4
  assert len(problem_canvas.circles) == 0
  assert len(problem.name2obj) == 26, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 8
  assert chosen_proof_steps == [4, 5, 7, 8, 10, 11, 13, 14]

  # Test if the correct proof step applied on the problem statement
  # give the correct solution
  print('Proof execution:')
  steps = [
      (used_theorems['parallel'], 'A=C l=ab'),  # l6
      (used_theorems['line'], 'A=P1 B=C'),  # l7
      # -------------------------------------------------------
      (used_theorems['eq'], 'l=l7 l1=ab l2=l6'),
      (used_theorems['eq'], 'l=l7 l1=l2 l2=bc'),
      (used_theorems['asa'], 'A=P1 B=B C=C D=C F=P1 de=l6 ef=l2'),  # P4
      (used_theorems['eq'], 'l=ca l1=ab l2=l6'),
      (used_theorems['eq'], 'l=l2 l1=ab l2=l6'),
      (used_theorems['asa'], 'A=A B=P2 C=P1 D=C F=P4 de=ca ef=l2')
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 6
  assert len(proved_canvas.lines) == 6
  assert len(proved_canvas.circles) == 0
  assert len(proved_problem.name2obj) == 70, len(proved_problem.name2obj)
  # action = used_theorems['asa'].match_one_random(proved_problem)
  # assert action is None


def test_thales_whittle2():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = [
      (used_theorems['mid'], 'A=A B=B'),  # P1
      (used_theorems['parallel'], 'A=P1 l=bc'),  # l1
      (used_theorems['parallel'], 'A=A l=bc'),  # l2  noise
      (used_theorems['seg_line'], 'l=l1 A=A B=C'),  # P1
      # -------------------------------------------------------
      (used_theorems['parallel'], 'A=C l=ab'),  # l3
      (used_theorems['line'], 'A=P1 B=C'),  # l4
      (used_theorems['parallel'], 'A=B l=ca'),  # l5  noise
      # -------------------------------------------------------
      (used_theorems['eq'], 'l=l4 l1=ab l2=l3'),
      (used_theorems['eq'], 'l=l4 l1=l1 l2=bc'),
      (used_theorems['eq'], 'l=ab l1=l2 l2=bc'),  # noise
      (used_theorems['asa'], 'A=P1 B=B C=C D=C F=P1 de=l3 ef=l1'),  # P3
      (used_theorems['eq'], 'l=ca l1=ab l2=l3'),
      (used_theorems['eq'], 'l=bc l1=ca l2=l5'),  # noise
      (used_theorems['eq'], 'l=l1 l1=ab l2=l3'),
      (used_theorems['asa'], 'A=A B=P2 C=P1 D=C F=P3 de=ca ef=l1'),
  ]

  print('\nRunning thales redundant 2:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)

  # Extract state queue & proof queue that prove P2 is mid AC
  state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], '10m')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('thales whittle time ', time.time()-s)

  # Test if we are having the correct problem statement
  assert len(problem_canvas.points) == 5
  assert len(problem_canvas.lines) == 4
  assert len(problem_canvas.circles) == 0
  assert len(problem.name2obj) == 26, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]

  assert len(chosen_proof_steps) == 8
  assert chosen_proof_steps == [4, 5, 7, 8, 10, 11, 13, 14]

  thales_check = theorems.ThalesCheck()
  if not thales_check.found(problem, proof_queue[0]):
    import pdb; pdb.set_trace()

  # Test if the correct proof step applied on the problem statement
  # give the correct solution
  print('Proof execution:')
  steps = [
      (used_theorems['parallel'], 'A=C l=ab'),  # l6
      (used_theorems['line'], 'A=P1 B=C'),  # l7
      # -------------------------------------------------------
      (used_theorems['eq'], 'l=l7 l1=ab l2=l6'),
      (used_theorems['eq'], 'l=l7 l1=l1 l2=bc'),
      (used_theorems['asa'], 'A=P1 B=B C=C D=C F=P1 de=l6 ef=l1'),  # P4
      (used_theorems['eq'], 'l=ca l1=ab l2=l6'),
      (used_theorems['eq'], 'l=l1 l1=ab l2=l6'),
      (used_theorems['asa'], 'A=A B=P2 C=P1 D=C F=P4 de=ca ef=l1')
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 6
  assert len(proved_canvas.lines) == 6
  assert len(proved_canvas.circles) == 0
  assert len(proved_problem.name2obj) == 70, len(proved_problem.name2obj)
  # action = used_theorems['asa'].match_one_random(proved_problem)
  # assert action is None


def test_whittle0():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

  steps = [
      (used_theorems['parallel'], 'A=B l=ca'),  # l1
      (used_theorems['parallel'], 'A=C l=ab'),  # l2
      (used_theorems['parallel'], 'A=A l=bc'),  # l3
      (used_theorems['eq'], 'l=ab l1=ca l2=l1'),
      (used_theorems['eq'], 'l=bc l1=l2 l2=ab'),
      (used_theorems['eq'], 'l=ab l1=l3 l2=bc'),
      (used_theorems['eq'], 'l=ca l1=l3 l2=bc'),
      (used_theorems['eq'], 'l=bc l1=l1 l2=ca'),
      (used_theorems['asa'], 'A=A B=C C=B D=B F=A de=l1 ef=l3'),
      (used_theorems['eq'], 'l=ca l1=ab l2=l2'),
      (used_theorems['asa'], 'A=B B=A C=C D=C F=B de=l2 ef=l1'),
      (used_theorems['asa'], 'A=C B=B C=A D=A F=C de=l3 ef=l2'),
  ]

  print('\nRunning ABC=CAX redundant 0:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  # import pdb; pdb.set_trace()
  # Extract state queue & proof queue that prove AB = CP3
  # conclusion = action_chain[-1].matched_conclusion
  # queue = list(conclusion.topological_list[-6])
  # queue += conclusion.topological_list[-5]
  # state_queue = [r.init_list[0] for r in queue[1:]]
  # proof_queue = [tuple(queue)]

  # print([r.name for r in action_chain[-1].new_objects])
  # import pdb; pdb.set_trace()

  state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], '5m')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)

  # Test if we are having the correct problem statement
  assert len(problem_canvas.points) == 4
  assert len(problem_canvas.lines) == 5
  assert len(problem.name2obj) == 25, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 3
  assert chosen_proof_steps == [6, 9, 11]

  steps = [
      (used_theorems['eq'], 'l=ca l1=l3 l2=bc'),
      (used_theorems['eq'], 'l=ca l1=ab l2=l2'),
      (used_theorems['asa'], 'A=C B=B C=A D=A F=C de=l3 ef=l2')
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 4
  assert len(proved_canvas.lines) == 5
  assert len(proved_problem.name2obj) == 43, len(proved_problem.name2obj)


def test_whittle1():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

  steps = [
      (used_theorems['parallel'], 'A=C l=ab'),  # l1
      (used_theorems['parallel'], 'A=A l=bc'),  # l2
      (used_theorems['mid'], 'A=A B=B'),  # P1
      (used_theorems['parallel'], 'A=P1 l=l2'),  # l3
      (used_theorems['seg_line'], 'l=l3 A=A B=C'),  # P2
      (used_theorems['eq'], 'l=ca l1=bc l2=l2'),
      (used_theorems['eq'], 'l=ca l1=l1 l2=ab'),
      (used_theorems['eq'], 'l=ca l1=l2 l2=l3'),
      (used_theorems['asa'], 'A=A B=B C=C D=C F=A de=l1 ef=l2'),
  ]

  print('\nRunning ABC=CAX redundant 1:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  # import pdb; pdb.set_trace()
  # Extract state queue & proof queue that prove AB = CP3
  # conclusion = action_chain[-1].matched_conclusion
  # queue = list(conclusion.topological_list[-6])
  # queue += conclusion.topological_list[-5]
  # state_queue = [r.init_list[0] for r in queue[1:]]
  # proof_queue = [tuple(queue)]

  state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], '5m')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)

  # Test if we are having the correct problem statement
  assert len(problem_canvas.points) == 4
  assert len(problem_canvas.lines) == 5
  assert len(problem.name2obj) == 25, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 3
  assert chosen_proof_steps == [5, 6, 8]

  steps = [
      (used_theorems['eq'], 'l=ca l1=bc l2=l2'),
      (used_theorems['eq'], 'l=ca l1=l1 l2=ab'),
      (used_theorems['asa'], 'A=A B=B C=C D=C F=A de=l1 ef=l2')
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 4
  assert len(proved_canvas.lines) == 5
  assert len(proved_problem.name2obj) == 43, len(proved_problem.name2obj)


def test_whittle2():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

  steps = [
      (used_theorems['parallel'], 'A=C l=ab'),  # l1
      (used_theorems['parallel'], 'A=A l=bc'),  # l2
      (used_theorems['mid'], 'A=A B=C'),  # P1
      (used_theorems['parallel'], 'A=P1 l=l1'),  # l3
      (used_theorems['eq'], 'l=ca l1=ab l2=l1'),
      (used_theorems['eq'], 'l=ca l1=l2 l2=bc'),
      (used_theorems['asa'], 'A=A B=B C=C D=C F=A de=l1 ef=l2'),
  ]

  print('\nRunning ABC=CAX redundant 2:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  # Extract state queue & proof queue that prove AB = CP3
  # conclusion = action_chain[-1].matched_conclusion
  # queue = list(conclusion.topological_list[-6])
  # queue += conclusion.topological_list[-5]
  # state_queue = [r.init_list[0] for r in queue[1:]]
  # proof_queue = [tuple(queue)]
  # import pdb; pdb.set_trace()
  state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], '5m')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)
  assert len(problem_canvas.points) == 4
  assert len(problem_canvas.lines) == 5
  assert len(problem.name2obj) == 25, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 3
  assert chosen_proof_steps == [4, 5, 6]

  steps = [
      (used_theorems['eq'], 'l=ca l1=ab l2=l1'),
      (used_theorems['eq'], 'l=ca l1=l2 l2=bc'),
      (used_theorems['asa'], 'A=A B=B C=C D=C F=A de=l1 ef=l2')
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 4
  assert len(proved_canvas.lines) == 5
  assert len(proved_problem.name2obj) == 43, len(proved_problem.name2obj)


def test_whittle3():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()
  steps = [
      (used_theorems['mid'], 'A=C B=A'),  # P34=P1
      (used_theorems['parallel'], 'A=P1 l=bc'),  # l30=l1
      (used_theorems['mid'], 'A=B B=C'),  # P43=P2
      (used_theorems['line'], 'A=P2 B=P1'),  # l40=l2
      (used_theorems['parallel'], 'A=P2 l=ca'),  # l41=l3

      (used_theorems['eq'], 'l=l2 l1=l3 l2=ca'),
      (used_theorems['eq'], 'l=l2 l1=bc l2=l1'),
      (used_theorems['asa'], 'A=P2 B=C C=P1 D=P1 F=P2 de=l1 ef=l3'),  # P45=P3
      (used_theorems['eq'], 'l=l1 l1=ca l2=l3'),
      (used_theorems['sas'], 'A=P2 B=C C=P1 D=P3 E=P1 F=A'),  # l42=l4
      (used_theorems['.parallel'], 'l=ca l1=l2 l2=l4'),  # df=l4
  ]

  print('\nRunning Parallel proof redundant:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  # Extract state queue & proof queue that prove AB = CP3
  # conclusion = action_chain[-1].matched_conclusion
  # queue = list(conclusion.topological_list[-2])
  # queue += conclusion.topological_list[-1]
  # state_queue = [r.init_list[0] for r in queue[1:]]
  # proof_queue = [tuple(queue)]
  # import pdb; pdb.set_trace()
  state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], 'd3')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)

  assert len(problem_canvas.points) == 6
  assert len(problem_canvas.lines) == 7
  assert len(problem.name2obj) == 39, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 6
  assert chosen_proof_steps == [5, 6, 7, 8, 9, 10]

  steps = [
      (used_theorems['eq'], 'l=l2 l1=l3 l2=ca'),
      (used_theorems['eq'], 'l=l2 l1=bc l2=l1'),
      (used_theorems['asa'], 'A=P2 B=C C=P1 D=P1 F=P2 de=l1 ef=l3'),
      (used_theorems['eq'], 'l=l1 l1=ca l2=l3'),
      (used_theorems['sas'], 'A=P2 B=C C=P1 D=P3 E=P1 F=A'),
      (used_theorems['.parallel'], 'l=ca l1=l2 l2=l4')
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 6
  assert len(proved_canvas.lines) == 7
  assert len(proved_problem.name2obj) == 65, len(proved_problem.name2obj)


def test_whittle3a():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()
  steps = [
      (used_theorems['mid'], 'A=C B=A'),  # P34=P1
      (used_theorems['parallel'], 'A=P1 l=bc'),  # l30=l1
      (used_theorems['mid'], 'A=B B=C'),  # P43=P2
      (used_theorems['line'], 'A=P2 B=P1'),  # l40=l2
      (used_theorems['parallel'], 'A=P2 l=ca'),  # l41=l3

      (used_theorems['eq'], 'l=l2 l1=l3 l2=ca'),
      (used_theorems['eq'], 'l=l2 l1=bc l2=l1'),
      (used_theorems['asa'], 'A=P2 B=C C=P1 D=P1 F=P2 de=l1 ef=l3'),  # P45=P3
      (used_theorems['eq'], 'l=l1 l1=ca l2=l3'),
      (used_theorems['sas'], 'A=P2 B=C C=P1 D=P3 E=P1 F=A'),  # l42=l4
      (used_theorems['.parallel'], 'l=ca l1=l2 l2=l4'),  # df=l4
  ]

  print('\nRunning Parallel proof redundant:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  # Extract state queue & proof queue that prove AB = CP3
  # conclusion = action_chain[-1].matched_conclusion
  # queue = list(conclusion.topological_list[-2])
  # queue += conclusion.topological_list[-1]
  # state_queue = [r.init_list[0] for r in queue[1:]]
  # proof_queue = [tuple(queue)]
  # import pdb; pdb.set_trace()

  d3, ab, l4 = map(state.name2obj.get, ['d3', 'ab', 'l4'])
  ab_d = state.obj2valrel[ab]
  l4_d = state.obj2valrel[l4]
  queue = [d3, ab_d, l4_d]
  state_queue = [r.init_list[0] for r in queue[1:]]
  proof_queue = [tuple(queue)]
  # state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], 'd3')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)

  assert len(problem_canvas.points) == 6
  assert len(problem_canvas.lines) == 7
  assert len(problem.name2obj) == 39, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 6
  assert chosen_proof_steps == [5, 6, 7, 8, 9, 10]

  steps = [
      (used_theorems['eq'], 'l=l2 l1=l3 l2=ca'),
      (used_theorems['eq'], 'l=l2 l1=bc l2=l1'),
      (used_theorems['asa'], 'A=P2 B=C C=P1 D=P1 F=P2 de=l1 ef=l3'),
      (used_theorems['eq'], 'l=l1 l1=ca l2=l3'),
      (used_theorems['sas'], 'A=P2 B=C C=P1 D=P3 E=P1 F=A'),
      (used_theorems['.parallel'], 'l=ca l1=l2 l2=l4')
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 6
  assert len(proved_canvas.lines) == 7
  assert len(proved_problem.name2obj) == 66, len(proved_problem.name2obj)


def test_whittle4():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()
  steps = [
      (used_theorems['mid'], 'A=B B=A'),  # P1
      (used_theorems['parallel'], 'A=C l=ab'),  # l1
      (used_theorems['line'], 'A=P1 B=C'),  # l2
      (used_theorems['parallel'], 'A=B l=l2'),  # l3
      (used_theorems['mid'], 'A=P1 B=C'),  # P2

      (used_theorems['eq'], 'l=bc l1=ab l2=l1'),
      (used_theorems['eq'], 'l=l2 l1=ab l2=l1'),
      (used_theorems['eq'], 'l=bc l1=l3 l2=l2'),

      (used_theorems['asa'], 'A=C B=P1 C=B D=B de=l3 ef=l1 F=C'),  # P3
      (used_theorems['eq'], 'l=l3 l1=ab l2=l1'),
      (used_theorems['sas'], 'A=A B=P1 C=C D=P3 E=C F=P1'),  # l4

      (used_theorems['eq'], 'l=l4 l1=l1 l2=ab l_hp2=l4_hp'),  # <-- 11
      (used_theorems['eq'], 'l=ca l1=l1 l2=ab'),
      (used_theorems['asa'], 'A=B B=C C=P3 D=C de=bc ef=ab F=P1'),
      (used_theorems['.parallel'], 'l=ab l1=l4 l2=ca'), # <-- 14
  ]

  print('\nRunning Parallel proof redundant:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  # Extract state queue & proof queue that prove AB = CP3
  # conclusion = action_chain[-1].matched_conclusion
  # queue = list(conclusion.topological_list[-2])
  # queue += conclusion.topological_list[-1]
  # state_queue = [r.init_list[0] for r in queue[1:]]
  # proof_queue = [tuple(queue)]
  state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], 'd3')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)

  assert len(problem_canvas.points) == 5
  assert len(problem_canvas.lines) == 7
  assert len(problem.name2obj) == 35, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 7
  assert chosen_proof_steps == [5, 6, 7, 8, 10, 11, 14]

  steps = [
      (used_theorems['eq'], 'l=bc l1=ab l2=l1'),
      (used_theorems['eq'], 'l=bc l1=l3 l2=l2'),
      (used_theorems['asa'], 'A=C B=P1 C=B D=B F=C de=l3 ef=l1'),
      (used_theorems['eq'], 'l=l2 l1=ab l2=l1'),
      (used_theorems['sas'], 'A=A B=P1 C=C D=P3 E=C F=P1'),
      (used_theorems['eq'], 'l=l4 l1=l1 l2=ab'),
      (used_theorems['.parallel'], 'l=ab l1=l4 l2=ca')
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 5
  assert len(proved_canvas.lines) == 7
  assert len(proved_problem.name2obj) == 70, len(proved_problem.name2obj)


def test_whittle5():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()
  steps = [
      (used_theorems['parallel'], 'A=A l=bc'),  # l1
      (used_theorems['parallel'], 'A=C l=ab'),  # l2
      (used_theorems['mid'], 'A=C B=A'),  # P1
      (used_theorems['mirror'], 'A=C B=A'),  # P2

      (used_theorems['eq'], 'l=ca l1=l2 l2=ab'),
      (used_theorems['eq'], 'l=ca l1=bc l2=l1'),
      (used_theorems['asa'], 'A=C B=B C=A D=A F=C de=l1 ef=l2'),  # P3
      (used_theorems['sas'], 'A=P1 B=C C=P3 D=P1 E=A F=B'),
  ]

  print('\nRunning Parallel proof redundant:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)
  state_queue, proof_queue = get_state_and_proof_objects(action_chain[-1], '7m')

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)

  assert len(problem_canvas.points) == 5
  assert len(problem_canvas.lines) == 5
  assert len(problem.name2obj) == 30, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 4
  assert chosen_proof_steps == [4, 5, 6, 7]

  steps = [
      (used_theorems['eq'], 'l=ca l1=l2 l2=ab'),
      (used_theorems['eq'], 'l=ca l1=bc l2=l1'),
      (used_theorems['asa'], 'A=C B=B C=A D=A F=C de=l1 ef=l2'),  # P3
      (used_theorems['sas'], 'A=P1 B=C C=P3 D=P1 E=A F=B'),
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 5 
  assert len(proved_canvas.lines) == 7
  assert len(proved_problem.name2obj) == 62, len(proved_problem.name2obj)


def test_whittle6():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()
  steps = [
      (used_theorems['parallel'], 'A=A l=bc'),  # l1
      (used_theorems['parallel'], 'A=B l=ca'),  # l2
      (used_theorems['parallel'], 'A=C l=ab'),  # l3
      (used_theorems['eq'], 'l=ca l1=ab l2=l3'),
      (used_theorems['eq'], 'l=ca l1=bc l2=l1'),
      (used_theorems['eq'], 'l=ab l1=ca l2=l2'),
      (used_theorems['eq'], 'l=ab l1=bc l2=l1'),
      (used_theorems['asa'], 'A=A B=B C=C D=C F=A de=l3 ef=l1'),  # P1
      (used_theorems['asa'], 'A=A B=C C=B D=B F=A de=l2 ef=l1'),  # P2
  ]

  print('\nRunning Parallel proof redundant:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)

  obj1 = state.name2obj['s2']
  obj2 = state.name2obj['s4']
  rel1 = state.obj2valrel[obj1]
  rel2 = state.obj2valrel[obj2]
  assert rel1.init_list[1] == rel2.init_list[1]
  val = rel1.init_list[1]

  state_queue = [obj1, obj2]
  proof_queue = [(val, rel1, rel2)]

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)

  assert len(problem_canvas.points) == 5
  assert len(problem_canvas.lines) == 6
  assert len(problem.name2obj) == 31, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 6
  assert chosen_proof_steps == [3, 4, 5, 6, 7, 8]

  steps = [
      (used_theorems['eq'], 'l=ca l1=ab l2=l3'),
      (used_theorems['eq'], 'l=ca l1=bc l2=l1'),
      (used_theorems['eq'], 'l=ab l1=ca l2=l2'),
      (used_theorems['eq'], 'l=ab l1=bc l2=l1'),
      (used_theorems['asa'], 'A=A B=B C=C D=C F=A de=l3 ef=l1'),  # P1
      (used_theorems['asa'], 'A=A B=C C=B D=B F=A de=l2 ef=l1'),  # P2
  ]

  proved_problem, proved_canvas, _ = action_chain_lib.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 5 
  assert len(proved_canvas.lines) == 6
  assert len(proved_problem.name2obj) == 60, len(proved_problem.name2obj)


def whittle(state_queue, proof_queue, action_chain, 
            init_state, init_canvas, canvas):
  # Basically shave off any excess from action_chain
  # and crystallize what is relevant as premise & conclusion
  # of a discovered theorem.

  whittled_state = whittling.whittle_from(list(state_queue), action_chain)
  proof_whittled = whittling.whittle_from(
      list(proof_queue), action_chain, 
      goal_objects=state_queue, whittled_state=whittled_state)

  for i, p in enumerate(proof_whittled):
    if not (p == [] or p == True):
      # if whittled_state[i] != True:
      #   whittled_state[i] += p
      proof_whittled[i] = []

  new_state = init_state.copy()
  new_canvas = init_canvas.copy()

  print('\nWhittled state: ')
  for i, (step, action) in enumerate(zip(whittled_state, action_chain)):
    if step == []:
      continue
    if step == True:
      print(i, action.to_str())
      new_state.add_relations(action.conclusion_objects)
    else:
      all_constructions = sum(step, [])
      new_state.add_relations(all_constructions)
      print('{}. {} : {}'.format(i, action.theorem.name,
                            [r.name for r in all_constructions]))

  info = {}
  for name, obj in new_state.name2obj.items():
    if isinstance(obj, Point):
      new_canvas.update_point(obj, canvas.points[obj])
    elif isinstance(obj, Line):
      new_canvas.update_line(obj, canvas.lines[obj])
    elif isinstance(obj, Circle):
      new_canvas.circles[obj] = canvas.circles[obj]
  new_state.add_spatial_relations(new_canvas.line2points)

  proof_steps = []
  for i, (step, action) in enumerate(zip(proof_whittled, action_chain)):
    if step == []:
      action = None
    proof_steps.append(action)

  print()
  print('Whittled Proof {}'.format([r.name for r in proof_queue[0]]))
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

def hey():
  a = [(PointEndsSegment, Point, 'A', Segment, 'AB'), (PointEndsSegment, Point, 'B', Segment, 'AB'), (PointEndsSegment, Point, 'B', Segment, 'BC'), (PointEndsSegment, Point, 'C', Segment, 'BC'), (PointEndsSegment, Point, 'C', Segment, 'CA'), (PointEndsSegment, Point, 'A', Segment, 'CA'), (LineContainsPoint, Line, 'ab', Point, 'A'), (LineContainsPoint, Line, 'ab', Point, 'B'), (LineContainsPoint, Line, 'bc', Point, 'B'), (LineContainsPoint, Line, 'bc', Point, 'C'), (LineContainsPoint, Line, 'ca', Point, 'C'), (LineContainsPoint, Line, 'ca', Point, 'A'), (LineBordersHalfplane, Line, 'ab', HalfPlane, 'ab_hp1'), (LineBordersHalfplane, Line, 'ab', HalfPlane, 'ab_hp2'), (HalfPlaneContainsPoint, HalfPlane, 'ab_hp2', Point, 'C'), (LineBordersHalfplane, Line, 'bc', HalfPlane, 'bc_hp1'), (LineBordersHalfplane, Line, 'bc', HalfPlane, 'bc_hp2'), (HalfPlaneContainsPoint, HalfPlane, 'bc_hp2', Point, 'A'), (LineBordersHalfplane, Line, 'ca', HalfPlane, 'ca_hp1'), (LineBordersHalfplane, Line, 'ca', HalfPlane, 'ca_hp2'), (HalfPlaneContainsPoint, HalfPlane, 'ca_hp2', Point, 'B'), (SegmentHasLength, Segment, 'CA', SegmentLength, '740m'), (LineContainsPoint, Line, 'ca', Point, 'P150'), (PointEndsSegment, Point, 'C', Segment, 's276'), (PointEndsSegment, Point, 'P150', Segment, 's276'), (SegmentHasLength, Segment, 's276', SegmentLength, '740m'), (HalfPlaneContainsPoint, HalfPlane, 'ab_hp2', Point, 'P150'), (HalfPlaneContainsPoint, HalfPlane, 'bc_hp1', Point, 'P150'), (LineHasDirection, Line, 'bc', LineDirection, 'd25'), (LineContainsPoint, Line, 'l94', Point, 'A'), (LineHasDirection, Line, 'l94', LineDirection, 'd25'), (LineBordersHalfplane, Line, 'l94', HalfPlane, 'l94_hp1'), (LineBordersHalfplane, Line, 'l94', HalfPlane, 'l94_hp2'), (HalfPlaneContainsPoint, HalfPlane, 'l94_hp1', Point, 'B'), (HalfPlaneContainsPoint, HalfPlane, 'l94_hp1', Point, 'C'), (HalfPlaneContainsPoint, HalfPlane, 'l94_hp1', Point, 'P150'), (LineContainsPoint, Line, 'l97', Point, 'P150'), (LineHasDirection, Line, 'l97', LineDirection, 'd25'), (LineBordersHalfplane, Line, 'l97', HalfPlane, 'l97_hp1'), (LineBordersHalfplane, Line, 'l97', HalfPlane, 'l97_hp2'), (HalfPlaneContainsPoint, HalfPlane, 'l97_hp2', Point, 'A'), (HalfPlaneContainsPoint, HalfPlane, 'l97_hp2', Point, 'B'), (HalfPlaneContainsPoint, HalfPlane, 'l97_hp2', Point, 'C'), (LineHasDirection, Line, 'ab', LineDirection, 'd26'), (LineContainsPoint, Line, 'l98', Point, 'C'), (LineHasDirection, Line, 'l98', LineDirection, 'd26'), (LineBordersHalfplane, Line, 'l98', HalfPlane, 'l98_hp1'), (LineBordersHalfplane, Line, 'l98', HalfPlane, 'l98_hp2'), (HalfPlaneContainsPoint, HalfPlane, 'l98_hp1', Point, 'A'), (HalfPlaneContainsPoint, HalfPlane, 'l98_hp1', Point, 'B'), (HalfPlaneContainsPoint, HalfPlane, 'l98_hp2', Point, 'P150'), (LineContainsPoint, Line, 'l99', Point, 'B'), (LineBordersHalfplane, Line, 'l99', HalfPlane, '.77'), (LineBordersHalfplane, Line, 'l99', HalfPlane, '.78'), (HalfplaneCoversAngle, HalfPlane, '.77', Angle, '^968'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp1', Angle, '^968'), (HalfplaneCoversAngle, HalfPlane, '.77', Angle, '^969'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp1', Angle, '^969'), (HalfplaneCoversAngle, HalfPlane, '.78', Angle, '^970'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp1', Angle, '^970'), (HalfplaneCoversAngle, HalfPlane, '.78', Angle, '^971'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp1', Angle, '^971'), (AngleHasMeasure, Angle, '^968', AngleMeasure, '531"'), (AngleHasMeasure, Angle, '^971', AngleMeasure, '531"'), (AngleHasMeasure, Angle, '^969', AngleMeasure, '532"'), (AngleHasMeasure, Angle, '^970', AngleMeasure, '532"'), (HalfPlaneContainsPoint, HalfPlane, '.77', Point, 'C'), (HalfPlaneContainsPoint, HalfPlane, '.77', Point, 'P150'), (HalfPlaneContainsPoint, HalfPlane, '.78', Point, 'A'), (LineContainsPoint, Line, 'l100', Point, 'A'), (LineBordersHalfplane, Line, 'l100', HalfPlane, '.79'), (LineBordersHalfplane, Line, 'l100', HalfPlane, '.80'), (HalfplaneCoversAngle, HalfPlane, '.79', Angle, '^972'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp2', Angle, '^972'), (HalfplaneCoversAngle, HalfPlane, '.79', Angle, '^973'), (HalfplaneCoversAngle, HalfPlane, 'l94_hp1', Angle, '^973'), (HalfplaneCoversAngle, HalfPlane, '.80', Angle, '^974'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp2', Angle, '^974'), (HalfplaneCoversAngle, HalfPlane, '.80', Angle, '^975'), (HalfplaneCoversAngle, HalfPlane, 'l94_hp1', Angle, '^975'), (AngleHasMeasure, Angle, '^972', AngleMeasure, '533"'), (AngleHasMeasure, Angle, '^975', AngleMeasure, '533"'), (AngleHasMeasure, Angle, '^973', AngleMeasure, '534"'), (AngleHasMeasure, Angle, '^974', AngleMeasure, '534"'), (HalfPlaneContainsPoint, HalfPlane, '.80', Point, 'B'), (HalfPlaneContainsPoint, HalfPlane, '.80', Point, 'C'), (HalfPlaneContainsPoint, HalfPlane, '.80', Point, 'P150'), (LineContainsPoint, Line, 'l101', Point, 'A'), (LineBordersHalfplane, Line, 'l101', HalfPlane, '.81'), (LineBordersHalfplane, Line, 'l101', HalfPlane, '.82'), (HalfplaneCoversAngle, HalfPlane, '.81', Angle, '^976'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp2', Angle, '^976'), (HalfplaneCoversAngle, HalfPlane, '.81', Angle, '^977'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp1', Angle, '^977'), (HalfplaneCoversAngle, HalfPlane, '.82', Angle, '^978'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp2', Angle, '^978'), (HalfplaneCoversAngle, HalfPlane, '.82', Angle, '^979'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp1', Angle, '^979'), (AngleHasMeasure, Angle, '^976', AngleMeasure, '535"'), (AngleHasMeasure, Angle, '^979', AngleMeasure, '535"'), (AngleHasMeasure, Angle, '^977', AngleMeasure, '536"'), (AngleHasMeasure, Angle, '^978', AngleMeasure, '536"'), (HalfPlaneContainsPoint, HalfPlane, '.81', Point, 'B'), (HalfPlaneContainsPoint, HalfPlane, '.81', Point, 'C'), (HalfPlaneContainsPoint, HalfPlane, '.81', Point, 'P150'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp1', Angle, '^980'), (HalfplaneCoversAngle, HalfPlane, 'l98_hp1', Angle, '^980'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp1', Angle, '^981'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp2', Angle, '^981'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp2', Angle, '^982'), (HalfplaneCoversAngle, HalfPlane, 'l98_hp1', Angle, '^982'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp2', Angle, '^983'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp2', Angle, '^983'), (AngleHasMeasure, Angle, '^980', AngleMeasure, '537"'), (AngleHasMeasure, Angle, '^983', AngleMeasure, '537"'), (AngleHasMeasure, Angle, '^981', AngleMeasure, '538"'), (AngleHasMeasure, Angle, '^982', AngleMeasure, '538"'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp1', Angle, '^984'), (HalfplaneCoversAngle, HalfPlane, 'l94_hp1', Angle, '^984'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp1', Angle, '^985'), (HalfplaneCoversAngle, HalfPlane, 'l97_hp2', Angle, '^985'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp2', Angle, '^986'), (HalfplaneCoversAngle, HalfPlane, 'l94_hp1', Angle, '^986'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp2', Angle, '^987'), (HalfplaneCoversAngle, HalfPlane, 'l97_hp2', Angle, '^987'), (AngleHasMeasure, Angle, '^984', AngleMeasure, '546"'), (AngleHasMeasure, Angle, '^987', AngleMeasure, '546"'), (AngleHasMeasure, Angle, '^985', AngleMeasure, '540"'), (AngleHasMeasure, Angle, '^986', AngleMeasure, '540"'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp1', Angle, '^988'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp1', Angle, '^988'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp2', Angle, '^989'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp1', Angle, '^989'), (AngleHasMeasure, Angle, '^988', AngleMeasure, '546"'), (AngleHasMeasure, Angle, '^989', AngleMeasure, '540"'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp2', Angle, '^990'), (HalfplaneCoversAngle, HalfPlane, 'l94_hp1', Angle, '^990'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp2', Angle, '^991'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp2', Angle, '^991'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp1', Angle, '^992'), (HalfplaneCoversAngle, HalfPlane, 'l94_hp1', Angle, '^992'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp1', Angle, '^993'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp2', Angle, '^993'), (AngleHasMeasure, Angle, '^990', AngleMeasure, '542"'), (AngleHasMeasure, Angle, '^993', AngleMeasure, '542"'), (AngleHasMeasure, Angle, '^991', AngleMeasure, '544"'), (AngleHasMeasure, Angle, '^992', AngleMeasure, '544"'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp1', Angle, '^994'), (HalfplaneCoversAngle, HalfPlane, 'l98_hp1', Angle, '^994'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp1', Angle, '^995'), (HalfplaneCoversAngle, HalfPlane, 'ab_hp2', Angle, '^995'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp2', Angle, '^996'), (HalfplaneCoversAngle, HalfPlane, 'l98_hp1', Angle, '^996'), (AngleHasMeasure, Angle, '^994', AngleMeasure, '544"'), (AngleHasMeasure, Angle, '^995', AngleMeasure, '545"'), (AngleHasMeasure, Angle, '^996', AngleMeasure, '545"'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp2', Angle, '^997'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp2', Angle, '^997'), (HalfplaneCoversAngle, HalfPlane, 'ca_hp1', Angle, '^998'), (HalfplaneCoversAngle, HalfPlane, 'bc_hp2', Angle, '^998'), (AngleHasMeasure, Angle, '^997', AngleMeasure, '546"'), (AngleHasMeasure, Angle, '^998', AngleMeasure, '540"')]



if __name__ == '__main__':
  np.random.seed(1234)
  t = time.time()
  test_intersect_line_line()
  test_intersect_line_line2()
  test_thales()
  test_thales_whittle1()
  test_thales_whittle2()
  test_whittle0()
  test_whittle1()
  test_whittle2()
  test_whittle3()
  test_whittle4()
  test_whittle5()
  test_whittle6()
  sas_hp()
  conclusion_match()
  sas()
  time_sas()
  state_merge_and_copy()
  print('OK')
  print('Total time = {}'.format(time.time()-t))
