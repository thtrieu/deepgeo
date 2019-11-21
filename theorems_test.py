from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import networkx as nx
import theorems_utils
import trieu_graph_match
import geometry

from theorems_utils import *

import theorems
import sketch
import explore

from theorems import *

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import LineDirectionPerpendicular, PointCentersCircle
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

  matches = trieu_graph_match.match_relations(
      premise, state.relations,
      conclusion=theorem.conclusion, 
      randomize=False, distinct=theorem.distinct)
  start = time.time()
  matches = list(matches)
  print(time.time() - start)
  for _, m in matches:
    print_match(m, [Point])
  assert len(matches) == 2, len(matches)

  state_conclusion, match = matches[0]
  state.add_relations(
      sum(state_conclusion.topological_list, [])
  )

  matches = trieu_graph_match.match_relations(
      premise, state.relations,
      conclusion=theorem.conclusion, 
      randomize=False, distinct=theorem.distinct)
  start = time.time()
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

  X, Y, Z, T = map(Point, 'XYZT')
  XY = Segment('XY')
  ZT = Segment('ZT')

  premise_match = {X: A, Y: B, Z: B, T: A}

  conclusion = Conclusion()
  conclusion.add(*segment_def(XY, X, Y))
  conclusion.add(*segment_def(ZT, Z, T))
  conclusion.add(*have_length('1m', XY, ZT))

  state_conclusion, match = trieu_graph_match.match_conclusions(
      conclusion, 
      dict(state_candidates), 
      dict(premise_match), 
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
      [A, B, C, AB, BC, CA, ab, bc, ca] +
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


used_theorems = {
    'mid': theorems.ConstructMidPoint(),
    # 'line_line': theorems.ConstructIntersectLineLine(),
    'seg_line': theorems.ConstructIntersectSegmentLine(),
    'parallel': theorems.ConstructParallelLine(),
    'line': theorems.ConstructThirdLine(),
    'eq': theorems.EqualAnglesBecauseParallel(),
    'sas': theorems.SAS(),
    'asa': theorems.ASA(),
    '.parallel': theorems.ParallelBecauseCorrespondingAngles()

}


def test_thales():
  geometry.reset()
  init_state, init_canvas = triangle_seed()
  state, canvas = init_state.copy(), init_canvas.copy()

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
  state, canvas, action_chain = explore.execute_steps(steps, state, canvas)
  print('thales exec time ', time.time()-s)
  assert len(state.name2obj) == 68, len(state.name2obj)
  assert len(canvas.points) == 6, len(canvas.points)
  assert len(canvas.lines) == 6, len(canvas.lines)
  assert len(canvas.circles) == 0, len(canvas.circles)
  # print([r.name for r in action_chain[-1].new_objects])
  # s = time.time()
  # action = used_theorems['asa'].match_one_random(state)
  # print(time.time() - s)
  # assert action is None


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

  state, canvas, action_chain = explore.execute_steps(steps, state, canvas)

  # Extract state queue & proof queue that prove P2 is mid AC
  conclusion = action_chain[-1].matched_conclusion
  queue = list(conclusion.topological_list[-6])
  queue += conclusion.topological_list[-5]
  state_queue = [r.init_list[0] for r in queue[1:]]
  proof_queue = [tuple(queue)]

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('thales whittle time ', time.time()-s)

  # Test if we are having the correct problem statement
  assert len(problem_canvas.points) == 5
  assert len(problem_canvas.lines) == 4
  assert len(problem_canvas.circles) == 0
  assert len(problem.name2obj) == 23, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 8
  assert chosen_proof_steps == [4, 5, 7, 8, 10, 11, 13, 14]

  # Test if the correct proof step applied on the problem statement
  # give the correct solution
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

  proved_problem, proved_canvas, _ = explore.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 6
  assert len(proved_canvas.lines) == 6
  assert len(proved_canvas.circles) == 0
  assert len(proved_problem.name2obj) == 68, len(proved_problem.name2obj)
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

  state, canvas, action_chain = explore.execute_steps(steps, state, canvas)

  # Extract state queue & proof queue that prove P2 is mid AC
  conclusion = action_chain[-1].matched_conclusion
  queue = list(conclusion.topological_list[-6])
  queue += conclusion.topological_list[-5]
  state_queue = [r.init_list[0] for r in queue[1:]]
  proof_queue = [tuple(queue)]

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('thales whittle time ', time.time()-s)

  # Test if we are having the correct problem statement
  assert len(problem_canvas.points) == 5
  assert len(problem_canvas.lines) == 4
  assert len(problem_canvas.circles) == 0
  assert len(problem.name2obj) == 23, len(problem.name2obj)

  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]

  assert len(chosen_proof_steps) == 8
  assert chosen_proof_steps == [4, 5, 7, 8, 10, 11, 13, 14]

  # Test if the correct proof step applied on the problem statement
  # give the correct solution
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

  proved_problem, proved_canvas, _ = explore.execute_steps(
      steps, problem, problem_canvas)

  assert len(proved_canvas.points) == 6
  assert len(proved_canvas.lines) == 6
  assert len(proved_canvas.circles) == 0
  assert len(proved_problem.name2obj) == 68, len(proved_problem.name2obj)
  # action = used_theorems['asa'].match_one_random(proved_problem)
  # assert action is None


def test_whittle():
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

  state, canvas, action_chain = explore.execute_steps(steps, state, canvas)
  # Extract state queue & proof queue that prove AB = CP3
  conclusion = action_chain[-1].matched_conclusion
  queue = list(conclusion.topological_list[-6])
  queue += conclusion.topological_list[-5]
  state_queue = [r.init_list[0] for r in queue[1:]]
  proof_queue = [tuple(queue)]

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)
  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 3
  assert chosen_proof_steps == [5, 6, 8]


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

  state, canvas, action_chain = explore.execute_steps(steps, state, canvas)
  # Extract state queue & proof queue that prove AB = CP3
  conclusion = action_chain[-1].matched_conclusion
  queue = list(conclusion.topological_list[-6])
  queue += conclusion.topological_list[-5]
  state_queue = [r.init_list[0] for r in queue[1:]]
  proof_queue = [tuple(queue)]

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)
  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 3
  assert chosen_proof_steps == [4, 5, 6]


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

  state, canvas, action_chain = explore.execute_steps(steps, state, canvas)
  # Extract state queue & proof queue that prove AB = CP3
  conclusion = action_chain[-1].matched_conclusion
  queue = list(conclusion.topological_list[-2])
  queue += conclusion.topological_list[-1]
  state_queue = [r.init_list[0] for r in queue[1:]]
  proof_queue = [tuple(queue)]

  s = time.time()
  problem, problem_canvas, proof_steps = whittle(
      state_queue, proof_queue, action_chain,
      init_state, init_canvas, canvas)
  print('whittle time ', time.time()-s)
  # Test if we have the correct proof steps
  chosen_proof_steps = [i for i, step in enumerate(proof_steps)
                        if step is not None]
  assert len(chosen_proof_steps) == 6
  assert chosen_proof_steps == [5, 6, 7, 8, 9, 10]



def whittle(state_queue, proof_queue, action_chain, 
            init_state, init_canvas, canvas):
  # Basically shave off any excess from action_chain
  # and crystallize what is relevant as premise & conclusion
  # of a discovered theorem.

  whittled_state = explore.whittle_from(list(state_queue), action_chain)
  proof_whittled = explore.whittle_from(
      list(proof_queue), action_chain, 
      goal_objects=state_queue, whittled_state=whittled_state)

  for i, p in enumerate(proof_whittled):
    if not (p == [] or p == True):
      if whittled_state[i] != True:
        whittled_state[i] += p
      proof_whittled[i] = []

  new_state = init_state.copy()
  new_canvas = init_canvas.copy()

  print('\nWhittled state: ')
  for i, (step, action) in enumerate(zip(whittled_state, action_chain)):
    if step == []:
      continue
    if step == True:
      action.to_str()
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
  new_state.add_spatial_relations(new_canvas.line2hps)

  print()
  proof_steps = []
  for i, (step, action) in enumerate(zip(proof_whittled, action_chain)):
    if step == []:
      action = None
    proof_steps.append(action)

  print()
  print('Proof of {}'.format([r.name for r in proof_queue[0]]))
  for i, (step, action) in enumerate(zip(proof_whittled, action_chain)):
    if step == []:
      continue
    if step == True:
      mapping = {a.name: action.mapping[a].name for _, a in action.theorem.names.items()}
      print('Apply {}. {} {}'.format(i, action.theorem.name, mapping))
    else:
      all_constructions = sum(step, [])
      print('{}. {}'.format(i, [r.name for r in all_constructions]))
  print()
  return new_state, new_canvas, proof_steps


if __name__ == '__main__':
  test_thales()
  test_thales_whittle1()
  test_thales_whittle2()
  test_whittle()
  test_whittle2()
  test_whittle3()
  sas_hp()
  conclusion_match()
  sas()
  state_merge_and_copy()
  print('OK')
