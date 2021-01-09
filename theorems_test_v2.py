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
import debugging
import state

from theorems_utils import *
from theorems import *

from state import State, Conclusion

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle, TransitiveRelation
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, LineBordersHalfplane
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


all_theorems = theorems.theorem_from_short_name


def test_sss_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, BC, CA = map(Segment, 'AB BC CA'.split())

  init_state.add_relations(
      distinct(A, B, C) +
      segment_def(AB, A, B) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      have_length('1m', AB, CA)
  )

  init_state.add_spatial_relations(
      init_canvas.add_triangle(A, B, C, ab, bc, ca))

  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = [
      (all_theorems['sss'], 'A=B B=A C=C D=C E=A F=B')
  ]

  print('\nRunning SSS isosceles test:')
  action_chain_lib.execute_steps(steps, state, canvas)


def test_bisect_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, CA = map(Segment, 'AB CA'.split())

  ab_hp, ca_hp = map(HalfPlane, 'ab_hp ca_hp'.split())

  init_state.add_relations(
      # [A, B, C, AB, BC, CA, ab, bc, ca] +
      distinct(A, B, C) +
      distinct(ab, bc, ca) + 
      segment_def(AB, A, B) +
      segment_def(CA, C, A) +
      divides_halfplanes(ab, ab_hp, p1=C) +
      divides_halfplanes(ca, ca_hp, p1=B) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      have_length('1m', AB, CA)
  )

  line2points = init_canvas.add_triangle(A, B, C, ab, bc, ca)
  init_state.add_spatial_relations(line2points)
  init_canvas.update_hps(init_state.line2hps)

  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = [
      (all_theorems['bisect'], 'hp1=ab_hp hp2=ca_hp'),  # l1
      (all_theorems['seg_line'], 'l=l1 A=B B=C'),
      (all_theorems['sas'], 'A=B B=A C=P1 D=C E=A F=P1')
  ]

  print('\nRunning bisector isosceles test:')
  state, canvas, action_chain = action_chain_lib.execute_steps(steps, state, canvas)


def test_bisect_sss_isosceles():
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

  line2points = init_canvas.add_triangle(A, B, C, ab, bc, ca)
  init_state.add_spatial_relations(line2points)
  init_canvas.update_hps(init_state.line2hps)

  state, canvas = init_state.copy(), init_canvas.copy()

  steps = [
      (all_theorems['mid'], 'A=B B=C'),
      (all_theorems['line'], 'A=A B=P1'),
      (all_theorems['sss'], 'A=B B=A C=P1 D=C E=A F=P1')
  ]

  print('\nRunning bisector SSS isosceles test:')
  action_chain_lib.execute_steps(steps, state, canvas)


def test_asa_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, BC, CA = map(Segment, 'AB BC CA'.split())
  ab_hp, bc_hp, ca_hp = map(HalfPlane, 'ab_hp bc_hp ca_hp'.split())

  ABC, BCA = Angle('ABC'), Angle('BCA')

  init_state.add_relations(
      segment_def(AB, A, B) +
      segment_def(BC, B, C) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      divides_halfplanes(ab, ab_hp, p1=C) +
      divides_halfplanes(ca, ca_hp, p1=B) +
      divides_halfplanes(bc, bc_hp, p1=A) +
      angle_def(ABC, ab_hp, bc_hp) +
      angle_def(BCA, bc_hp, ca_hp) +
      have_measure('^1', ABC, BCA)
  )

  line2points = init_canvas.add_triangle(A, B, C, ab, bc, ca)
  init_state.add_spatial_relations(line2points)
  init_canvas.update_hps(init_state.line2hps)

  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = [
      (all_theorems['asa'], 'B=A C=B A=C D=B F=C de=ab ef=ca')
  ]

  # db = debugging.get_db()
  # best, miss = db.why_fail_to_match(
  #     all_theorems['asa'], state, command_str='B=A C=B A=C E=A D=B F=C')

  print('\nRunning ASA isosceles test:')
  action_chain_lib.execute_steps(steps, state, canvas)


def test_sas_isosceles():
  geometry.reset()

  init_canvas = sketch.Canvas()
  init_state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, BC, CA = map(Segment, 'AB BC CA'.split())
  ab_hp, ca_hp = map(HalfPlane, 'ab_hp ca_hp'.split())
  CAB = Angle('CAB')

  init_state.add_relations(
      segment_def(AB, A, B) +
      segment_def(BC, B, C) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      divides_halfplanes(ab, ab_hp, p1=C) +
      divides_halfplanes(ca, ca_hp, p1=B) +
      angle_def(CAB, ab_hp, ca_hp) + 
      have_length('1m', AB, CA)
  )

  line2points = init_canvas.add_triangle(A, B, C, ab, bc, ca)
  init_state.add_spatial_relations(line2points)
  init_canvas.update_hps(init_state.line2hps)

  state, canvas = init_state.copy(), init_canvas.copy()

  # Original thales + noises
  steps = [
      (all_theorems['sas'], 'B=A A=B C=C E=A D=C F=B')
  ]

  # db = debugging.get_db()
  # best, miss = db.why_fail_to_match(
  #     all_theorems['sas'], state, command_str='B=A A=B C=C E=A D=C F=B')
  # print(state.name_map(miss))

  print('\nRunning SAS isosceles test:')
  action_chain_lib.execute_steps(steps, state, canvas)




if __name__ == '__main__':
  np.random.seed(1234)
  t = time.time()
  # Self-congruences:
  test_sss_isosceles()
  # test_asa_isosceles()
  # test_sas_isosceles()
  # Aux point/lines
  test_bisect_isosceles()
  test_bisect_sss_isosceles()
  print('\n [OK!]')
  print('Total time = {}'.format(time.time()-t))