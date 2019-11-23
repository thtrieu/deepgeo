"""Implement the environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theorems_utils
import geometry
import trieu_graph_match
import time

from collections import OrderedDict as odict
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


class Action(object):

  def __init__(self, matched_conclusion, mapping, theorem):
    self.matched_conclusion = matched_conclusion  # A list of list
    self.new_objects = sum(matched_conclusion.topological_list, [])
    self.mapping = mapping
    self.theorem = theorem
    self.premise_objects = [mapping[x] for x in theorem.premise_objects]
    self.conclusion_objects = [mapping[x] for x in theorem.conclusion_objects]
    self.duration = None

  def set_chain_position(self, pos):
    vals = {}
    for obj in self.new_objects:
      if obj.chain_position is None:
        obj.set_chain_position(pos)

      if isinstance(obj, (SegmentHasLength, AngleHasMeasure, LineHasDirection)):
        val = obj.init_list[1]
        if val not in vals:
          val.set_chain_position(pos)
          vals[val] = True
        # if val.name == '1m':
        #   print(val, {x.name: {a.name: b for a, b in y.items()} for x, y in val.edges.items()})

  def draw(self, canvas):
    return self.theorem.draw(self.mapping, canvas)

  def to_str(self):
    names_match = [(x, self.mapping[y].name)
                   for x, y in self.theorem.names.items()]
    conclusion_match = [(x.name, y.name)
                        for x, y in self.mapping.items()
                        if y in self.new_objects
                        and isinstance(y, (Point, Line))
                        ]
    s = self.theorem.name + ': '
    s += ' '.join(
        ['{}::{}'.format(x, y)
         for x, y in sorted(names_match)])
    s += ' => '
    s += ' '.join(
        ['{}::{}'.format(x, y)
         for x, y in sorted(conclusion_match)])
    return s


class FundamentalTheorem(object):

  def __init__(self):
    self.gather_objects()
    self.conclusion.gather_val2objs()

  def draw(self, mapping, canvas):
    return {}

  def gather_objects(self):
    self.premise_objects = set()
    for rel in self.premise:
      obj1, obj2 = rel.init_list
      self.premise_objects.update([rel, obj1, obj2])

    self.conclusion_objects = set()
    for constructions in self.conclusion.topological_list:
      for rel in constructions:
        obj1, obj2 = rel.init_list
        self.conclusion_objects.update([rel, obj1, obj2])

  def get_augmented_relations(self, state):
    return []

  def match_one_random(self, state):
    try:
      constructions, mapping = trieu_graph_match.match_relations(
          premise_relations=self.premise, 
          state_relations=state.relations,
          augmented_relations=self.get_augmented_relations(state),
          conclusion=self.conclusion,
          randomize=True,
          distinct=self.distinct
      ).next()
    except StopIteration:
      return None

    return Action(constructions, mapping, self)

  def match_all(self, state, randomize=True):
    timeout = []
    matches = trieu_graph_match.match_relations(
        premise_relations=self.premise, 
        state_relations=state.relations,
        augmented_relations=self.get_augmented_relations(state),
        conclusion=self.conclusion,
        randomize=randomize,
        distinct=self.distinct,
        timeout=timeout
    )

    try:
      timeout.append(time.time() + self.timeout)
      for constructions, mapping in matches:
        yield Action(constructions, mapping, self)
        timeout[0] = time.time() + self.timeout
    except trieu_graph_match.Timeout:
      return

  def match_from_input_mapping(self, state, mapping, randomize=False):
    # Check if there is a unique match that does not conflict with mapping.
    timeout = []
    matches = trieu_graph_match.match_relations(
        premise_relations=self.premise, 
        state_relations=state.relations,
        augmented_relations=self.get_augmented_relations(state),
        conclusion=self.conclusion,
        distinct=self.distinct,
        randomize=randomize,
        mapping=mapping,
        timeout=None,
    )
    timeout.append(time.time() + self.timeout)
    for matched_conclusion, mapping in matches:
      yield Action(matched_conclusion, mapping, self)
      timeout.append(time.time() + self.timeout)

  @property
  def distinct(self):
    if hasattr(self, '_distinct'):
      return self._distinct
    return None

  @property
  def name(self):
    return type(self).__name__

  @property
  def timeout(self):
    return 0.1
  


class ConstructNormalTriangle(FundamentalTheorem):

  def __init__(self):

    A, B, C = map(Point, 'ABC')
    ab, bc, ca = map(Line, 'ab bc ca'.split())
    AB, BC, CA = map(Segment, 'AB BC CA'.split())

    self.conclusion = Conclusion()
    state.add_relations(
        [A, B, C, ab, bc, ca, AB, BC, CA] +
        segment_def(AB, A, B) +
        segment_def(BC, B, C) +
        segment_def(CA, C, A) +
        collinear(ab, A, B) +
        collinear(bc, B, C) +
        collinear(ca, C, A)
    )



class ConstructMidPoint(FundamentalTheorem):

  def __init__(self):
    A, B = map(Point, 'AB')
    l = Line('l')

    self.premise = collinear(l, A, B)

    C = Point('C')
    CA, CB = Segment('CA'), Segment('CB')
    self.conclusion = Conclusion()
    self.conclusion.add_critical(*(
        collinear(l, C) +
        segment_def(CA, C, A) +
        segment_def(CB, C, B) +
        have_length('l1', CA, CB)
    ))

    self.for_drawing = [C, A, B]
    self.names = dict(A=A, B=B)
    super(ConstructMidPoint, self).__init__()

  def draw(self, mapping, canvas):
    C, A, B = map(mapping.get, self.for_drawing)
    return canvas.add_midpoint(C, A, B)


class ConstructIntersectLineLine(FundamentalTheorem):

  def __init__(self):
    A = Point('A')
    l, l1, l2 = map(Line, 'l l1 l2'.split())

    self.premise = (
        collinear(l1, A) +
        collinear(l2, A) +
        have_direction('d1', l1, l)
    )

    B = Point('B')
    self.conclusion = Conclusion(*(
        collinear(l, B) + collinear(l2, B)
    ))
    # AB = Segment('AB')
    # self.conclusion.add(*segment_def(AB, A, B))

    self.for_drawing = [B, l, l2]
    self.names = dict(l=l, l1=l1, l2=l2)
    super(ConstructIntersectLineLine, self).__init__()

  def draw(self, mapping, canvas):
    B, l, l2 = map(mapping.get, self.for_drawing)
    return canvas.add_intersect_line_line(B, l, l2)



class ConstructIntersectSegmentLine(FundamentalTheorem):

  def __init__(self):
    A, B = map(Point, 'AB')
    ab, l = map(Line, 'ab l'.split())
    l_hp1, l_hp2 = HalfPlane('l_hp1'), HalfPlane('l_hp2')

    self.premise = (
        collinear(ab, A, B) +
        divides_halfplanes(l, l_hp1, l_hp2, A, B)
    )

    C = Point('C')
    self.conclusion = Conclusion()
    self.conclusion.add_critical(*(
        collinear(ab, C) + collinear(l, C)))

    self.for_drawing = [C, l, A, B]

    self.names = dict(l=l, A=A, B=B)
    super(ConstructIntersectSegmentLine, self).__init__()

  def draw(self, mapping, canvas):
    C, l, A, B = map(mapping.get, self.for_drawing)
    return canvas.add_intersect_seg_line(C, l, A, B)


class ConstructPerpendicularLine1(FundamentalTheorem):

  def __init__(self):
    l = Line('l')
    A = Point('A')
    hp1, hp2 = map(HalfPlane, 'hp1 hp2'.split())

    self.premise = collinear(l, A) + divides_halfplanes(l, hp1, hp2)

    l2 = Line('l2')
    self.conclusion = Conclusion()
    self.conclusion.add_critical(*(
        collinear(l2, A) +
        divides_halfplanes(l2, hp3, hp4) +
        angle_def(angle13, hp1, hp3) +
        [AngleHasMeasure(angle13, geometry.halfpi)]
    ))
    self.conclusion.add(*angle_def(angle14, hp1, hp4))
    self.conclusion.add(*angle_def(angle23, hp2, hp3))
    self.conclusion.add(*angle_def(angle24, hp2, hp4))

    self.conclusion.add(AngleHasMeasure(angle14, geometry.halfpi))
    self.conclusion.add(AngleHasMeasure(angle23, geometry.halfpi))
    self.conclusion.add(AngleHasMeasure(angle24, geometry.halfpi))

    self.for_drawing = [l2, A, l]
    self.names = dict(A=A, l=l)

    super(ConstructPerpendicularLine1, self).__init__()

  def draw(self, mapping, canvas):
    l2, A, l = map(mapping.get, self.for_drawing)
    return canvas.add_perp_line(l2, A, l)


class ConstructPerpendicularLine2(FundamentalTheorem):

  def __init__(self):
    l = Line('l')
    A = Point('A')
    hp1, hp2 = map(HalfPlane, 'hp1 hp2'.split())

    self.premise = divides_halfplanes(l, hp1, hp2, A)

    l2 = Line('l2')
    self.conclusion = Conclusion()
    self.conclusion.add_critical(*(
        collinear(l2, A) +
        divides_halfplanes(l2, hp3, hp4) +
        angle_def(angle13, hp1, hp3) +
        [AngleHasMeasure(angle13, geometry.halfpi)]
    ))
    self.conclusion.add(*angle_def(angle14, hp1, hp4))
    self.conclusion.add(*angle_def(angle23, hp2, hp3))
    self.conclusion.add(*angle_def(angle24, hp2, hp4))

    self.conclusion.add(AngleHasMeasure(angle14, geometry.halfpi))
    self.conclusion.add(AngleHasMeasure(angle23, geometry.halfpi))
    self.conclusion.add(AngleHasMeasure(angle24, geometry.halfpi))

    self.for_drawing = [l2, A, l]
    self.names = dict(A=A, l=l)

    super(ConstructPerpendicularLine2, self).__init__()

  def draw(self, mapping, canvas):
    l2, A, l = map(mapping.get, self.for_drawing)
    return canvas.add_perp_line(l2, A, l)


class ConstructParallelLine(FundamentalTheorem):

  def __init__(self):
    l = Line('l')
    A = Point('A')
    hp1, hp2 = map(HalfPlane, 'hp1 hp2'.split())

    self.premise = divides_halfplanes(l, hp1, hp2, A)

    l2 = Line('l2')
    self.conclusion = Conclusion()
    d = LineDirection('d1')
    self.conclusion.add(LineHasDirection(l, d))
    self.conclusion.add_critical(
        LineContainsPoint(l2, A), LineHasDirection(l2, d))
    # self.conclusion.add_critical(
    #     LineHasDirection(l, d),
    #     LineContainsPoint(l2, A),
    #     LineHasDirection(l2, d))

    self.for_drawing = [l2, A, l]
    self.names = dict(A=A, l=l)

    super(ConstructParallelLine, self).__init__()

  def draw(self, mapping, canvas):
    l2, A, l = map(mapping.get, self.for_drawing)
    return canvas.add_parallel_line(l2, A, l)


class ConstructThirdLine(FundamentalTheorem):

  def __init__(self):
    A, B, C = map(Point, 'ABC')
    ac, bc = map(Line, 'ab bc'.split())

    self.premise = collinear(ac, A, C) + collinear(bc, B, C)

    ab = Line('ab')
    AB = Segment('AB')
    self.conclusion = Conclusion(*collinear(ab, A, B))
    self.conclusion.add(*segment_def(AB, A, B))
    # self.conclusion.add(*have_length("1m", AB))

    self.for_drawing = [ab, A, B]
    self.names = dict(A=A, B=B)

    super(ConstructThirdLine, self).__init__()

  def draw(self, mapping, canvas):
    ab, A, B = map(mapping.get, self.for_drawing)
    return canvas.add_line(ab, A, B)


class OppositeAngles(FundamentalTheorem):

  def __init__(self):
    a, b, c, d, m = map(Point, 'XYZTM')
    ab, cd = map(Line, 'xy zt'.split())
    ab_h1, ab_h2, cd_h1, cd_h2 = map(
        HalfPlane, 'xy_h1 xy_h2 zt_h1 zt_h2'.split())

    self.premise = (
        collinear(ab, a, b, m) +  # A, M, B on e
        collinear(cd, c, d, m) +  # C, M, D on f
        divides_halfplanes(ab, ab_h1, ab_h2, c, d) +  # M between a and b
        divides_halfplanes(cd, cd_h1, cd_h2, a, b)  # M between c and d
    )

    amc, bmd = Angle('XMZ'), Angle('YMT')
    amd, bmc = Angle('XMT'), Angle('YMZ')
    self.conclusion = Conclusion()
    self.conclusion.add(*angle_def(amc, ab_h1, cd_h1))
    self.conclusion.add(*angle_def(bmd, ab_h2, cd_h2))
    self.conclusion.add(*angle_def(bmc, ab_h1, cd_h2))
    self.conclusion.add(*angle_def(amd, ab_h2, cd_h1))
    self.conclusion.add_critical(*have_measure('1"', amc, bmd))
    self.conclusion.add_critical(*have_measure('2"', amd, bmc))

    self.names = dict(A=a, M=m, C=c) 

    super(OppositeAngles, self).__init__()


class ParallelBecauseCorrespondingAngles(FundamentalTheorem):

  def __init__(self):
    l1, l1_hp1, _ = line_and_halfplanes('l1')
    l2, l2_hp1, l2_hp2 = line_and_halfplanes('l2')
    l3, l3_hp1, _ = line_and_halfplanes('l3')
    X, Y = Point('X'), Point('Y')

    angle12, angle13 = Angle('l12'), Angle('l13')
    self.premise = (
        collinear(l1, X, Y) +
        collinear(l2, X) +
        collinear(l3, Y) +
        divides_halfplanes(l1, l1_hp1) +
        divides_halfplanes(l3, l3_hp1, p1=X) +
        divides_halfplanes(l2, l2_hp1, l2_hp2, p2=Y) +
        angle_def(angle12, l1_hp1, l2_hp1) +
        angle_def(angle13, l1_hp1, l3_hp1) +
        have_measure('\'1', angle12, angle13)
    )

    self.conclusion = Conclusion()
    d = LineDirection('d')
    self.conclusion.add(LineHasDirection(l2, d))
    self.conclusion.add_critical(LineHasDirection(l3, d))
    # self.conclusion.add_critical(LineHasDirection(l2, d),
    #                              LineHasDirection(l3, d))

    self.names = dict(l=l1, l1=l2, l2=l3)

    super(ParallelBecauseCorrespondingAngles, self).__init__()


class EqualAnglesBecauseParallel(FundamentalTheorem):

  def __init__(self):
    A, B = map(Point, 'AB')
    l, l1, l2 = map(Line, 'l l1 l2'.split())
    l_hp1, l_hp2 = HalfPlane('l_hp1'), HalfPlane('l_hp2')
    l1_hp, l2_hp = HalfPlane('l1_hp'), HalfPlane('l2_hp')

    self.premise = (
        collinear(l, A, B) +
        collinear(l1, A) +
        collinear(l2, B) +
        divides_halfplanes(l, l_hp1, l_hp2) +
        divides_halfplanes(l1, l1_hp, p1=B) +
        divides_halfplanes(l2, l2_hp, p1=A) +
        have_direction('d1', l1, l2)
    )

    angle11, angle12, angle21, angle22 = map(
        Angle, '^11 ^12 ^21 ^22'.split())
    self.conclusion = Conclusion()
    self.conclusion.add(*angle_def(angle11, l_hp1, l1_hp))
    self.conclusion.add(*angle_def(angle12, l_hp1, l2_hp))
    self.conclusion.add(*angle_def(angle21, l_hp2, l1_hp))
    self.conclusion.add(*angle_def(angle22, l_hp2, l2_hp))

    m1, m2 = AngleMeasure('1"'), AngleMeasure('2"')
    self.conclusion.add(AngleHasMeasure(angle11, m1))
    self.conclusion.add_critical(AngleHasMeasure(angle22, m1))
    # self.conclusion.add_critical(AngleHasMeasure(angle11, m1),
    #                              AngleHasMeasure(angle22, m1))

    self.conclusion.add(AngleHasMeasure(angle12, m2))
    self.conclusion.add_critical(AngleHasMeasure(angle21, m2))
    # self.conclusion.add_critical(AngleHasMeasure(angle12, m2),
    #                              AngleHasMeasure(angle21, m2))

    self.names = dict(l=l, l1=l1, l2=l2)
    super(EqualAnglesBecauseParallel, self).__init__()



class Congruences(FundamentalTheorem):

  def get_augmented_relations(self, state):
    return state.augmented_relations()


class SAS(Congruences):

  def __init__(self):
    A, B, C, D, E, F = map(Point, 'ABCDEF')
    ab, ab_hp1, _ = line_and_halfplanes('ab')
    bc, bc_hp1, _ = line_and_halfplanes('bc')
    de, de_hp1, _ = line_and_halfplanes('de')
    ef, ef_hp1, _ = line_and_halfplanes('ef')
    ABC, DEF = Angle('ABC'), Angle('DEF')
    AB, BC, DE, EF = map(Segment, ['AB', 'BC', 'DE', 'EF'])

    self.premise = (
        collinear(ab, A, B) +
        collinear(bc, B, C) +
        collinear(de, D, E) +
        collinear(ef, E, F) +

        divides_halfplanes(ab, ab_hp1, p1=C) +
        divides_halfplanes(bc, bc_hp1, p1=A) +

        divides_halfplanes(de, de_hp1, p1=F) +
        divides_halfplanes(ef, ef_hp1, p1=D) +
        
        segment_def(AB, A, B) +
        segment_def(BC, B, C) +
        segment_def(DE, D, E) +
        segment_def(EF, E, F) +
        
        angle_def(ABC, ab_hp1, bc_hp1) +
        angle_def(DEF, de_hp1, ef_hp1) +
        
        have_length('l1', AB, DE) +
        have_length('l2', BC, EF) +
        have_measure('\'0', ABC, DEF)
    )

    conclusion = Conclusion()

    AC, DF = Segment('AC'), Segment('DF')
    BAC, BCA, EDF, EFD = map(Angle, 'BAC BCA EDF EFD'.split())
    ac, ac_hp1, _ = line_and_halfplanes('ac')
    df, df_hp1, _ = line_and_halfplanes('df')

    conclusion.add(*collinear(ac, A, C))
    conclusion.add(*collinear(df, D, F))

    conclusion.add(*divides_halfplanes(ac, ac_hp1, p1=B))
    conclusion.add(*divides_halfplanes(df, df_hp1, p1=E))

    conclusion.add(*segment_def(AC, A, C))
    conclusion.add(*segment_def(DF, D, F))

    conclusion.add(*angle_def(BAC, ab_hp1, ac_hp1))
    conclusion.add(*angle_def(BCA, bc_hp1, ac_hp1))
    conclusion.add(*angle_def(EDF, de_hp1, df_hp1))
    conclusion.add(*angle_def(EFD, ef_hp1, df_hp1))

    l1, m1, m2 = SegmentLength('l1'), AngleMeasure('1"'), AngleMeasure('2"')
    conclusion.add(SegmentHasLength(AC, l1))
    conclusion.add_critical(SegmentHasLength(DF, l1))
    # conclusion.add_critical(SegmentHasLength(AC, l1),
    #                         SegmentHasLength(DF, l1))

    conclusion.add(AngleHasMeasure(BAC, m1))
    conclusion.add_critical(AngleHasMeasure(EDF, m1))
    # conclusion.add_critical(AngleHasMeasure(BAC, m1),
    #                         AngleHasMeasure(EDF, m1))

    conclusion.add(AngleHasMeasure(BCA, m2))
    conclusion.add_critical(AngleHasMeasure(EFD, m2))
    # conclusion.add_critical(AngleHasMeasure(BCA, m2),
    #                         AngleHasMeasure(EFD, m2))

    self.conclusion = conclusion
    self._distinct = [(AB, DE), 
                      (AB, BC), (BC, AC), (AC, AB),
                      (DE, EF), (EF, DF), (DF, DE),
                      (ab, bc), (bc, ac), (ac, ab),
                      (de, ef), (ef, df), (df, de),
                      (BAC, BCA), (BCA, ABC), (ABC, BAC),
                      (DEF, EDF), (EDF, EFD), (EFD, DEF),
                      (A, B), (B, C), (C, A),
                      (D, E), (E, F), (F, D),
                      ]

    self.for_drawing = [ac, A, C, df, D, F]

    self.names = dict(A=A, B=B, C=C, D=D, E=E, F=F)

    super(SAS, self).__init__()

  def draw(self, mapping, canvas):
    ac, A, C, df, D, F = map(mapping.get, self.for_drawing)
    info = {}
    if ac not in canvas.lines:
      info.update(canvas.add_line(ac, A, C))
    if df not in canvas.lines:
      info.update(canvas.add_line(df, D, F))
    return info

  @property
  def timeout(self):
    return 0.5


class SSS(Congruences):

  def __init__(self):
    A, B, C, D, E, F = map(Point, 'ABCDEF')
    AB, BC, CA, DE, EF, FE = map(
        Segment, ['AB', 'BC', 'CA', 'DE', 'EF', 'FD'])

    self.premise = (
        segment_def(AB, A, B) +
        segment_def(BC, B, C) +
        segment_def(CA, C, A) +
        segment_def(DE, D, E) +
        segment_def(EF, E, F) +
        segment_def(FD, F, D) +

        have_length('l1', AB, DE) +
        have_length('l2', BC, EF) +
        have_length('l3', CA, FD)
    )

    conclusion = Conclusion()

    ab, ab_hp1, _ = line_and_halfplanes('ab')
    bc, bc_hp1, _ = line_and_halfplanes('bc')
    ca, ca_hp1, _ = line_and_halfplanes('ca')

    de, de_hp1, _ = line_and_halfplanes('de')
    ef, ef_hp1, _ = line_and_halfplanes('ef')
    fd, fd_hp1, _ = line_and_halfplanes('fd')

    conclusion.add(*collinear(ab, A, B))
    conclusion.add(*collinear(bc, B, C))
    conclusion.add(*collinear(ca, C, A))
    conclusion.add(*collinear(de, D, E))
    conclusion.add(*collinear(ef, E, F))
    conclusion.add(*collinear(fd, F, D))

    conclusion.add(*divides_halfplanes(ab, ab_hp1, p1=C))
    conclusion.add(*divides_halfplanes(bc, bc_hp1, p1=A))
    conclusion.add(*divides_halfplanes(ca, ca_hp1, p1=B))

    conclusion.add(*divides_halfplanes(de, de_hp1, p1=F))
    conclusion.add(*divides_halfplanes(ef, ef_hp1, p1=D))
    conclusion.add(*divides_halfplanes(fd, fd_hp1, p1=E))

    conclusion.add(*angle_def(ABC, ab_hp1, bc_hp1))
    conclusion.add(*angle_def(BCA, bc_hp1, ca_hp1))
    conclusion.add(*angle_def(CAB, ca_hp1, ab_hp1))

    conclusion.add(*angle_def(DEF, de_hp1, ef_hp1))
    conclusion.add(*angle_def(EFD, ef_hp1, fd_hp1))
    conclusion.add(*angle_def(FDE, fd_hp1, de_hp1))

    conclusion.add_critical(*have_measure('\'1', ABC, DEF))
    conclusion.add_critical(*have_measure('\'2', BCA, EFD))
    conclusion.add_critical(*have_measure('\'2', CAB, FDE))

    self.conclusion = conclusion
    self._distinct = [(AB, DE),
                      (AB, BC), (BC, AC), (AC, AB),
                      (DE, EF), (EF, DF), (DF, DE),
                      (ab, bc), (bc, ac), (ac, ab),
                      (de, ef), (ef, df), (df, de),
                      (BAC, BCA), (BCA, ABC), (ABC, BAC),
                      (DEF, EDF), (EDF, EFD), (EFD, DEF),
                      (A, B), (B, C), (C, A),
                      (D, E), (E, F), (F, D),
                      ]

    self.for_drawing = [ab, bc, ca, de, ef, fd,
                        A, B, C, D, E, F]
    self.names = dict(A=A, B=B, C=C, D=D, E=E, F=F)

    super(SAS, self).__init__()

  def draw(self, mapping, canvas):
    (ab, bc, ca, de, ef, fd,
     A, B, C, D, E, F) = map(mapping.get, self.for_drawing)
    info = {}
    if ab not in canvas.lines:
      info.update(canvas.add_line(ab, A, B))
    if bc not in canvas.lines:
      info.update(canvas.add_line(bc, B, C))
    if ca not in canvas.lines:
      info.update(canvas.add_line(ca, C, A))
    if de not in canvas.lines:
      info.update(canvas.add_line(de, D, E))
    if ef not in canvas.lines:
      info.update(canvas.add_line(ef, E, F))
    if fd not in canvas.lines:
      info.update(canvas.add_line(fd, F, D))
    return info

  @property
  def timeout(self):
    return 0.5


class ASA(Congruences):

  def __init__(self):
    A, B, C, D, E, F = map(Point, 'ABCDEF')
    ab, ab_hp1, _ = line_and_halfplanes('ab')
    bc, bc_hp1, _ = line_and_halfplanes('bc')
    ca, ca_hp1, _ = line_and_halfplanes('ca')
    de, de_hp1, _ = line_and_halfplanes('de')
    ef, ef_hp1, _ = line_and_halfplanes('ef')
    fd, fd_hp1, _ = line_and_halfplanes('fd')
    BAC, BCA, EDF, EFD = map(
        Angle, 'BAC BCA EDF EFD'.split())
    CA, FD = map(Segment, ['CA', 'FD'])

    self.premise = (
        collinear(ab, A, B) +
        collinear(bc, B, C) +
        collinear(ca, C, A) +

        collinear(de, D) +
        collinear(ef, F) +
        collinear(fd, F, D) +

        divides_halfplanes(ab, ab_hp1, p1=C) +
        divides_halfplanes(bc, bc_hp1, p1=A) +
        divides_halfplanes(ca, ca_hp1, p1=B) +

        divides_halfplanes(de, de_hp1, p1=F) +
        divides_halfplanes(ef, ef_hp1, p1=D) +
        divides_halfplanes(fd, fd_hp1) +

        # segment_def(AB, A, B) +
        segment_def(CA, C, A) +
        # segment_def(DE, D, E) +
        segment_def(FD, F, D) +

        angle_def(BAC, ab_hp1, ca_hp1) +
        angle_def(BCA, bc_hp1, ca_hp1) +
        angle_def(EDF, de_hp1, fd_hp1) +
        angle_def(EFD, ef_hp1, fd_hp1) +

        have_length('l1', CA, FD) +
        have_measure('0"', BAC, EDF) +
        have_measure('1"', BCA, EFD)
    )

    self.conclusion = Conclusion()

    # Now we introduce point E:
    self.conclusion.add(LineContainsPoint(de, E),
                        LineContainsPoint(ef, E))
                        # HalfPlaneContainsPoint(fd_hp1, E))

    ABC, DEF = Angle('ABC'), Angle('DEF')
    self.conclusion.add(*angle_def(DEF, de_hp1, ef_hp1))
    self.conclusion.add(*angle_def(ABC, ab_hp1, bc_hp1))

    AB, BC, DE, EF = map(Segment, ['AB', 'BC', 'DE', 'EF'])
    self.conclusion.add(*segment_def(AB, A, B))
    self.conclusion.add(*segment_def(BC, B, C))
    self.conclusion.add(*segment_def(DE, D, E))
    self.conclusion.add(*segment_def(EF, E, F))

    l1, l2, m0 = SegmentLength('1m'), SegmentLength('2m'), AngleMeasure('0"')

    self.conclusion.add(SegmentHasLength(AB, l1)) 
    self.conclusion.add_critical(SegmentHasLength(DE, l1)) 
    # self.conclusion.add_critical(SegmentHasLength(AB, l1),
    #                              SegmentHasLength(DE, l1)) 

    self.conclusion.add(SegmentHasLength(BC, l2))
    self.conclusion.add_critical(SegmentHasLength(EF, l2))
    # self.conclusion.add_critical(SegmentHasLength(BC, l2),
    #                              SegmentHasLength(EF, l2))

    self.conclusion.add(AngleHasMeasure(ABC, m0)) 
    self.conclusion.add_critical(AngleHasMeasure(DEF, m0)) 
    # self.conclusion.add_critical(AngleHasMeasure(ABC, m0),
    #                              AngleHasMeasure(DEF, m0)) 

    # self.conclusion.add_critical(*have_length('l1', AB, DE))
    # self.conclusion.add_critical(*have_length('l2', BC, EF))
    # self.conclusion.add_critical(*have_measure('0"', ABC, DEF))

    self._distinct = [(A, D),
                      (AB, CA), (AB, BC), (CA, BC),
                      (DE, FD), (DE, EF), (FD, EF),
                      (BAC, BCA), (BAC, ABC), (BCA, ABC),
                      (EDF, EFD), (EDF, DEF), (EFD, DEF),
                      (ab, bc), (bc, ca), (ca, ab),
                      (de, ef), (ef, fd), (fd, de),
                      (A, B), (B, C), (C, A), 
                      (D, E), (E, F), (F, D),
                      (ab_hp1, bc_hp1), (ab_hp1, ca_hp1), (bc_hp1, ca_hp1),
                      (de_hp1, ef_hp1), (de_hp1, fd_hp1), (ef_hp1, fd_hp1)
                      ]

    self.names = dict(A=A, B=B, C=C, D=D, F=F, de=de, ef=ef)
    self.for_drawing = [E, de, ef]
    super(ASA, self).__init__()

  def draw(self, mapping, canvas):
    E, de, ef = map(mapping.get, self.for_drawing)
    if E not in canvas.points:
      return canvas.add_intersect_line_line(E, de, ef)
    return {}

  @property
  def timeout(self):
    return 3.0
  


class ASA_(Congruences):

  def __init__(self):
    A, B, C, D, E, F = map(Point, 'ABCDEF')
    ab, ab_hp1, _ = line_and_halfplanes('ab')
    bc, bc_hp1, _ = line_and_halfplanes('bc')
    ca, ca_hp1, _ = line_and_halfplanes('ca')
    de, de_hp1, _ = line_and_halfplanes('de')
    ef, ef_hp1, _ = line_and_halfplanes('ef')
    fd, fd_hp1, _ = line_and_halfplanes('fd')
    BAC, BCA, EDF, EFD = map(
        Angle, 'BAC BCA EDF EFD'.split())
    AB, BC, DE, EF = map(Segment, ['AB', 'BC', 'DE', 'EF'])
    CA, FD = map(Segment, ['CA', 'FD'])

    self.premise = (
        collinear(ab, A, B) +
        collinear(bc, B, C) +
        collinear(ca, C, A) +

        collinear(de, D, E) +
        collinear(ef, E, F) +
        collinear(fd, F, D) +

        divides_halfplanes(ab, ab_hp1, p1=C) +
        divides_halfplanes(bc, bc_hp1, p1=A) +
        divides_halfplanes(ca, ca_hp1, p1=B) +

        divides_halfplanes(de, de_hp1, p1=F) +
        divides_halfplanes(ef, ef_hp1, p1=D) +
        divides_halfplanes(fd, fd_hp1, p1=E) +

        # segment_def(AB, A, B) +
        segment_def(CA, C, A) +
        # segment_def(DE, D, E) +
        segment_def(FD, F, D) +

        angle_def(BAC, ab_hp1, ca_hp1) +
        angle_def(BCA, bc_hp1, ca_hp1) +
        angle_def(EDF, de_hp1, fd_hp1) +
        angle_def(EFD, ef_hp1, fd_hp1) +

        have_length('l1', CA, FD) +
        have_measure('0"', BAC, EDF) +
        have_measure('1"', BCA, EFD)
    )

    self.conclusion = Conclusion()

    ABC, DEF = Angle('ABC'), Angle('DEF')
    self.conclusion.add(*angle_def(ABC, ab_hp1, bc_hp1))
    self.conclusion.add(*angle_def(DEF, de_hp1, ef_hp1))

    self.conclusion.add(*segment_def(AB, A, B))
    self.conclusion.add(*segment_def(BC, B, C))
    self.conclusion.add(*segment_def(DE, D, E))
    self.conclusion.add(*segment_def(EF, E, F))

    self.conclusion.add_critical(*have_length('l1', AB, DE))
    self.conclusion.add_critical(*have_length('l2', BC, EF))
    self.conclusion.add_critical(*have_measure('0"', ABC, DEF))

    self._distinct = [(A, D),
                      (AB, CA), (AB, BC), (CA, BC),
                      (DE, FD), (DE, EF), (FD, EF),
                      (BAC, BCA), (BAC, ABC), (BCA, ABC),
                      (EDF, EFD), (EDF, DEF), (EFD, DEF),
                      (ab, bc), (bc, ca), (ca, ab),
                      (de, ef), (ef, fd), (fd, de),
                      (A, B), (B, C), (C, A), 
                      (D, E), (E, F), (F, D),
                      (ab_hp1, bc_hp1), (ab_hp1, ca_hp1), (bc_hp1, ca_hp1),
                      (de_hp1, ef_hp1), (de_hp1, fd_hp1), (ef_hp1, fd_hp1)
                      ]

    self.names = dict(A=A, B=B, C=C, D=D, E=E, F=F)
    super(ASA, self).__init__()


all_theorems = {
    'mid': ConstructMidPoint(),  # 0.000365972518921
    'seg_line': ConstructIntersectSegmentLine(),
    'parallel': ConstructParallelLine(),
    'line': ConstructThirdLine(),
    'eq': EqualAnglesBecauseParallel(),  # 1.73088312149
    'sas': SAS(),  # 0.251692056656
    'asa': ASA(),  # 2.26002907753 3.96637487411
    '.parallel': ParallelBecauseCorrespondingAngles()
}
