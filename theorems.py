"""Implement the environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Mapping

import theorems_utils
import geometry
import trieu_graph_match
import time

from collections import OrderedDict as odict
from collections import defaultdict as ddict

from profiling import Timer

from theorems_utils import collinear, concyclic, in_halfplane
from theorems_utils import divides_halfplanes, line_and_halfplanes
from theorems_utils import have_length, have_measure, have_direction
from theorems_utils import segment_def, fangle_def
from theorems_utils import diff_side, same_side, distinct
from state import State, Conclusion

from geometry import GeometryEntity, Merge, DistinctPoint, DistinctLine
from geometry import Point, Line, Segment, Angle, HalfPlane, Circle, SelectAngle
from geometry import AngleOfFullAngle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, LineBordersHalfplane
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


class Action(object):

  def __init__(self, matched_conclusion, mapping, state, theorem):
    self.matched_conclusion = matched_conclusion  # a Conclusion object.
    self.new_objects = sum(matched_conclusion.topological_list, [])
    self.mapping = mapping
    self.theorem = theorem
    self.state = state

    self.premise_objects = []
    for x in theorem.premise_objects:
      if isinstance(x, SelectAngle):
        x = x.select(mapping)
      if x in mapping:
        self.premise_objects.append(mapping[x])

    # List of Merge() relations associcated with this action.
    self.merges = [obj for obj in self.new_objects if isinstance(obj, Merge)]

    if any([isinstance(x, Merge) for x in theorem.conclusion_objects]):
      self.conclusion_objects = self.new_objects
    else:
      self.conclusion_objects = [
          mapping[x.select(mapping)] if isinstance(x, SelectAngle) 
          else mapping[x] for x in theorem.conclusion_objects]
    self.duration = None

  def update(self, other):
    assert isinstance(self.theorem, MergeTheorem), self.theorem
    assert isinstance(other.theorem, MergeTheorem)
    assert len(self.matched_conclusion.topological_list) == 1 
    assert len(other.matched_conclusion.topological_list) == 1 

    self.matched_conclusion.topological_list[0] += other.matched_conclusion.topological_list[0]
    self.new_objects += other.new_objects

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

      if isinstance(obj, Merge):
        from_obj, to_obj = obj.init_list
        obj.merge_graph[to_obj][from_obj] = pos
        obj.merge_graph[from_obj][to_obj] = pos

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
        ['{}={}'.format(x, y)
         for x, y in sorted(names_match)])
    s += ' => '
    s += ' '.join(
        ['{}={}'.format(x, y)
         for x, y in sorted(conclusion_match)])
    return s


class FundamentalTheorem(object):

  def __init__(self):
    if not hasattr(self, 'premise'):
      self.premise = []
    if not hasattr(self, 'conclusion'):
      self.conclusion = Conclusion()
    if not hasattr(self, '_distinct'):
      self._distinct = []
    if not hasattr(self, 'names'):
      self.names = {}

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

  def match_premise(self, state, mapping=None):
    try:
      constructions, mapping = trieu_graph_match.match_relations(
          premise_relations=self.premise, 
          state=state,
          conclusion=None,
          randomize=False,
          mapping=None,
          distinct=self.distinct
      ).next()
    except StopIteration:
      return None

    return Action(constructions, mapping, state, self)

  def match_one_random(self, state):
    try:
      constructions, mapping = trieu_graph_match.match_relations(
          premise_relations=self.premise, 
          state=state,
          conclusion=self.conclusion,
          randomize=True,
          distinct=self.distinct
      ).next()
    except StopIteration:
      return None

    return Action(constructions, mapping, state, self)

  def match_all(self, state, randomize=True):
    timeout = []
    matches = trieu_graph_match.match_relations(
        premise_relations=self.premise, 
        state=state,
        conclusion=self.conclusion,
        randomize=randomize,
        distinct=self.distinct,
        timeout=timeout
    )

    try:
      timeout.append(time.time() + self.timeout)
      for constructions, mapping in matches:
        yield Action(constructions, mapping, state, self)
        timeout[0] = time.time() + self.timeout
    except trieu_graph_match.Timeout:
      return

  def match_from_input_mapping(self, state, mapping, randomize=False):
    # Check if there is a unique match that does not conflict with mapping.
    timeout = []
    matches = trieu_graph_match.match_relations(
        premise_relations=self.premise, 
        state=state,
        conclusion=self.conclusion,
        distinct=self.distinct,
        randomize=randomize,
        mapping=mapping,
        timeout=None,
    )
    timeout.append(time.time() + self.timeout)
    for matched_conclusion, mapping in matches:
      yield Action(matched_conclusion, mapping, state, self)
      timeout.append(time.time() + self.timeout)

  @property
  def distinct(self):
    if hasattr(self, '_distinct'):
      return self._distinct
    return None

  @property
  def name(self):
    s = ''
    for char in type(self).__name__:
      s += char if char == char.lower() else ' ' + char
    return s.strip()

  @property
  def timeout(self):
    return 0.1


# class Check(object):

#   def found(self, state, goal_objects):
#     _, rel1, rel2 = goal_objects
#     obj1 = rel1.init_list[0]
#     obj2 = rel2.init_list[0]

#     seg1, seg2 = self.equals

#     mapping1 = {obj1: seg1, obj2: seg2}
#     mapping2 = {obj1: seg2, obj2: seg1}

#     matches_gen1 = trieu_graph_match.match_relations(
#         premise_relations=self.premise,
#         state=state,
#         mapping=mapping1
#     )

#     matches_gen2 = trieu_graph_match.match_relations(
#         premise_relations=self.premise,
#         state=state,
#         mapping=mapping2
#     )

#     def matched(matches_gen):
#       try:
#         matches_gen.next()
#         return True
#       except StopIteration:
#         return False

#     return matched(matches_gen2) or matched(matches_gen1)


# class ThalesCheck(Check):

#   def __init__(self):
#     A, B, C, M, N = map(Point, 'ABCMN')
#     l, ab, bc, ca = map(Line, 'l ab bc ca'.split())
#     MA, MB, NA, NC = map(Segment, 'MA MB NA NC'.split())

#     self.premise = (
#         collinear(ab, A, B, M) +
#         segment_def(MA, M, A) +
#         segment_def(MB, M, B) +
#         have_length('1m', MA, MB) +
#         collinear(bc, B, C) +
#         collinear(l, M, N) +
#         have_direction('d1', bc, l) +
#         collinear(ca, C, A, N) +
#         segment_def(NA, N, A) +
#         segment_def(NC, N, C)
#     )

#     self.equals = [NA, NC]


# class OppositeAnglesCheck(Check):

#   def __init__(self):
#     l1, l1_hp1, l1_hp2 = line_and_halfplanes('l1')
#     l2, l2_hp1, l2_hp2 = line_and_halfplanes('l2')
#     angle11, angle22 = Angle('^11'), Angle('^22')

#     self.premise = (
#         divides_halfplanes(l1, l1_hp1, l1_hp2) +
#         divides_halfplanes(l2, l2_hp1, l2_hp2) +
#         angle_def(angle11, l1_hp1, l2_hp1) +
#         angle_def(angle22, l1_hp2, l2_hp2)
#     )

#     self.equals = [angle11, angle22]


# class ConstructNormalTriangle(FundamentalTheorem):

#   def __init__(self):

#     A, B, C = map(Point, 'ABC')
#     ab, bc, ca = map(Line, 'ab bc ca'.split())
#     AB, BC, CA = map(Segment, 'AB BC CA'.split())

#     self.conclusion = Conclusion()
#     state.add_relations(
#         [A, B, C, ab, bc, ca, AB, BC, CA] +
#         segment_def(AB, A, B) +
#         segment_def(BC, B, C) +
#         segment_def(CA, C, A) +
#         collinear(ab, A, B) +
#         collinear(bc, B, C) +
#         collinear(ca, C, A)
#     )

# class AutoMerge(FundamentalTheorem):

#   def find_auto_merge_from_trigger(
#       self, state_candidates, object_mappings):

#     pairs = []
#     for mapping in trieu_graph_match.recursively_match(
#         query_relations=self.premise,
#         state_candidates=state_candidates,
#         object_mappings=object_mappings,
#         distinct=self.distinct,
#         match_all=1):

#       x, y = map(mapping.get, self.merge_pair)
#       if (x, y) not in pairs and (y, x) not in pairs:
#         pairs.append((x, y))
#     return pairs

"""
Theorem set 1. Merge (=>)

1. Point => Segment (Length)
2. Line => HP => HalfAngle
3. Line (LineDirection) => Fangle, Angle (Measure)
3. Point <=> Lines
"""


class MergeTheorem(FundamentalTheorem):

  def __init__(self):
    self._distinct = []
    for l in self.conclusion.topological_list:
      for m in l:
        assert isinstance(m, Merge), 'MergeTheorem only accepts Merge Conclusions'
        self._distinct.append((m.from_obj, m.to_obj))
    super(MergeTheorem, self).__init__()


class SameSegmentBecauseSamePoint(MergeTheorem):

  def __init__(self):
    A, B = map(Point, 'A B'.split())
    AB, AB2 = map(Segment, 'AB AB2'.split())

    self.premise = (
        segment_def(AB, A, B) +
        segment_def(AB2, A, B) +
        distinct(A, B)
    )
    self.trigger_obj = A
    self.conclusion = Conclusion()
    self.conclusion.add_critical(Merge(AB, AB2))
    self.names = dict(A=A, B=B)
    super(SameSegmentBecauseSamePoint, self).__init__()


class SameHalfplaneBecauseSameLine(MergeTheorem):

  def __init__(self):
    l = Line('l')
    hp1, hp2 = HalfPlane('hp1'), HalfPlane('hp2')
    A = Point('A')

    self.premise = (
        divides_halfplanes(l, hp1, p1=A) +
        divides_halfplanes(l, hp2, p1=A)
    )
    self.trigger_obj = l
    self.conclusion = Conclusion()
    self.conclusion.add_critical(Merge(hp1, hp2))
    self.names = dict(hp1=hp1, hp2=hp2)
    super(SameHalfplaneBecauseSameLine, self).__init__()


# class SameHalfAngleBecauseSameHalfPlane(MergeTheorem):

#   def __init__(self):
#     l1, l2 = Line('l1'), Line('l2')
#     hp1, hp2 = HalfPlane('hp1'), HalfPlane('hp2')
#     hangle1, hangle2 = HalfAngle('ha_1'), HalfAngle('ha_2')

#     self.premise = (
#         divides_halfplanes(l1, hp1) +
#         divides_halfplanes(l2, hp2) +
#         hangle_def(hangle1, hp1, hp2) +
#         hangle_def(hangle2, hp1, hp2) +
#         distinct(l1, l2)
#     )
#     self.trigger_obj = hp1
#     self.conclusion = Conclusion()
#     self.conclusion.add_critical(Merge(hangle1, hangle2))
#     self.names = dict(hp1=hp1, hp2=hp2)
#     super(SameHalfAngleBecauseSameHalfPlane, self).__init__()


# class SameAngleBecauseSameHangle(MergeTheorem):

#   def __init__(self):
#     hangle = HalfAngle('ha')
#     angle1, angle2 = Angle('a1'), Angle('a2')

#     self.premise = [
#         HalfAngleOfAngle(hangle, angle1),
#         HalfAngleOfAngle(hangle, angle2)
#     ]
#     self.trigger_obj = hangle

#     self.conclusion = Conclusion()

#     self.conclusion.add_critical(Merge(angle1, angle2))
#     self.names = dict(a1=angle1, a2=angle2)
    
#     super(SameAngleBecauseSameHangle, self).__init__()


# class SameFangleBecauseSameLineDirection(MergeTheorem):

#   def __init__(self):
#     l1, l2 = Line('d1'), Line('d2')
#     d1, d2 = LineDirection('d1'), LineDirection('d2')
#     fangle1, fangle2 = FullAngle('fa_1'), HalfAngle('fa_2')

#     self.premise = (
#         [LineHasDirection(l1, d1),
#          LineHasDirection(l2, d2)] +
#         distinct(l1, l2) + 
#         fangle_def(fangle1, d1, d2) +
#         fangle_def(fangle2, d1, d2) 
#     )
#     self.trigger_obj = d1
#     self.conclusion = Conclusion()

#     self.conclusion.add_critical(Merge(fangle1, fangle2))
#     self.names = dict(d1=d1, d2=d2)

#     super(SameFangleBecauseSameLineDirection, self).__init__()


# class SameFangleBecauseSameAngle(MergeTheorem):

#   def __init__(self):
#     angle1, angle2 = Angle('a1'), Angle('a2')
#     fangle1, fangle2 = FullAngle('fa1'), HalfAngle('fa2')

#     self.premise = (
#         fangle_def(fangle1, angle1=angle1, angle2=angle2) +
#         fangle_def(fangle2, angle1=angle1, angle2=angle2)
#     )
#     self.trigger_obj = angle1
#     self.conclusion = Conclusion()

#     self.conclusion.add_critical(Merge(fangle1, fangle2))
#     self.names = dict(a1=angle1, a2=angle2)

#     super(SameFangleBecauseSameAngle, self).__init__()


class SameLineBecauseSamePoint(MergeTheorem):

  def __init__(self):
    l1, l2 = Line('l1'), Line('l2')
    A, B = Point('A'), Point('B')

    self.premise = (
      collinear(l1, A, B) +
      collinear(l2, A, B) +
      distinct(A, B)
    )

    self.trigger_obj = A
    self.merge_pair = (l1, l2)

    self.conclusion = Conclusion()
    self.conclusion.add_critical(Merge(l1, l2))

    self.for_drawing = []
    self.names = dict(l1=l1, l2=l2, A=A, B=B)

    super(SameLineBecauseSamePoint, self).__init__()


class SamePointBecauseSameLine(MergeTheorem):

  def __init__(self):
    l1, l2 = Line('l1'), Line('l2')
    A, B = Point('A'), Point('B')

    self.premise = (
      collinear(l1, A, B) +
      collinear(l2, A, B) +
      distinct(l1, l2)
    )

    self.trigger_obj = l1
    self.merge_pair = (A, B)

    self.conclusion = Conclusion()
    self.conclusion.add_critical(Merge(A, B))

    self.for_drawing = []
    self.names = dict(l1=l1, l2=l2, A=A, B=B)

    super(SamePointBecauseSameLine, self).__init__()


class SamePointBecauseSameSideEqualDistance(MergeTheorem):

  def __init__(self):
    l, l1 = Line('l'), Line('l1')
    l1_hp = HalfPlane('l1_hp')
    A, B, C = Point('A'), Point('B'), Point('C')
    AB, AC = Segment('AB'), Segment('AC')

    self.premise = (
        collinear(l, A, B, C) +
        collinear(l1, A) +
        divides_halfplanes(l1, l1_hp, p1=B) +  # l1!=l, B!=A
        [HalfPlaneContainsPoint(l1_hp, C)] +  # C!=A
        segment_def(AB, A, B) +
        segment_def(AC, A, C) +
        have_length('1m', AB, AC)
    )

    self.conclusion = Conclusion()
    self.conclusion.add_critical(Merge(B, C))

    self.for_drawing = []
    self.names = dict(A=A, B=B, C=C)
  
    super(SamePointBecauseSameSideEqualDistance, self).__init__()


class SameLineBecauseSameDirection(MergeTheorem):

  def __init__(self):
    l1, l2 = Line('l1'), Line('l2')
    A = Point('A')

    self.premise = (
        collinear(l1, A) +
        collinear(l2, A) +
        # both l1 and l2 has the same direction
        have_direction('d1', l1, l2)
    )
    self.conclusion = Conclusion()
    self.conclusion.add_critical(Merge(l1, l2))

    self.for_drawing = []
    self.names = dict(l1=l1, l2=l2)

    super(SameLineBecauseSameDirection, self).__init__()

  def draw(self, mapping, canvas):
    return {}


class SamePointBecauseSameDistances(MergeTheorem):

  def __init__(self):
    l = Line('l')
    AM, BM, AN, BN = map(Segment, 'AM BM AN BN'.split())
    A, B, M, N = map(Point, 'A B M N'.split())

    length1, length2 = SegmentLength('lAM'), SegmentLength('lBM')

    self.premise = (
        collinear(l, A, B, M, N) +
        distinct(A, B) +
        segment_def(AM, A, M) +
        segment_def(BM, B, M) +
        segment_def(AN, A, N) +
        segment_def(BN, B, N) +
        have_length(length1, AM, AN) +
        have_length(length2, BM, BN)
    )

    self.conclusion = Conclusion()
    self.conclusion.add_critical(Merge(M, N))
    self.for_drawing = [A, B, M, N]
    self.names = dict(A=A, B=B, M=M, N=N)

    super(SamePointBecauseSameDistances, self).__init__()

  def draw(self, mapping, canvas):
    # return canvas.remove_point(N)
    return {}


class ConstructRightAngle(FundamentalTheorem):

  def __init__(self):
    self.premise = []

    self.conclusion = Conclusion()

    halfpi = geometry.get_halfpi()
    halfpi_measure = AngleMeasure('^90degree')
    self.conclusion.add_critical(
        *have_measure(halfpi_measure, halfpi))
    self.names = {}
    super(ConstructRightAngle, self).__init__()


class ConstructMidPoint(FundamentalTheorem):

  def __init__(self):
    A, B = map(Point, 'AB')

    self.premise = distinct(A, B)

    self.conclusion = Conclusion()
    l = Line('l')
    self.conclusion.add(*collinear(l, A, B))
    
    C = Point('C')
    CA, CB = Segment('CA'), Segment('CB')    
    self.conclusion.add_critical(*(  #  all C related
        collinear(l, C) +
        segment_def(CA, C, A) +
        segment_def(CB, C, B) +
        have_length('lCA', CA, CB) +
        distinct(A, C) + distinct(C, B)
    ))

    self.for_drawing = [C, A, B, l]
    self.names = dict(A=A, B=B)
    super(ConstructMidPoint, self).__init__()

  def draw(self, mapping, canvas):
    C, A, B, l = map(mapping.get, self.for_drawing)
    info = canvas.add_line(l, A, B)
    info.update(canvas.add_midpoint(C, A, B))
    return info


class ConstructMirrorPoint(FundamentalTheorem):

  def __init__(self):
    A, B = map(Point, 'AB')

    self.premise = distinct(A, B)

    self.conclusion = Conclusion()

    l = Line('l')
    self.conclusion.add(*collinear(l, A, B))

    AB = Segment('AB')
    lAB = SegmentLength('lAB')
    self.conclusion.add(*segment_def(AB, A, B))
    self.conclusion.add(SegmentHasLength(AB, lAB))

    C = Point('C')
    BC = Segment('BC')
    self.conclusion.add_critical(*(  # C related.
        collinear(l, C) +
        segment_def(BC, B, C) +
        [SegmentHasLength(BC, lAB)] +
        distinct(A, C) + distinct(C, B)
    ))

    self.for_drawing = [C, A, B, l]
    self.names = dict(A=A, B=B)
    super(ConstructMirrorPoint, self).__init__()

  def draw(self, mapping, canvas):
    C, A, B, l = map(mapping.get, self.for_drawing)
    info = canvas.add_line(l, A, B)
    info.update(canvas.add_mirrorpoint(C, A, B))
    return info


# class UserConstructIntersectLineLine(FundamentalTheorem):

#   def __init__(self):
#     l1, l2 = map(Line, 'l1 l2'.split())
#     hp1, hp2 = map(HalfPlane, 'hp1 hp2'.split())

#     self.premise = [LineBordersHalfplane(l1, hp1), 
#                     LineBordersHalfplane(l2, hp2)]

#     B = Point('B')
#     self.conclusion = Conclusion(*(
#         collinear(l1, B) + collinear(l2, B)
#     ))
#     # AB = Segment('AB')
#     # self.conclusion.add(*segment_def(AB, A, B))

#     self.for_drawing = [B, l1, l2]
#     self.names = dict(l1=l1, l2=l2)
#     super(UserConstructIntersectLineLine, self).__init__()

#   def draw(self, mapping, canvas):
#     B, l1, l2 = map(mapping.get, self.for_drawing)
#     info = canvas.add_intersect_line_line(B, l1, l2)
#     return info


class ConstructIntersectSegmentLine(FundamentalTheorem):

  def __init__(self):
    A, B = map(Point, 'AB')
    l = Line('l')
    l_hp1, l_hp2 = HalfPlane('l_hp1'), HalfPlane('l_hp2')

    self.premise = (
        divides_halfplanes(l, l_hp1, l_hp2, A, B)  # l!=ab, A!=B
    )

    self.conclusion = Conclusion()

    ab = Line('ab')
    self.conclusion.add(*(
        collinear(ab, A, B)
    ))
    self.conclusion.add_critical(*distinct(ab, l))

    C = Point('C')    
    self.conclusion.add_critical(*(  # C related.
        collinear(ab, C) + 
        collinear(l, C) +
        distinct(C, A) +
        distinct(C, B)
    ))

    self.for_drawing = [C, l, A, B, ab]

    self.names = dict(l=l, A=A, B=B)
    super(ConstructIntersectSegmentLine, self).__init__()

  def draw(self, mapping, canvas):
    C, l, A, B, ab = map(mapping.get, self.for_drawing)
    info = canvas.add_line(ab, A, B)
    info.update(canvas.add_intersect_seg_line(C, l, A, B))
    return info

  @property
  def name(self):
    return 'Construct Line-Segment Intersection'


class ConstructPerpendicularLineFromPointOn(FundamentalTheorem):

  def __init__(self):
    l1 = Line('l1')
    P = Point('P')

    # If there is a line l1 and a line P
    self.premise = collinear(l1, P) # divides_halfplanes(l1, hp1, hp2)

    # Then:
    self.conclusion = Conclusion()

    d1, d2 = LineDirection('d1'), LineDirection('d2')

    # Suppose l1 has direction d1
    self.conclusion.add(LineHasDirection(l1, d1))

    # Suppose halfpi has measure 90
    mhalfpi = AngleMeasure('90')
    self.conclusion.add(AngleHasMeasure(geometry.get_halfpi(), mhalfpi))

    l2 = Line('l2')

    # Suppose d2 has angle measure 90 to d1
    angle12, angle21, angle12_def = fangle_def(d1, d2)
    self.conclusion.add(*(
        angle12_def + have_measure('m', angle12, angle21)
    ))  # d2 related

    # Then l2 goes through P, has direction d2 and distinct to l1
    self.conclusion.add_critical(*(  # l2 related.
        collinear(l2, P) + distinct(l1, l2) + [LineHasDirection(l2, d2)] +
        have_measure(mhalfpi, angle12)
    ))

    self.for_drawing = [l2, P, l1]
    self.names = dict(P=P, l=l1)

    super(ConstructPerpendicularLineFromPointOn, self).__init__()

  def draw(self, mapping, canvas):
    l2, P, l1 = map(mapping.get, self.for_drawing)
    info = canvas.add_perp_line_from_point_on(l2, P, l1)
    return info


class ConstructPerpendicularLineFromPointOut(FundamentalTheorem):

  def __init__(self):
    l1 = Line('l1')
    P = Point('P')

    # If there is a line l1 and a line P
    self.premise = collinear(l1, P) # divides_halfplanes(l1, hp1, hp2)

    # Then:
    self.conclusion = Conclusion()

    d1, d2 = LineDirection('d1'), LineDirection('d2')

    # Suppose l1 has direction d1
    self.conclusion.add(LineHasDirection(l1, d1))

    # Suppose halfpi has measure 90
    mhalfpi = AngleMeasure('90')
    self.conclusion.add(AngleHasMeasure(geometry.get_halfpi(), mhalfpi))

    l2 = Line('l2')

    # Suppose d2 has angle measure 90 to d1
    angle12, angle21, angle12_def = fangle_def(d1, d2)
    self.conclusion.add(*(
      angle12_def + have_measure('m', angle12, angle21)))  # d2 related

    # Then l2 goes through P, has direction d2 and distinct to l1
    self.conclusion.add_critical(*(  # l2 related.
        collinear(l2, P) + distinct(l1, l2) + [LineHasDirection(l2, d2)] +
        have_measure(mhalfpi, angle12, angle21)
    ))

    X = Point('X')
    self.conclusion.add(*(
        collinear(l1, X) + collinear(l2, X)
    ))

    self.for_drawing = [l2, P, l1, X]
    self.names = dict(P=P, l=l1)

    super(ConstructPerpendicularLineFromPointOut, self).__init__()

  def draw(self, mapping, canvas):
    l2, P, l1, X = map(mapping.get, self.for_drawing)
    return canvas.add_perp_line_from_point_out(l2, X, P, l1)


# class ConstructPerpendicularLineFromPointOut(FundamentalTheorem):

#   def __init__(self):
#     l = Line('l')
#     A = Point('A')
#     hp1, hp2 = map(HalfPlane, 'hp1 hp2'.split())

#     self.premise = divides_halfplanes(l, hp1, hp2, A)

#     l2 = Line('l2')
#     hp3, hp4 = map(HalfPlane, 'hp3 hp4'.split())
#     angle13, angle14, angle23, angle24 = map(
#         Angle, '^13 ^14 ^23 ^24'.split())
#     self.conclusion = Conclusion()
#     self.conclusion.add_critical(*(
#         collinear(l2, A) + distinct(l, l2) +
#         divides_halfplanes(l2, hp3, hp4) +
#         angle_def(angle13, hp1, hp3) +
#         angle_def(angle14, hp1, hp4) +
#         angle_def(angle23, hp2, hp3) +
#         angle_def(angle24, hp2, hp4) +
#         have_measure('halfpi', 
#                      geometry.halfpi,
#                      angle13, 
#                      angle14,
#                      angle23,
#                      angle24)
#     ))

#     B = Point('B')
#     AB = Segment('AB')
#     self.conclusion.add(*(collinear(l, B) + collinear(l2, B)))
#     self.conclusion.add(*segment_def(AB, A, B))

#     self.for_drawing = [l2, B, A, l]
#     self.names = dict(A=A, l=l)

#     super(ConstructPerpendicularLineFromPointOut, self).__init__()

#   def draw(self, mapping, canvas):
#     l2, B, A, l = map(mapping.get, self.for_drawing)
#     return canvas.add_perp_line_from_point_out(l2, B, A, l)


class MappingCheck(object):

  def __init__(self, fn, *objs):
    self.fn = fn
    self.objs = list(objs)
  
  def is_mapped(self, mapping):
    return all([obj in mapping for obj in self.objs])

  def __call__(self, mapping):
    return self.fn(*[mapping[obj] for obj in self.objs])


class ConstructAngleBisector(FundamentalTheorem):

  def __init__(self):
    l1, l2 = map(Line, 'l1 l2'.split())
    A = Point('A')
    l1_hp, l2_hp = map(HalfPlane, 'l1_hp l2_hp'.split())

    self.premise = [
        DistinctLine(l1, l2),
        LineContainsPoint(l1, A),  # So that l1 and l2 is not perpendicular.
        LineContainsPoint(l2, A),
        LineBordersHalfplane(l1, l1_hp),
        LineBordersHalfplane(l2, l2_hp),
    ]

    l3 = Line('l3')

    self.conclusion = Conclusion()

    d1, d2, d3 = LineDirection('d1'), LineDirection('d2'), LineDirection('d3')

    self.conclusion.add(LineHasDirection(l1, d1))
    self.conclusion.add(LineHasDirection(l2, d2))

    # Angle 1->3 == Angle 3->2
    m1 = AngleMeasure('0.5{}^{}_1'.format(l1.name, l2.name))
    m2 = AngleMeasure('0.5{}^{}_2'.format(l1.name, l2.name))
    angle13_xx, angle13_xo, fangle13_def = fangle_def(d1, d3)
    angle32_xx, angle32_xo, fangle32_def = fangle_def(d3, d2)
    # Unfortunately, there are two satisfying d3: inner bisector & outer.
    # So we have to numerically check using SelectAngle()

    self.conclusion.add(*(
        fangle13_def + fangle32_def +
        # if l1_hp.sign == l2_hp.sign, 
        # then angle13_xx == angle32_xo and angle13_xo == angle32_xx
        # else angle13_xx == angle32_xx and angle13_xo == angle32_xo
        have_measure(m1, angle13_xx, SelectAngle(angle32_xo, angle32_xx, l1_hp, l2_hp)) +
        have_measure(m2, angle13_xo, SelectAngle(angle32_xx, angle32_xo, l1_hp, l2_hp))
    ))

    self.conclusion.add_critical(*(
        collinear(l3, A) +
        distinct(l3, l1) + distinct(l3, l2) +
        have_direction(d3, l3)
    ))

    self.for_drawing = [l3, A, l1, l1_hp, l2, l2_hp]
    self.names = dict(hp1=l1_hp, hp2=l2_hp)

    super(ConstructAngleBisector, self).__init__()

  def draw(self, mapping, canvas):
    l3, A, l1, hp1, l2, hp2 = map(mapping.get, self.for_drawing)
    return canvas.add_angle_bisector(l3, A, l1, hp1, l2, hp2)


class ConstructMirrorAngle(FundamentalTheorem):

  def __init__(self):
    l1, l = map(Line, 'l1 l'.split())
    A = Point('A')
    hp1, l_hp1, l_hp2 = map(HalfPlane, 'hp1 l_hp1 l_hp2'.split())

    self.premise = [
        DistinctLine(l1, l),
        LineContainsPoint(l1, A),
        LineContainsPoint(l, A),
        LineBordersHalfplane(l1, hp1),
        LineBordersHalfplane(l, l_hp1),
        LineBordersHalfplane(l, l_hp2)
    ]

    l2 = Line('l2')
    self.conclusion = Conclusion()
    hp2 = HalfPlane('hp2')
    angle11, angle12, angle21, angle22 = map(Angle, '^11 ^12 ^21 ^22'.split())

    self.conclusion = Conclusion()
    self.conclusion.add(*angle_def(angle11, l_hp1, hp1))
    angle11_measure = AngleMeasure('1"')
    self.conclusion.add(AngleHasMeasure(angle11, angle11_measure))
    self.conclusion.add_critical(*(
        collinear(l2, A) +
        distinct(l2, l1) + distinct(l, l2) +
        LineBordersHalfplane(l2, hp2) +
        angle_def(angle12, l_hp1, hp2) +
        angle_def(angle21, l_hp2, hp1) +
        angle_def(angle22, l_hp2, hp2) +
        [AngleHasMeasure(angle22, angle11_measure)] +
        have_measure('2"', angle12, angle21)
    ))

    self.for_drawing = [l, l1, l2]
    self.names = dict(hp1=hp1, hp2=hp2)

    super(ConstructAngleBisector, self).__init__()

  def draw(self, mapping, canvas):
    l, l1, l2 = map(mapping.get, self.for_drawing)
    return canvas.add_angle_bisector(l, l1, l2)


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
    self.conclusion.add_critical(  # l2 related
        DistinctLine(l, l2),
        LineContainsPoint(l2, A), 
        LineHasDirection(l2, d))
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


class ConstructIntersectLineLine(FundamentalTheorem):

  def __init__(self):
    a, b = Line('a'), Line('b')

    self.premise = distinct(a, b)

    X = Point('X')
    self.conclusion = Conclusion()
    self.conclusion.add_critical(*(
        collinear(a, X) + collinear(b, X)
    ))

    self.for_drawing = [a, b, X]
    self.names = dict(a=a, b=b)

    super(ConstructIntersectLineLine, self).__init__()
  
  def draw(self, mapping, canvas):
    a, b, X = map(mapping.get, self.for_drawing)
    return canvas.add_intersect_line_line(X, a, b)


class ConstructLine(FundamentalTheorem):

  def __init__(self):
    A, B = Point('A'), Point('B')

    self.premise = distinct(A, B)

    ab = Line('ab')
    AB = Segment('AB')

    self.conclusion = Conclusion()
    self.conclusion.add_critical(*  # ab related.
        collinear(ab, A, B))
    self.conclusion.add(*segment_def(AB, A, B))

    self.for_drawing = [ab, A, B]
    self.names = dict(A=A, B=B)

    super(ConstructLine, self).__init__()

  def draw(self, mapping, canvas):
    ab, A, B = map(mapping.get, self.for_drawing)
    return canvas.add_line(ab, A, B)


class ConstructThirdLine(FundamentalTheorem):

  def __init__(self):
    A, B, C = map(Point, 'ABC')
    ac, bc = map(Line, 'ac bc'.split())

    self.premise = (
        collinear(ac, A, C) +
        collinear(bc, B, C) +
        # nothing implies these three
        # and without one of them the theorem failed.
        distinct(ac, bc) + 
        distinct(A, C) + 
        distinct(B, C)
    )

    ab = Line('ab')
    AB = Segment('AB')
    self.conclusion = Conclusion()
    self.conclusion.add_critical(*
        distinct(A, B))
    self.conclusion.add_critical(*  # ab related.
        collinear(ab, A, B) +
        distinct(ab, ac) +
        distinct(ab, bc))
    self.conclusion.add(*segment_def(AB, A, B))
    # self.conclusion.add(*have_length("1m", AB))

    self.for_drawing = [ab, A, B]
    self.names = dict(A=A, B=B)

    super(ConstructThirdLine, self).__init__()

  def draw(self, mapping, canvas):
    ab, A, B = map(mapping.get, self.for_drawing)
    return canvas.add_line(ab, A, B)

  @property
  def name(self):
    return 'Construct Line'


# class OppositeAngles(FundamentalTheorem):

#   def __init__(self):
#     a, b, c, d, m = map(Point, 'XYZTM')
#     ab, cd = map(Line, 'xy zt'.split())
#     ab_h1, ab_h2, cd_h1, cd_h2 = map(
#         HalfPlane, 'xy_h1 xy_h2 zt_h1 zt_h2'.split())

#     self.premise = (
#         collinear(ab, a, b, m) +  # A, M, B on e
#         collinear(cd, c, d, m) +  # C, M, D on f
#         divides_halfplanes(ab, ab_h1, ab_h2, c, d) +  # M between a and b
#         divides_halfplanes(cd, cd_h1, cd_h2, a, b)  # M between c and d
#     )

#     amc, bmd = Angle('XMZ'), Angle('YMT')
#     amd, bmc = Angle('XMT'), Angle('YMZ')
#     self.conclusion = Conclusion()
#     self.conclusion.add(*angle_def(amc, ab_h1, cd_h1))
#     self.conclusion.add(*angle_def(bmd, ab_h2, cd_h2))
#     self.conclusion.add(*angle_def(bmc, ab_h1, cd_h2))
#     self.conclusion.add(*angle_def(amd, ab_h2, cd_h1))
#     self.conclusion.add_critical(*have_measure('1"', amc, bmd))
#     self.conclusion.add_critical(*have_measure('2"', amd, bmc))

#     self.names = dict(A=a, M=m, C=c)

#     super(OppositeAngles, self).__init__()


class ParallelBecauseCorrespondingAngles(FundamentalTheorem):

  def __init__(self):
    l1, l1_hp1, _ = line_and_halfplanes('l1')
    l2, l2_hp1, l2_hp2 = line_and_halfplanes('l2')
    l3, l3_hp1, _ = line_and_halfplanes('l3')
    X, Y = Point('X'), Point('Y')

    angle12, angle13 = Angle('l12'), Angle('l13')
    self.premise = (
        collinear(l1, X, Y) +
        collinear(l2, X) +  # X != Y
        collinear(l3, Y) +
        divides_halfplanes(l1, l1_hp1) +
        divides_halfplanes(l3, l3_hp1, p1=X) +  # l3 != l1
        divides_halfplanes(l2, l2_hp1, l2_hp2, p2=Y) +  # l2 != l1
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


class ParallelBecauseInteriorAngles(FundamentalTheorem):

  def __init__(self):
    l1, l1_hp1, l1_hp2 = line_and_halfplanes('l1')
    l2, l2_hp, _ = line_and_halfplanes('l2')
    l3, l3_hp, _ = line_and_halfplanes('l3')
    X, Y = Point('X'), Point('Y')

    angle12, angle13 = Angle('l12'), Angle('l13')
    self.premise = (
        collinear(l1, X, Y) + 
        collinear(l2, X) +  # l2 != l3, X != Y
        collinear(l3, Y) +
        divides_halfplanes(l1, l1_hp1, l1_hp2) +
        divides_halfplanes(l2, l2_hp, p1=Y) +  # l2 != l1
        divides_halfplanes(l3, l3_hp, p1=X) +  # l3 != l1
        angle_def(angle12, l1_hp1, l2_hp) +
        angle_def(angle13, l1_hp2, l3_hp) +
        have_measure('\'1', angle12, angle13)
    )

    self.conclusion = Conclusion()
    d = LineDirection('d')
    self.conclusion.add(LineHasDirection(l2, d))
    self.conclusion.add_critical(LineHasDirection(l3, d))

    self.names = dict(l=l1, l1=l2, l2=l3)
    super(ParallelBecauseInteriorAngles, self).__init__()


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
        divides_halfplanes(l1, l1_hp, p1=B) +  # l1 != l, l1 != l2
        divides_halfplanes(l2, l2_hp, p1=A) +  # l2 != l, A != B
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

    self.conclusion.add(AngleHasMeasure(angle12, m2))
    self.conclusion.add_critical(AngleHasMeasure(angle21, m2))

    self.names = dict(l=l, l1=l1, l2=l2)
    super(EqualAnglesBecauseParallel, self).__init__()


class SamePairSkip(object):

  def __init__(self, a, b, x, y):
    self.pairs = a, b, x, y


def same_pair_skip(a, b, x, y, premise):
  skip = SamePairSkip(a, b, x, y)
  for p in premise:
    p.skip = skip
  return premise


class Congruences(FundamentalTheorem):
  pass


class SAS(Congruences):

  def __init__(self):
    A, B, C, D, E, F = map(Point, 'ABCDEF')
    ab, ab_hp, _ = line_and_halfplanes('ab')
    bc, bc_hp, _ = line_and_halfplanes('bc')
    de, de_hp, _ = line_and_halfplanes('de')
    ef, ef_hp, _ = line_and_halfplanes('ef')
    ABC, DEF = Angle('ABC'), Angle('DEF')
    AB, BC, DE, EF = map(Segment, ['AB', 'BC', 'DE', 'EF'])
               
    dAB, dBC = LineDirection('dAB'), LineDirection('dBC')
    dDE, dEF = LineDirection('dDE'), LineDirection('dEF')

    B_xx, B_xo, B_def = fangle_def(dAB, dBC)
    E_xx, E_xo, E_def = fangle_def(dDE, dEF)

    self.premise = (
        collinear(ab, A, B) +
        collinear(bc, B, C) +
        collinear(de, D, E) +
        collinear(ef, E, F) +

        # ab != bc
        divides_halfplanes(ab, ab_hp, p1=C) +  # C != A, C != B
        divides_halfplanes(bc, bc_hp, p1=A) +  # A != B

        # de != ef
        divides_halfplanes(de, de_hp, p1=F) +  # F != D, F != E
        divides_halfplanes(ef, ef_hp, p1=D) +  # D != E
        
        same_pair_skip(
            A, B, D, E,
            # If set(A, B) == set(D, E), skip:
            segment_def(AB, A, B) +
            segment_def(DE, D, E) +
            have_length('lAB', AB, DE)
        ) +
        
        same_pair_skip(
            B, C, E, F,
            # If set(B, C) == set(E, F), skip:
            segment_def(BC, B, C) +
            segment_def(EF, E, F) +
            have_length('lBC', BC, EF)
        ) +
        
        same_pair_skip(
            ab_hp, bc_hp, de_hp, ef_hp,
            # If set(ab_hp, bc_hp) == set(de_hp, ef_hp), skip:
            have_direction(dAB, ab) +
            have_direction(dBC, bc) +
            have_direction(dDE, de) +
            have_direction(dEF, ef) +
            B_def + E_def +
            have_measure('mB', SelectAngle(B_xx, B_xo, ab_hp, bc_hp), 
                               SelectAngle(E_xx, E_xo, de_hp, ef_hp))
        )
    )

    conclusion = Conclusion()

    ca, ca_hp, _ = line_and_halfplanes('ca')
    fd, fd_hp, _ = line_and_halfplanes('fd')
    conclusion.add(*collinear(ca, A, C))
    conclusion.add(*collinear(fd, D, F))
    conclusion.add(*divides_halfplanes(ca, ca_hp, p1=B))
    conclusion.add(*divides_halfplanes(fd, fd_hp, p1=E))

    AC, DF = Segment('AC'), Segment('DF')
    conclusion.add(*segment_def(AC, A, C))
    conclusion.add(*segment_def(DF, D, F))

    lAC = SegmentLength('lAC')
    conclusion.add(*have_length(lAC, AC))
    conclusion.add_critical(*have_length(lAC, DF))

    dCA, dFD = LineDirection('dCA'), LineDirection('dFD')
    conclusion.add(LineHasDirection(ca, dCA))
    conclusion.add(LineHasDirection(fd, dFD))

    A_xx, A_xo, A_def = fangle_def(dAB, dCA)
    C_xx, C_xo, C_def = fangle_def(dBC, dCA)
    D_xx, D_xo, D_def = fangle_def(dDE, dFD)
    F_xx, F_xo, F_def = fangle_def(dEF, dFD)

    conclusion.add(*A_def)
    conclusion.add(*C_def)
    conclusion.add(*D_def)
    conclusion.add(*F_def)

    mA, mA_supplement = AngleMeasure('mA'), AngleMeasure('mA_')
    mC, mC_supplement = AngleMeasure('mC'), AngleMeasure('mC_')

    angleA = SelectAngle(A_xx, A_xo, ab_hp, ca_hp)

    conclusion.add(*have_measure(mA, angleA))
    conclusion.add(*have_measure(mA_supplement, angleA.supplement()))

    angleD = SelectAngle(D_xx, D_xo, de_hp, fd_hp)

    conclusion.add_critical(*have_measure(mA, angleD))
    conclusion.add_critical(*have_measure(mA_supplement, angleD.supplement()))

    angleC = SelectAngle(C_xx, C_xo, bc_hp, ca_hp)

    conclusion.add(*have_measure(mC, angleC))
    conclusion.add(*have_measure(mC_supplement, angleC.supplement()))

    angleF = SelectAngle(F_xx, F_xo, ef_hp, fd_hp)
    conclusion.add_critical(*have_measure(mC, angleF))
    conclusion.add_critical(*have_measure(mC_supplement, angleF.supplement()))

    self.conclusion = conclusion
    self._distinct = [(AB, DE), 
                      (AB, BC), (BC, AC), (AC, AB),
                      (DE, EF), (EF, DF), (DF, DE),
                      (ab, bc), (bc, ca), (ca, ab),
                      (de, ef), (ef, fd), (fd, de),
                      # (BAC, BCA), (BCA, ABC), (ABC, BAC),
                      # (DEF, EDF), (EDF, EFD), (EFD, DEF),
                      (A, B), (B, C), (C, A),
                      (D, E), (E, F), (F, D),
                      ]

    self.for_drawing = [ca, A, C, fd, D, F]

    self.names = dict(A=A, B=B, C=C, D=D, E=E, F=F)

    super(SAS, self).__init__()

  def draw(self, mapping, canvas):
    ca, A, C, fd, D, F = map(mapping.get, self.for_drawing)
    info = {}
    if ca not in canvas.lines:
      info.update(canvas.add_line(ca, A, C))
    if fd not in canvas.lines:
      info.update(canvas.add_line(fd, D, F))
    return info

  @property
  def timeout(self):
    return 0.5

  @property
  def name(self):
    return 'Equal Triangles: Side-Angle-Side'


class SSS(Congruences):

  def __init__(self):
    A, B, C, D, E, F = map(Point, 'ABCDEF')
    AB, BC, CA, DE, EF, FD = map(
        Segment, ['AB', 'BC', 'CA', 'DE', 'EF', 'FD'])

    self.premise = (
        distinct(A, B, C) +
        distinct(D, E, F) +
        same_pair_skip(
            A, B, D, E,
            segment_def(AB, A, B) +
            segment_def(DE, D, E) +
            have_length('lAB', AB, DE)
        ) +
        same_pair_skip(
            B, C, E, F,
            segment_def(BC, B, C) +
            segment_def(EF, E, F) +
            have_length('lBC', BC, EF)
        ) +
        same_pair_skip(
            C, A, F, D,
            segment_def(CA, C, A) +
            segment_def(FD, F, D) +
            have_length('l3', CA, FD)
        )
    )

    conclusion = Conclusion()

    ab, ab_hp, _ = line_and_halfplanes('ab')
    bc, bc_hp, _ = line_and_halfplanes('bc')
    ca, ca_hp, _ = line_and_halfplanes('ca')

    de, de_hp, _ = line_and_halfplanes('de')
    ef, ef_hp, _ = line_and_halfplanes('ef')
    fd, fd_hp, _ = line_and_halfplanes('fd')

    conclusion.add(*collinear(ab, A, B))
    conclusion.add(*collinear(bc, B, C))
    conclusion.add(*collinear(ca, C, A))
    conclusion.add(*collinear(de, D, E))
    conclusion.add(*collinear(ef, E, F))
    conclusion.add(*collinear(fd, F, D))

    conclusion.add(*divides_halfplanes(ab, ab_hp, p1=C))
    conclusion.add(*divides_halfplanes(bc, bc_hp, p1=A))
    conclusion.add(*divides_halfplanes(ca, ca_hp, p1=B))

    conclusion.add(*divides_halfplanes(de, de_hp, p1=F))
    conclusion.add(*divides_halfplanes(ef, ef_hp, p1=D))
    conclusion.add(*divides_halfplanes(fd, fd_hp, p1=E))

    dAB, dBC = LineDirection('dAB'), LineDirection('dBC')
    dDE, dEF = LineDirection('dDE'), LineDirection('dEF')
    dCA, dFD = LineDirection('dCA'), LineDirection('dFD')

    conclusion.add(LineHasDirection(ab, dAB))
    conclusion.add(LineHasDirection(bc, dBC))
    conclusion.add(LineHasDirection(de, dDE))
    conclusion.add(LineHasDirection(ef, dEF))
    conclusion.add(LineHasDirection(ca, dCA))
    conclusion.add(LineHasDirection(fd, dFD))

    A_xx, A_xo, A_def = fangle_def(dAB, dCA)
    B_xx, B_xo, B_def = fangle_def(dAB, dBC)
    C_xx, C_xo, C_def = fangle_def(dBC, dCA)
    D_xx, D_xo, D_def = fangle_def(dDE, dFD)
    E_xx, E_xo, E_def = fangle_def(dDE, dEF)
    F_xx, F_xo, F_def = fangle_def(dEF, dFD)

    conclusion.add(*A_def)
    conclusion.add(*B_def)
    conclusion.add(*C_def)
    conclusion.add(*D_def)
    conclusion.add(*E_def)
    conclusion.add(*F_def)

    mA, mA_supplement = AngleMeasure('mA'), AngleMeasure('mA_')
    mB, mB_supplement = AngleMeasure('mB'), AngleMeasure('mB_')
    mC, mC_supplement = AngleMeasure('mC'), AngleMeasure('mC_')

    angleA = SelectAngle(A_xx, A_xo, ab_hp, ca_hp)
    conclusion.add(*have_measure(mA, angleA))
    conclusion.add(*have_measure(mA_supplement, angleA.supplement()))

    angleD = SelectAngle(D_xx, D_xo, de_hp, fd_hp)
    conclusion.add_critical(*have_measure(mA, angleD))
    conclusion.add_critical(*have_measure(mA_supplement, angleD.supplement()))

    angleB = SelectAngle(B_xx, B_xo, ab_hp, bc_hp)
    conclusion.add(*have_measure(mB, angleB))
    conclusion.add(*have_measure(mB_supplement, angleB.supplement()))

    angleE = SelectAngle(D_xx, D_xo, de_hp, ef_hp)
    conclusion.add_critical(*have_measure(mB, angleE))
    conclusion.add_critical(*have_measure(mB_supplement, angleE.supplement()))

    angleC = SelectAngle(C_xx, C_xo, bc_hp, ca_hp)
    conclusion.add(*have_measure(mC, angleC))
    conclusion.add(*have_measure(mC_supplement, angleC.supplement()))

    angleF = SelectAngle(F_xx, F_xo, ef_hp, fd_hp)
    conclusion.add_critical(*have_measure(mC, angleF))
    conclusion.add_critical(*have_measure(mC_supplement, angleF.supplement()))


    self.conclusion = conclusion
    self._distinct = [(A, D),
                      (AB, BC), (BC, CA), (CA, AB),
                      (DE, EF), (EF, FD), (FD, DE),
                      (ab, bc), (bc, ca), (ca, ab),
                      (de, ef), (ef, fd), (fd, de),
                      (A, B), (B, C), (C, A),
                      (D, E), (E, F), (F, D),
                      ]

    self.for_drawing = [ab, bc, ca, de, ef, fd,
                        A, B, C, D, E, F]
    self.names = dict(A=A, B=B, C=C, D=D, E=E, F=F)

    super(SSS, self).__init__()

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

        # ab != bc, ab != ca, ca != bc
        divides_halfplanes(ab, ab_hp1, p1=C) +
        divides_halfplanes(bc, bc_hp1, p1=A) +
        divides_halfplanes(ca, ca_hp1, p1=B) +

        # de != fd, de != ef, ef != fd
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

  @property
  def name(self):
    return 'Equal Triangles: Angle-Side-Angle'


all_theorems = [
    # 'unq_line_dir': SameLineBecauseParallel(),
    # SamePointBecauseSameMidpoint(),
    ConstructRightAngle(),
    ConstructMidPoint(),  # 0.000365972518921
    ConstructMirrorPoint(),
    ConstructAngleBisector(),
    ConstructIntersectSegmentLine(),
    # UserConstructIntersectLineLine(),
    ConstructParallelLine(),
    ConstructPerpendicularLineFromPointOn(),
    ConstructPerpendicularLineFromPointOut(),
    ConstructThirdLine(),
    # EqualAnglesBecauseParallel(),  # 1.73088312149
    SAS(),  # 0.251692056656
    # ASA(),  # 2.26002907753 3.96637487411
    SSS(),
    # ParallelBecauseCorrespondingAngles(),
    # ParallelBecauseInteriorAngles(),
    # OppositeAnglesCheck(),
    # ThalesCheck(),
    # For auto-mergings
    # SameAngleBecauseSameHalfPlane(),
    # SameHalfplaneBecauseSameLine(),
    # SameLineBecauseSameDirection(),
    # SameLineBecauseSamePoint(),
    # SamePointBecauseEqualSegments(),
    # SamePointBecauseSameLine(),
    # SamePointBecauseSameMidpoint(),
    # SameSegmentBecauseSamePoint()
]

theorem_from_name = {
  theorem.__class__.__name__: theorem
  for theorem in all_theorems
}

theorem_from_short_name = {
    # 'unq_line_dir': SameLineBecauseParallel(){}
    # 'unq_mid_point': SamePointBecauseSameMidpoint(),
    # 'unq_line_dir': SameLineBecauseSameDirection(),
    # 'right': ConstructRightAngle(),
    'mid': ConstructMidPoint(),  # 0.000365972518921
    # 'mirror': ConstructMirrorPoint(),
    'bisect': ConstructAngleBisector(),
    'seg_line': ConstructIntersectSegmentLine(),
    # 'line_line': UserConstructIntersectLineLine(),
    # 'parallel': ConstructParallelLine(),
    # 'perp_on': ConstructPerpendicularLineFromPointOn(),
    # 'perp_out': ConstructPerpendicularLineFromPointOut(),
    'line': ConstructThirdLine(),
    # 'eq': EqualAnglesBecauseParallel(),  # 1.73088312149
    'sas': SAS(),  # 0.251692056656
    # 'asa': ASA(),  # 2.26002907753 3.96637487411
    'sss': SSS(),
    # '.parallel': ParallelBecauseCorrespondingAngles(),
    # '.parallel2': ParallelBecauseInteriorAngles(),
    # 'angle_check': OppositeAnglesCheck(),
    # 'thales_check': ThalesCheck(),
    # For auto-mergings
    # 'auto_seg': SameSegmentBecauseSamePoint(),
    # 'auto_hp': SameHalfplaneBecauseSameLine(),
    # 'auto_angle': SameAngleBecauseSameHalfPlane()
}


"""
Theorem set 1. Merge (=>)

1. Point => Segment (Length)
2. Line (LineDirection) => HP => Angle (Measure)
3. Point <=> LineS
"""


def auto_merge_theorems_from_trigger_obj(obj):
  if isinstance(obj, Point):
    return [
        theorem_from_name[name]
        for name in ['SameSegmentBecauseSamePoint',
                     'SameLineBecauseSamePoint']
    ]
  if isinstance(obj, Line):
    return [
        theorem_from_name[name]
        for name in ['SameHalfplaneBecauseSameLine',
                     'SamePointBecauseSameLine']
    ]
  if isinstance(obj, HalfPlane):
    return [
        theorem_from_name[name]
        for name in ['SameAngleBecauseSameHalfPlane']
    ]
  return []
  
  
