
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import geometry

from geometry import Distinct, RatioHasValue, RatioValue
from geometry import Point, Line, Segment, Angle, HalfPlane, Circle, Ratio
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, LinedirectionOfAngle, SegmentlengthInRatio, LineBordersHalfplane
from geometry import PointCentersCircle, Merge
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint
from geometry import OppositeAngles, OppositeRatios, OppositeSegment


def line_and_halfplanes(name):
  return Line(name), HalfPlane(name + '_hp1'), HalfPlane(name + '_hp2')


def divides_halfplanes(line, hp1, hp2=None, p1=None, p2=None):
  result = [LineBordersHalfplane(line, hp1)]
  if hp2:
    result.append(LineBordersHalfplane(line, hp2))
  if p1:
    result.append(HalfPlaneContainsPoint(hp1, p1))
  if p2:
    result.append(HalfPlaneContainsPoint(hp2, p2))
  return result


def distinct(*obj_list):
  result = []
  for i, obj1 in enumerate(obj_list[:-1]):
    for obj2 in obj_list[i+1:]:
      result.append(Distinct(obj1, obj2))
  return result


def in_halfplane(hp, *point_list):
  return [HalfPlaneContainsPoint(hp, p) for p in list(point_list)]


def collinear(l, *point_list):
  return [LineContainsPoint(l, p) for p in list(point_list)]


def concyclic(o, *point_list):
  return [CircleContainsPoint(o, p) for p in list(point_list)]


def have_value(name, *ratio_list):
  if isinstance(name, str):
    value = RatioValue(name)
  elif isinstance(name, RatioValue):
    value = name
  else:
    raise ValueError('{} is not str nor RatioValue'.format(name))
  return [RatioHasValue(ratio, value)
          for ratio in list(ratio_list)]


def have_length(name, *segment_list):
  if isinstance(name, str):
    length = SegmentLength(name)
  elif isinstance(name, SegmentLength):
    length = name
  else:
    raise ValueError('{} is not str nor SegmentLength'.format(name))
  return [SegmentHasLength(seg, length)
          for seg in list(segment_list)]


def have_measure(name, *angle_list):
  if isinstance(name, str):
    angle_measure = AngleMeasure(name)
  elif isinstance(name, AngleMeasure):
    angle_measure = name
  else:
    raise ValueError('{} is not str nor AngleMeasure'.format(name))
  return [AngleHasMeasure(angle, angle_measure)
          for angle in list(angle_list)]


def have_direction(name, *line_list):
  if isinstance(name, str):
    direction = LineDirection(name)
  elif isinstance(name, LineDirection):
    direction = name
  else:
    raise ValueError('{} is not str nor LineDirection'.format(name))
  return [LineHasDirection(line, direction)
          for line in list(line_list)]


def two_segments(p1, p2):
  return [Segment(p1.name + p2.name), Segment(p2.name + p1.name)]


def two_ratios(l1, l2):
  return [Ratio(l1.name+':'+l2.name), Ratio(l2.name+':'+l1.name)]


def two_angles(d1, d2):
  return [Angle(d1.name+'-'+d2.name), Angle(d2.name+'-'+d1.name)]


# def vector_def(p1, p2, v1=None, v2=None):
#   seg12, seg21 = two_segments(p1, p2)
#   result = [PointEndsSegment(p1, seg12), 
#             PointEndsSegment(p2, seg21),
#             OppositeSegment(seg12, seg21)]
#   if length12:
#     result += have_length(length12, seg12)
#   if length21:
#     result += have_length(length21, seg21)
#   return seg12, seg21, result


def segment_def(p1, p2, length=None):
  seg = Segment(p1.name + p2.name)
  result = [
      PointEndsSegment(p1, seg),
      PointEndsSegment(p2, seg)
  ]
  if length:
    result += [SegmentHasLength(seg, length)]
  return seg, result



def angle_def(direction1, direction2, measure12=None, measure21=None):
  angle12, angle21 = two_angles(direction1, direction2)
  result = [
      LinedirectionOfAngle(direction1, angle12),
      LinedirectionOfAngle(direction2, angle21),
      OppositeAngles(angle12, angle21)
  ]
  if measure12:
    result += have_measure(measure12, angle12)
  if measure21:
    result += have_measure(measure21, angle21)
  return angle12, angle21, result


def ratio_def(length1, length2, value12=None, value21=None):
  ratio12, ratio21 = two_ratios(length1, length2)
  result = [
      SegmentlengthInRatio(length1, ratio12),
      SegmentlengthInRatio(length2, ratio21),
      OppositeRatios(ratio12, ratio21)
  ]
  if value12:
    result += have_value(value12, ratio12)
  if value21:
    result += have_value(value21, ratio21)
  return ratio12, ratio21, result


def diff_side(line, point1, point2, h1=None, h2=None):
  h1 = h1 or HalfPlane(line.name + '_some_hp1')
  h2 = h2 or HalfPlane(line.name + '_some_hp2')
  return [
      LineBordersHalfplane(line, h1),
      LineBordersHalfplane(line, h2),
      HalfPlaneContainsPoint(h1, point1),
      HalfPlaneContainsPoint(h2, point2)
  ]


def same_side(l, p1, p2, hp=None):
  hp = hp or HalfPlane(l.name + '_hp')
  return [
      LineBordersHalfplane(l, hp),
      HalfPlaneContainsPoint(hp, p1),
      HalfPlaneContainsPoint(hp, p2)
  ]