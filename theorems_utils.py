
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import geometry

from geometry import Distinct
from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import PointCentersCircle, Merge
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


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


def segment_def(seg, p1, p2):
  return [PointEndsSegment(p1, seg), PointEndsSegment(p2, seg)]


def angle_def(angle, hp1, hp2):
  return [HalfplaneCoversAngle(hp1, angle), HalfplaneCoversAngle(hp2, angle)]


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
  hp = hp or HalfPlane(l.name + '_some_hp1')
  return [
      LineBordersHalfplane(l, hp),
      HalfPlaneContainsPoint(hp, p1),
      HalfPlaneContainsPoint(hp, p2)
  ]