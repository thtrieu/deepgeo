"""Implement the environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict as ddict


name_to_obj = {}


def get_obj(name):
  global name_to_obj
  return name_to_obj.get(name, None)


class GeometryEntity(object):

  def __init__(self, name=None):
    self.name = name or self.get_name()
    global name_to_obj

    count = 0
    name = self.name
    while name in name_to_obj:
      name = self.name + '_' + str(count)
      count += 1

    self.name = name
    name_to_obj[self.name] = self

  def get_name(self):
    global _name_bank
    if isinstance(self, Point):
      _name_bank[Point] += 1
      return 'P' + str(_name_bank[Point])

    elif isinstance(self, Line):
      _name_bank[Line] += 1
      return 'l' + str(_name_bank[Line])

    elif isinstance(self, Angle):
      _name_bank[Angle] += 1
      return '^' + str(_name_bank[Angle])

    elif isinstance(self, Segment):
      _name_bank[Segment] += 1
      return 's' + str(_name_bank[Segment])

    elif isinstance(self, Circle):
      _name_bank[Circle] += 1
      return '(' + str(_name_bank[Circle]) + ')'

    elif isinstance(self, HalfPlane):
      _name_bank[HalfPlane] += 1
      return '.' + str(_name_bank[HalfPlane])

    elif isinstance(self, SegmentLength):
      _name_bank[SegmentLength] += 1
      return str(_name_bank[SegmentLength]) + 'm'

    elif isinstance(self, AngleMeasure):
      _name_bank[AngleMeasure] += 1
      return str(_name_bank[AngleMeasure]) + '"'

    elif isinstance(self, LineDirection):
      _name_bank[LineDirection] += 1
      return 'd' + str(_name_bank[LineDirection])

  # Any object in our exploration is associated with
  # the following 3 concepts:
  #  * conclusion position: Since every object is create by
  # applying a theorem T, it belong to one position on 
  # T.conclusion.topological_list
  #  * critical: A position on T.conclusion.topological_list
  # is either critical or not (i.e. its creation requires
  # the full T.premise or not). 
  # E.g. in EqualAnglesBecauseIntersectCords, where 2 cords
  # are XY and ZT, then line XT belong on the conclusion
  # but not critical
  #  * chain_position: each application of a theorem is
  # associated with a position on the action_chain so far

  # We record all these information into the geometric entity
  # Because they are necessary information to figure out
  # the dependency on the action chain and whittle proofs.

  @property
  def conclusion_position(self):
    """In what position in the conclusion this obj is on?"""
    if not hasattr(self, '_conclusion_position'):
      return None
    return self._conclusion_position

  def set_conclusion_position(self, pos):
    """Set this during exploration."""
    if hasattr(self, '_conclusion_position'):
      raise ValueError(
          'Cannot set conclusion position for {} {} twice.'.format(
                type(self).__name__,  self.name))
    self._conclusion_position = pos

  @property
  def chain_position(self):
    """In what position in the action chain is this object?"""
    if not hasattr(self, '_chain_position'):
      return None
    return self._chain_position

  def set_chain_position(self, pos):
    """Set this during exploration."""
    if hasattr(self, '_chain_position'):
      raise ValueError(
          'Cannot set chain position for {} {} twice.'.format(
                type(self).__name__,  self.name))
    self._chain_position = pos

  @property
  def critical(self):
    """Does this object requires the full premise of its creation?"""
    if not hasattr(self, '_critical'):
      return None
    return self._critical

  def set_critical(self, critical):
    """Set this during exploration."""
    if hasattr(self, '_critical'):
      raise ValueError(
          'Cannot set critical for {} {} twice.'.format(
                type(self).__name__,  self.name))
    self._critical = critical


class CausalValue(GeometryEntity):
  """Handles transitivity causal dependency."""

  def __init__(self, name=None, obj=None):
    self.edges = ddict(lambda: {})
    if obj:
      self.edges[obj] = {}
    super(CausalValue, self).__init__(name)


  def copy(self):
    self2 = type(self)(None)
    self2.edges = {obj: list(neighbors)
                   for obj, neighbors in self.edges.items()}
    return self2

  def add(self, objs):
    # for source in self.edges:
    #   if source != obj:
    #     self.edges[source][obj] = None
    # if obj not in self.edges:
    #   self.edges[obj] = {x: None for x in self.edges.keys()}
    if len(objs) < 2:
      return

    # Loop through all pairs of objects
    for i, obj1 in enumerate(objs[:-1]):
      for j, obj2 in enumerate(objs[i+1:]):
        if obj2 not in self.edges[obj1]:
          self.edges[obj1][obj2] = None
        if obj1 not in self.edges[obj2]:
          self.edges[obj2][obj1] = None

  def merge(self, val):
    # print('**', {x.name: {a.name: b for a, b in y.items()} for x, y in val.edges.items()})
    # print('**', {x.name: {a.name: b for a, b in y.items()} for x, y in self.edges.items()})
    for node, neighbors in val.edges.items():
      self.edges[node].update(neighbors)

  def dependency_path(self, obj1, obj2):
    visited = {obj: False for obj in self.edges}
    parent = {obj: None for obj in self.edges}

    queue = [obj1] 
    visited[obj1] = True

    found = False
    while queue:
      s = queue.pop(0) 
      for i in self.edges[s]: 
        if i not in visited:
          continue
        if not visited[i]: 
          parent[i] = s
          if i == obj2:
            found = True
            break
          queue.append(i) 
          visited[i] = True
      if found:
        break

    path = [obj2]
    while path[-1] != obj1:
      p = parent[path[-1]]
      path.append(p)

    # return path
    return [self.edges[p1][p2] for p1, p2 in zip(path[:-1], path[1:])]

  def set_chain_position(self, pos):
    if not hasattr(self, '_chain_position'):
      self._chain_position = pos
    for p1, neighbors in self.edges.items():
      for p2, old_pos in neighbors.items():
        neighbors[p2] = pos if old_pos is None else old_pos


class LineDirection(CausalValue):
  pass


class SegmentLength(CausalValue):
  pass


class AngleMeasure(CausalValue):
  pass


halfpi = AngleMeasure('pi/2')


class Point(GeometryEntity):
  pass


class Segment(GeometryEntity):
  pass


class Angle(GeometryEntity):
  pass


class HalfPlane(GeometryEntity):
  pass


class Line(GeometryEntity):
  pass


class Circle(GeometryEntity):
  pass


alphabet = 'abcdefghijklmnopqrstuvwxyz'

_name_bank = {
    Point: 0,
    Line: 0,
    Segment: 0,
    Angle: 0,
    HalfPlane: 0,
    Circle: 0,
    LineDirection: 0,
    AngleMeasure: 0,
    SegmentLength: 0
}

def reset():
  global name_to_obj
  name_to_obj = {}
  global _name_bank
  _name_bank = {k: 0 for k in _name_bank}



# TODO(thtrieu): handle the case where relations does not cover all objs.
# or prove that it is not the case for any algorithm
# in fact it might as well be.
# no: given a segment and a point, exists a line through the point perp to segment.



class Relation(GeometryEntity):

  @property
  def init_list(self):
    return self._init_list


class PointEndsSegment(Relation):

  def __init__(self, point, segment):
    assert isinstance(point, Point) and isinstance(segment, Segment)
    self.name = '{}[{}'.format(point.name, segment.name)
    self._init_list = point, segment


class HalfplaneCoversAngle(Relation):

  def __init__(self, halfplane, angle):
    assert (isinstance(halfplane, HalfPlane) and 
            isinstance(angle, Angle))

    self.name = '{}/{}'.format(halfplane.name, angle.name)
    self._init_list = halfplane, angle


class TransitiveRelation(Relation):

  def new_rel(self, new_val):
    return type(self)(self.init_list[0], new_val)


class SegmentHasLength(TransitiveRelation):

  def __init__(self, segment, length):
    assert (isinstance(segment, Segment) and
            isinstance(length, SegmentLength))

    self.name = '{}={}'.format(segment.name, length.name)
    self._init_list = segment, length


class AngleHasMeasure(TransitiveRelation):

  def __init__(self, angle, measure):
    assert (isinstance(angle, Angle) and
            isinstance(measure, AngleMeasure))
    self.name = '{}={}'.format(angle.name, measure.name)
    self._init_list = angle, measure


class LineHasDirection(TransitiveRelation):

  def __init__(self, line, direction):
    assert isinstance(line, Line) and isinstance(direction, LineDirection)
    self.name = '{}|{}'.format(line.name, direction.name)
    self._init_list = line, direction


class LineDirectionPerpendicular(Relation):

  def __init__(self, dir1, dir2):

    assert isinstance(dir1, LineDirection) and isinstance(dir2, LineDirection)
    self.name = '{}T{}'.format(dir1.name, dir2.name)
    self._init_list = dir1, dir2


class LineContainsPoint(Relation):

  def __init__(self, line, point):
    assert isinstance(line, Line)
    assert isinstance(point, Point)

    self.name = '{}[{}]'.format(line.name, point.name)
    self._init_list = line, point


class CircleContainsPoint(Relation):

  def __init__(self, circle, point):
    assert isinstance(circle, Circle) and isinstance(point, Point)
    self.name = '{}({})'.format(circle.name, point.name)
    self._init_list = circle, point


class HalfPlaneContainsPoint(Relation):

  def __init__(self, halfplane, point):
    assert (isinstance(halfplane, HalfPlane) and
            isinstance(point, Point))

    self.name = halfplane.name + '{' + point.name + '}'
    self._init_list = halfplane, point


class PointCentersCircle(Relation):

  def __init__(self, point, circle):
    assert isinstance(point, Point) and isinstance(circle, Circle)
    self.name = '{}@{}'.format(point.name, circle.name)
    self._init_list = point, circle


class LineBordersHalfplane(Relation):

  def __init__(self, line, halfplane):

    assert (isinstance(line, Line) and
            isinstance(halfplane, HalfPlane))
    self.name = '{}/{}'.format(line.name, halfplane.name)
    self._init_list = line, halfplane