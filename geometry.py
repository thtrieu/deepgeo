"""Implement the environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict as ddict
import traceback
import os


def print_tb():
  for line in traceback.format_stack()[:-2]:
    l1, l2 = line.split('\n')[:2]
    if 'anaconda2/lib' in l1:
      continue
    path = os.path.basename(l1.split('"')[1])
    line_num = l1.split(',')[1].strip().split()[-1]
    print('{}:{}\t\t{}'.format(path, line_num, l2.strip()))


name_to_obj = {}


name_scope = ''


def start_name_scope(name):
  global name_scope
  name_scope = name


def reset_name_scope():
  global name_scope
  name_scope = ''


def get_obj(name):
  global name_to_obj
  return name_to_obj.get(name, None)


def _bfs(edges, from_node, to_nodes):
  visited = {obj: False for obj in edges}
  parent = {obj: None for obj in edges}

  queue = [from_node] 
  visited[from_node] = True

  found = None
  while queue:
    s = queue.pop(0)
    for i in edges[s]:
      if i not in visited:
        continue
      if not visited[i]: 
        parent[i] = s
        if i in to_nodes:
          found = i
          break
        queue.append(i) 
        visited[i] = True
    if found:
      break

  if not found:
    return []

  path = [found]
  while path[-1] != from_node:
    p = parent[path[-1]]
    path.append(p)

  return path


class GeometryEntity(object):

  def __init__(self, name=None):
    self.name = name or self.get_name()
    # global name_scope
    # self.name = name_scope + '/' + self.name

    global name_to_obj

    count = 0
    name = self.name
    while name in name_to_obj:
      name = self.name + '_' + str(count)
      count += 1

    self.name = name
    name_to_obj[self.name] = self

  def get_merge_graph(self, state, default={}):
    if not hasattr(self, 'merge_graph'):
      self.merge_graph = {}
    if state not in self.merge_graph:
      if 'equivalents' not in default:
        default['equivalents'] = []
      self.merge_graph[state] = default
    return self.merge_graph[state]

  def get_name(self):
    global _name_bank
    if isinstance(self, Point):
      _name_bank[Point] += 1
      return 'P' + str(_name_bank[Point])

    elif isinstance(self, Line):
      _name_bank[Line] += 1
      return 'l' + str(_name_bank[Line])

    # elif isinstance(self, Angle):
    #   _name_bank[Angle] += 1
    #   return '^' + str(_name_bank[Angle])

    elif isinstance(self, AngleXX):
      _name_bank[AngleXX] += 1
      return '^xx' + str(_name_bank[AngleXX])

    elif isinstance(self, AngleXO):
      _name_bank[AngleXO] += 1
      return '^xo' + str(_name_bank[AngleXO])

    elif isinstance(self, FullAngle):
      _name_bank[FullAngle] += 1
      return '^' + str(_name_bank[FullAngle])

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
  # the dependencies on the action chain and to whittle proofs.

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

  def copy(self, old_state, new_state):
    if hasattr(self, 'merge_graph') and old_state in self.merge_graph:
      self.merge_graph[new_state] = {
        obj1: type(obj2_dict)(obj2_dict)
        for obj1, obj2_dict in self.merge_graph[old_state].items()
      }

  def pop(self, state):
    if hasattr(self, 'merge_graph') and state in self.merge_graph:
      self.merge_graph.pop(state)

  def find_merge_path(self, rel, obj, state):
    if obj == rel.init_list[0]:
      other = rel.init_list[1]
    else:
      other = rel.init_list[0]
    
    edges = self.merge_graph[state]
    equivalents = edges[other].keys()

    path = _bfs(edges, obj, equivalents)
    found = path[0]
    result = [
        edges[p2][p1] for p1, p2 in zip(path[:-1], path[1:])]
    rel = edges[other][found]
    return [rel] + result

  def find_min_span_subgraph(self, equivs, state):
    merge_graph = self.get_merge_graph(state)
    
    result = set()
    span = [self]
    for equiv in equivs:
      if equiv not in span:
        path = _bfs(merge_graph, equiv, span)
        span += path
        result.update([
            merge_graph[p2][p1] for p1, p2 in zip(path[:-1], path[1:])])
    return list(result)

def update_edges(edges1, edges2):
  for obj_a, obj_b_dict in edges2.items():
    if obj_a in edges1:
      for obj_b, pos in obj_b_dict.items():
        if obj_b not in edges1[obj_a]:
          edges1[obj_a][obj_b] = pos
        else:
          edges1[obj_a][obj_b] = min(pos, edges1[obj_a][obj_b])
    else: 
      edges1[obj_a] = dict(obj_b_dict)


class CausalValue(GeometryEntity):
  """Handles transitivity causal dependency."""

  def __init__(self, name=None):
    self.edges = ddict(lambda: ddict(lambda: {}))
    # if obj:
    #   self.edges[obj] = {}
    # This is to add a new clique when there is new equality
    self.edges_tmp = ddict(lambda: {})
    super(CausalValue, self).__init__(name)

  def copy(self, old_state, new_state):
    self.edges[new_state] = {
      obj1: dict(obj2_dict)
      for obj1, obj2_dict in self.edges[old_state].items()
    }

  def _has_edge_from_to(self, obj1, obj2, state):
    return (obj1 in self.edges[state] and obj2 in self.edges[state][obj1])
  
  def has_edges(self, obj1, obj2, state):
    return self._has_edge_from_to(obj1, obj2, state) or self._has_edge_from_to(obj2, obj1, state)

  def update_edges_tmp(self, edges):
    update_edges(self.edges_tmp, edges)

  def add_new_clique(self, objs):
    if len(objs) < 2:
      return

    # Clear edges_tmp
    self.edges_tmp = ddict(lambda: {})

    # Loop through all pairs of objects
    for i, obj1 in enumerate(objs[:-1]):
      for _, obj2 in enumerate(objs[i+1:]):
        if obj2 not in self.edges_tmp[obj1]:
          self.edges_tmp[obj1][obj2] = None
        if obj1 not in self.edges_tmp[obj2]:
          self.edges_tmp[obj2][obj1] = None

  def merge(self, val, state):
    for node, neighbors in val.edges[state].items():
      if node in self.edges[state]:
        self.edges[state][node].update(neighbors)
      else:
        self.edges[state][node] = dict(neighbors)

  def dependency_path(self, obj1, obj2, state):
    # perform a BFS
    edges = self.edges[state]
    path = _bfs(edges, obj1, [obj2])
    result = [
        edges[p2][p1] for p1, p2 in zip(path[:-1], path[1:])]
    return result

  def set_chain_position(self, pos):
    if not hasattr(self, '_chain_position'):
      self._chain_position = pos

    # Set the clique
    for _, neighbors in self.edges_tmp.items():
      for p2 in neighbors:
        if neighbors[p2] is None:
          neighbors[p2] = pos

  def merge_tmp_clique(self, state):
    for p1, neighbors in self.edges_tmp.items():
      if p1 in self.edges[state]:
        self.edges[state][p1].update(neighbors)
      else:
        self.edges[state][p1] = dict(neighbors)


class LineDirection(CausalValue):
  pass


class SegmentLength(CausalValue):
  pass


class AngleMeasure(CausalValue):
  pass


class Point(GeometryEntity):
  pass


class Segment(GeometryEntity):
  pass


class HalfAngle(GeometryEntity):
  pass


class Angle(GeometryEntity):
  pass


class AngleXX(Angle):
  pass


class AngleXO(Angle):
  pass


class SelectAngle(Angle):

  def __init__(self, angle_xx, angle_xo, hp1, hp2):
    assert isinstance(angle_xx, Angle) and isinstance(angle_xo, Angle)
    assert isinstance(hp1, HalfPlane) and isinstance(hp2, HalfPlane)

    self.choices = angle_xx, angle_xo
    self.hps = hp1, hp2
    self.name = '{}=={}?{}:{}'.format(hp1.name, hp2.name, angle_xx.name, angle_xo.name)

  def is_available(self, mapping):
    return all([hp in mapping for hp in self.hps])

  def select(self, mapping):
    angle_xx, angle_xo = self.choices
    hp1, hp2 = map(mapping.get, self.hps)
    if hp1.sign == hp2.sign:
      return angle_xx
    else:
      return angle_xo

  def supplement(self):
    angle_xx, angle_xo = self.choices
    hp1, hp2 = self.hps
    return SelectAngle(angle_xo, angle_xx, hp1, hp2)



class FullAngle(GeometryEntity):
  pass



halfpi = None



def get_halfpi():
  global halfpi
  if halfpi is None:
    halfpi = Angle('halfpi')
  return halfpi


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
    FullAngle: 0,
    AngleXX: 0,
    AngleXO: 0,
    HalfPlane: 0,
    Circle: 0,
    LineDirection: 0,
    AngleMeasure: 0,
    SegmentLength: 0,
}


def reset():
  global name_to_obj
  name_to_obj = {}
  global _name_bank
  _name_bank = {k: 0 for k in _name_bank}


def reset_auto_name_bank():
  global _name_bank
  _name_bank = {k: 0 for k in _name_bank}


# TODO(thtrieu): handle the case where relations does not cover all objs.
# or prove that it is not the case for any algorithm
# in fact it might as well be.
# no: given a segment and a point, exists a line through the point perp to segment.



class Relation(GeometryEntity):

  def __init__(self, obj1, obj2):
    raise NotImplementedError('Abstract class Relation.')

  @property
  def init_list(self):
    return self._init_list

  def replace(self, a, b):
    init_list = (x, y) = self._init_list
    if x == a:
      init_list = (b, y)
    elif y == a:
      init_list = (x, b)
    else:
      return self

    return type(self)(*init_list)
  
  def copy(self, old_state, new_state):
    a, b = self.init_list
    a.copy(old_state, new_state)
    b.copy(old_state, new_state)

  def pop(self, state):
    a, b = self.init_list
    a.pop(state)
    b.pop(state)
    


class Merge(Relation):

  def __init__(self, from_obj, to_obj, merge_graph={}):
    assert from_obj != to_obj
    assert isinstance(from_obj, type(to_obj))
    self._init_list = from_obj, to_obj
    self.from_obj = from_obj
    self.to_obj = to_obj

    # This is used by state.add_relation
    # to set to_obj.merge_graph[state]
    self.merge_graph = merge_graph

    self.name = 'merge({}=>{})'.format(from_obj.name, to_obj.name)


class Distinct(Relation):
  pass


class DistinctLine(Distinct):

  def __init__(self, obj1, obj2):
    assert isinstance(obj1, Line) and isinstance(obj2, Line)
    self.name = '{}!={}'.format(obj1.name, obj2.name)
    self._init_list = obj1, obj2


class DistinctPoint(Distinct):

  def __init__(self, obj1, obj2):
    assert isinstance(obj1, Point) and isinstance(obj2, Point)
    self.name = '{}!={}'.format(obj1.name, obj2.name)
    self._init_list = obj1, obj2


class PointEndsSegment(Relation):

  def __init__(self, point, segment):
    assert isinstance(point, Point) and isinstance(segment, Segment)
    self.name = '{}[{}'.format(point.name, segment.name)
    self._init_list = point, segment


class AngleOfFullAngle(Relation):

  def __init__(self, angle, fangle):
    assert isinstance(angle, Angle)
    assert isinstance(fangle, FullAngle)

    self.name = '{}-{}'.format(angle.name, fangle.name)
    self._init_list = angle, fangle


class DirectionOfFullAngle(Relation):

  def __init__(self, direction, fangle):
    assert (isinstance(direction, LineDirection) and 
            isinstance(fangle, FullAngle))

    self.name = '{}//{}'.format(direction.name, fangle.name)
    self._init_list = direction, fangle


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