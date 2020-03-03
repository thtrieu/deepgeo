
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import geometry

from collections import defaultdict as ddict

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import PointCentersCircle, Merge
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


non_relations = [
    Point, Line, Segment, Angle, HalfPlane, Circle,
    SegmentLength, AngleMeasure, LineDirection
]


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


def in_halfplane(hp, *point_list):
  return [HalfPlaneContainsPoint(hp, p) for p in list(point_list)]


def collinear(l, *point_list):
  return [LineContainsPoint(l, p) for p in list(point_list)]


def concyclic(o, *point_list):
  return [CircleContainsPoint(o, p) for p in list(point_list)]


def have_length(name, *segment_list):
  length = SegmentLength(name)
  return [SegmentHasLength(seg, length)
          for seg in list(segment_list)]


def have_measure(name, *angle_list):
  angle_measure = AngleMeasure(name)
  return [AngleHasMeasure(angle, angle_measure)
          for angle in list(angle_list)]


def have_direction(name, *line_list):
  direction = LineDirection(name)
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
  hp = hp or HalfPlane(line.name + '_some_hp1')
  return [
      LineBordersHalfplane(l, hp),
      HalfPlaneContainsPoint(hp, p1),
      HalfPlaneContainsPoint(hp, p2)
  ]


def _copy(structure):
  if not isinstance(structure, (list, tuple, dict)):
    return structure
  elif isinstance(structure, list):
    return [_copy(x) for x in structure]
  elif isinstance(structure, tuple):
    return tuple(_copy(x) for x in structure)
  else:
    return {_copy(key): _copy(val) 
            for (key, val) in structure.items()}


class State(object):

  def __init__(self):
    self.relations = []
    # For transitive handling:
    self.type2rel = {}
    self.obj2valrel = {}
    self.val2valrel = {}
    self.valrel2pos = {}
    self.name2obj = {}
    self.all_points = []
    self.all_hps = []

    self.line2hps = {}
    self.hp2points = {}

  def copy(self):
    copied = State()
    copied.relations = _copy(self.relations)
    copied.type2rel = _copy(self.type2rel)
    copied.obj2valrel = _copy(self.obj2valrel)
    copied.val2valrel = _copy(self.val2valrel)
    copied.valrel2pos = _copy(self.valrel2pos)
    copied.name2obj = _copy(self.name2obj)

    copied.all_points = _copy(self.all_points)
    copied.all_hps = _copy(self.all_hps)

    # For identifying halfplanes
    copied.line2hps = _copy(self.line2hps)
    copied.hp2points = _copy(self.hp2points)
    return copied

  def ends_of_segment(self, segment):
    points = []
    for p_seg in self.type2rel[PointEndsSegment]:
      if segment == p_seg.init_list[1]:
        points.append(p_seg.init_list[0])
    return points

  def hp_and_line_of_angle(self, angle):
    hps = []
    lines = []
    for hp_a in self.type2rel[HalfplaneCoversAngle]:
      if angle == hp_a.init_list[1]:
        hp = hp_a.init_list[0]
        l = self.line_of_hp(hp)
        lines.append(l)
        hps.append(self.line2hps[l].index(hp))  # 0: neg, 1: pos
    return hps, lines

  def line_of_hp(self, hp):
    for l, l_hps in self.line2hps.items():
      if hp in l_hps:
        return l

  def to_str(self):
    result = []
    for r in self.relations:
      a, b = r.init_list
      result += ['({}, {}, \'{}\', {}, \'{}\')'.format(
          type(r).__name__,
          type(a).__name__,
          a.name,
          type(b).__name__,
          b.name)]
    result = ', '.join(result)
    return '[{}]'.format(result)

  def new_relations_from_merge(self, obj1, obj2):
    """When obj1 and obj2 are recognized to be the same object.

    theorems allow merging:
      * point
      * line -> automatic -> hps
      * 

    Automatic merge:
      line -> hps
      segment -> points
      angle -> 

    All relations involving obj1 also applies to obj2 and vice versa.
    """
    if not isinstance(obj1, type(obj2)):
      raise ValueError('Cannot merge {} ({}) and {} ({})'.format(
          obj1, type(obj1), obj2, type(obj2)))

    if not isinstance(obj1, (Point, Segment, Line, Angle, Circle)):
      raise ValueError('Cannot merge {} and {} of type {}'.format(
          obj1, obj2, type(obj1)))

    new_rel1, new_rel2 = [], []
    for rel in self.relations:
      if obj1 in rel.init_list:
        new_rel2.append(rel.replace(obj1, obj2))
      elif obj2 in rel.init_list:
        new_rel1.append(rel.replace(obj2, obj1))

    return new_rel1, new_rel2



  def add_one(self, entity):
    if isinstance(entity, tuple(non_relations)):
      # if isinstance(entity, Point):
      #   self.all_points.append(entity)
      # elif isinstance(entity, HalfPlane):
      #   self.all_hps.append(entity)
      # self.name2obj[entity.name] = entity
      return

    if isinstance(entity, Merge):
      self.merge(*entity.init_list)
        
    relation = entity
    for obj in relation.init_list:
      if obj.name not in self.name2obj:
        self.name2obj[obj.name] = obj
        if isinstance(obj, Point):
          self.all_points.append(obj)
        elif isinstance(obj, HalfPlane):
          self.all_hps.append(obj)

    if isinstance(relation, 
                  (AngleHasMeasure, SegmentHasLength, LineHasDirection)):
      self.add_transitive_relation(relation)
      return

    # Check for existing relations
    rel_type = type(relation)
    if rel_type in self.type2rel:
      for rel in self.type2rel[rel_type]:
        if rel.init_list == relation.init_list:
          return
    else:
      # the first of its kind.
      self.type2rel[rel_type] = []

    if isinstance(relation, LineBordersHalfplane):
      line, hp = relation.init_list
      if line not in self.line2hps:
        self.line2hps[line] = []
      if hp not in self.line2hps[line]:
        if len(self.line2hps[line]) == 2:
          hp1, hp2 = self.line2hps[line]
          print(line.name, hp1.name, hp2.name, hp.name)
          raise ValueError('More than 2 halfplanes.')
        self.line2hps[line].append(hp)

    if isinstance(relation, HalfPlaneContainsPoint):
      hp, point = relation.init_list
      if hp not in self.hp2points:
        self.hp2points[hp] = []
      if point not in self.hp2points[hp]:
        self.hp2points[hp].append(point)

    self.relations.append(relation)
    self.type2rel[rel_type].append(relation)

  def augmented_relations(self):
    augment = []
    for obj in self.name2obj.values():
      if isinstance(obj, Segment) and obj not in self.obj2valrel:
        augment.append(SegmentHasLength(obj, SegmentLength()))
      if isinstance(obj, Angle) and obj not in self.obj2valrel:
        augment.append(AngleHasMeasure(obj, AngleMeasure()))
    return augment

  def add_transitive_relation(self, relation):
    obj, new_value = relation.init_list
    self.name2obj[obj.name] = obj
    self.name2obj[new_value.name] = new_value

    # import pdb; pdb.set_trace()
    if obj not in self.obj2valrel:
      # A new value assignment
      self.obj2valrel[obj] = relation
      if new_value not in self.val2valrel:
        self.val2valrel[new_value] = [relation]
      else:
        self.val2valrel[new_value] += [relation]

      self.valrel2pos[relation] = len(self.relations)
      self.relations.append(relation)
      return

    # obj has already been in obj2valrel & self.relations,
    # relation wont be added but all the relevant relations
    # currently stored will be update with new_value.
    # Now we need to update the new value for the object
    # and to all the objects that share this old value:
    old_value = self.obj2valrel[obj].init_list[1]

    if old_value == new_value:
      return

    # merge causal dependencies
    new_value.merge(old_value)

    # When we say "x has chain pos p" we meant 
    # "x is created at position p"
    # Consider:
    # step0 : a = b = v0
    # step1 : b = c = v1
    # step2 : a = d = v2
    # step3 : use c = d = v2
    # Although v2 is created at step 2,
    # The real value v = c = d is created at step 0.
    # If v2 chain pos = 2, when whittle back from c = d, we
    # end up at step 2 and fail to output step 0, 
    # the real reason why c = d.
    # so we need to update old_value chain pos to new_value
    # TODO(thtrieu): fix the following update, NOT SAFE
    # new_value._critical = old_value.critical
    # new_value._chain_position = old_value.chain_position
    # new_value._conclusion_position = old_value.conclusion_position

    if new_value not in self.val2valrel:
      self.val2valrel[new_value] = []

    # Go through all the value relation for the objects
    # that has old value and update:
    for valrel in self.val2valrel[old_value]:
      # One of the valrel here is obj2valrel[obj]
      obj, _ = valrel.init_list
      # Create a new rel with old obj and new value
      new_valrel = valrel.new_rel(new_value)
      # Similarly consider the above example,
      # if chain pos of c=v2 is 3, then trace back will
      # end up at step 2, not step 1 where c=v1 is first
      # created and also the reason why c=d (through a and b)
      # new_valrel.set_chain_position(valrel.chain_position)
      # new_valrel.set_critical(valrel.critical)
      # new_valrel.set_conclusion_position(valrel.conclusion_position)
      new_valrel.set_chain_position(relation.chain_position)
      new_valrel.set_critical(relation.critical)
      new_valrel.set_conclusion_position(relation.conclusion_position)

      self.val2valrel[new_value].append(new_valrel)
      self.obj2valrel[obj] = new_valrel
      # Update self.relations
      pos = self.valrel2pos[valrel]
      self.relations[pos] = new_valrel
      # Update self.valrel2pos
      self.valrel2pos.pop(valrel)
      self.valrel2pos[new_valrel] = pos

    # Remove old value from self.val2valrel
    self.val2valrel.pop(old_value)
    self.name2obj.pop(old_value.name)

  def add_relations(self, relations):
    for rel in relations:
      self.add_one(rel)

  def add_spatial_relations(self, line2pointgroups):
    for line in line2pointgroups:
      points_neg, points_pos = line2pointgroups[line]

      hps = self.line2hps.get(line, [])
      if not hps:  # no halfplane in state, create them:
        hp1 = HalfPlane(line.name + '_hp1')
        hp1.set_chain_position(line.chain_position)
        hp1.set_critical(line.critical)
        hp1.set_conclusion_position(line.conclusion_position)

        hp2 = HalfPlane(line.name + '_hp2')
        hp2.set_chain_position(line.chain_position)
        hp2.set_critical(line.critical)
        hp2.set_conclusion_position(line.conclusion_position)

        self.add_one(LineBordersHalfplane(line, hp1))
        self.add_one(LineBordersHalfplane(line, hp2))
        self.hp2points[hp1] = []
        self.hp2points[hp2] = []
      elif len(hps) == 1:
        hp = HalfPlane(line.name + '_hp')
        hp.set_chain_position(line.chain_position)
        hp.set_critical(line.critical)
        hp.set_conclusion_position(line.conclusion_position)
        self.add_one(LineBordersHalfplane(line, hp))
        self.hp2points[hp] = []

      # print(len(self.line2hps[line]), line.name)
      hp1, hp2 = self.line2hps[line]
      points_hp1 = self.hp2points.get(hp1, [])
      points_hp2 = self.hp2points.get(hp2, [])

      if (any(p in points_neg for p in points_hp2) or
          any(p in points_pos for p in points_hp1)):
        points_hp1, points_hp2 = points_hp2, points_hp1
        hp1, hp2 = hp2, hp1

      # Make sure that self.line2hps is also in order (neg, pos)
      self.line2hps[line] = [hp1, hp2]

      # if points_neg:
        # if hp1.name not in self.name2obj:
          # self.add_one(hp1)
      for p in points_neg:
        if p not in points_hp1:
          self.add_one(HalfPlaneContainsPoint(hp1, p))

      # if points_pos:
        # if hp2 not in self.hp2points:
        #   self.add_one(hp2)
      for p in points_pos:
        if p not in points_hp2:
          self.add_one(HalfPlaneContainsPoint(hp2, p))


class Conclusion(object):
  """The action's conclusion.
  """

  def __init__(self, *initial_list):
    if list(initial_list):
      # A list of lists
      self.topological_list = [list(initial_list)]
      self.critical = [True]
    else:
      self.topological_list = []
      self.critical = []

  def add(self, *relations):
    self.topological_list.append(list(relations))
    self.critical.append(False)

  def add_critical(self, *relations):
    self.topological_list.append(list(relations))
    self.critical.append(True)

  def __iter__(self):
    for relations, critical in zip(self.topological_list, self.critical):
      yield relations, critical

  def gather_val2objs(self):
    self.val2objs = ddict(lambda: [])
    for constructions in self.topological_list:
      for rel in constructions:
        if isinstance(rel, (SegmentHasLength, AngleHasMeasure, LineHasDirection)):
          obj, val = rel.init_list
          self.val2objs[val].append(obj)


def match_relations(premise_relations, 
                    state_relations,
                    conclusion=None,
                    match_one=False,
                    randomized=False):
  return trieu_graph_match.match_relations(
      premise_relations=premise_relations, 
      state_relations=state_relations,
      conclusio=conclusion,
      randomized=randomized,
      distinct=distinct)