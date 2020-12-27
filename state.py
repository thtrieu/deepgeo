from collections import defaultdict as ddict

import geometry 

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
    self.name2obj = {}
    self.all_points = []
    self.all_hps = []
    self.old2newvals = {}

    self.line2hps = {}
    self.hp2points = {}

  def pop(self):
    for val in self.val2valrel:
      val.edges.pop(self)
    
    for rel in self.relations:
      rel.pop(self)

  def copy(self):
    copied = State()
    copied.relations = _copy(self.relations)
    copied.type2rel = _copy(self.type2rel)
    copied.obj2valrel = _copy(self.obj2valrel)
    copied.val2valrel = _copy(self.val2valrel)
    # copied.valrel2pos = _copy(self.valrel2pos)
    copied.name2obj = _copy(self.name2obj)
    copied.old2newvals = _copy(self.old2newvals)

    for rel in self.relations:
      rel.copy(old_state=self, new_state=copied)

    for val in self.val2valrel:
      val.copy(old_state=self, new_state=copied)
    
    for k, v in self.old2newvals.items():
      k.copy(old_state=self, new_state=copied)
      v.copy(old_state=self, new_state=copied)

    copied.all_points = _copy(self.all_points)
    copied.all_hps = _copy(self.all_hps)

    # For identifying halfplanes
    copied.line2hps = _copy(self.line2hps)
    copied.hp2points = _copy(self.hp2points)
    return copied

  def name_map(self, struct):
    if isinstance(struct, (list, tuple)):
      return [self.name_map(o) for o in struct]
    elif isinstance(struct, dict):
      return {
        self.name_map(key): self.name_map(value)
        for (key, value) in struct.items()
      }
    else:
      if isinstance(struct, Segment):
        return self.segment_name(struct)
      elif isinstance(struct, Angle):
        return self.angle_name(struct)
      elif isinstance(struct, HalfPlane):
        return self.hp_name(struct)
      elif isinstance(struct, PointEndsSegment):
        p, s = struct.init_list
        return p.name + '[' + self.segment_name(s)
      elif isinstance(struct, SegmentHasLength):
        s, l = struct.init_list
        return self.segment_name(s) + '=' + l.name
      elif isinstance(struct, HalfplaneCoversAngle):
        hp, a = struct.init_list
        return self.hp_name(hp) + '/' + self.angle_name(a)
      elif isinstance(struct, Merge):
        return 'merge({}=>{})'.format(
            self.name_map(struct.from_obj), 
            self.name_map(struct.to_obj))
      elif hasattr(struct, 'name'):
        return struct.name
      else:
        return struct

  def hp_name(self, hp):
    l, p = None, None
    for r in self.relations:
      if isinstance(r, LineBordersHalfplane) and r.init_list[1] == hp:
        l = r.init_list[0]
      if isinstance(r, HalfPlaneContainsPoint) and r.init_list[0] == hp:
        p = r.init_list[1]
      if l and p:
        break
    return 'hp({},{})'.format(l.name, p.name)

  def halfpi_val(self):
    return self.obj2valrel[geometry.halfpi].init_list[1]

  def segment_name(self, segment):
    # return segment.name
    ends = self.ends_of_segment(segment)
    if ends:
      return ''.join([p.name for p in ends])
    else:
      return segment.name

  def angle_name(self, angle):
    if angle == geometry.halfpi:
      return angle.name
    
    [hp1, hp2], [l1, l2] = self.hp_and_line_of_angle(angle)
    line2points = {l1: [], l2: []}
    for rel in self.type2rel[LineContainsPoint]:
      line, point = rel.init_list
      if line in line2points:
        line2points[line].append(point)
    
    intersection = None
    for p in line2points[l1]:
      if p in line2points[l2]:
        intersection = p.name
        break 
    
    if not intersection:
      import pdb; pdb.set_trace()
      raise ValueError(
          'Not found intersection of {} and {}'.format(l1.name, l2.name))

    def _find_point_on_line_in_hp(line, other_line, hp_idx):
      hp = self.line2hps[other_line][hp_idx]
      hp_ = self.line2hps[other_line][1-hp_idx]
      
      p_name = None
      for p in line2points[line]:
        if p in self.hp2points.get(hp, {}):
          p_name = p.name
          break 
      if p_name is None:
        for p in line2points[line]:
          if p in self.hp2points.get(hp_, {}):
            p_name = '!' + p.name
            break 
      if p_name is None:
        return '?{}'.format(line.name)
      return p_name
      
    p1 = _find_point_on_line_in_hp(l1, l2, hp2)
    p2 = _find_point_on_line_in_hp(l2, l1, hp1)
    
    return '|{} {} {} {}|'.format(angle.name, p1, intersection, p2)

  def print_all_equal_angles(self):
    val2valrels = ddict(lambda: [])
    for rel in self.relations:
      if isinstance(rel, AngleHasMeasure):
        angle, measure = rel.init_list
        val2valrels[measure].append(self.angle_name(angle))

    for measure, equal_angles in val2valrels.items():
      print('>> ' + measure.name + ' = ' + ' = '.join(equal_angles))

  def print_all_equal_segments(self):
    val2valrels = ddict(lambda: [])
    for rel in self.relations:
      if isinstance(rel, SegmentHasLength):
        seg, length = rel.init_list
        val2valrels[length].append(self.segment_name(seg))

    for length, equal_segments in val2valrels.items():
      print('>> ' + length.name + ' = ' + ' = '.join(equal_segments))

  def ends_of_segment(self, segment):
    points = []
    for p_seg in self.type2rel.get(PointEndsSegment, []):
      if segment == p_seg.init_list[1]:
        points.append(p_seg.init_list[0])
    return points

  def hp_and_line_of_angle(self, angle):
    hps = []
    lines = []
    for hp_a in self.type2rel.get(HalfplaneCoversAngle, []):
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

  def to_str(self, join=', '):
    result = []
    for r in self.relations:
      a, b = r.init_list
      result += ['({}, {}, \'{}\', {}, \'{}\')'.format(
          type(r).__name__,
          type(a).__name__,
          a.name,
          type(b).__name__,
          b.name)]
    result = join.join(result)
    return '[{}]'.format(result)

  def add_one(self, entity):
    if isinstance(entity, tuple(non_relations)):
      # if isinstance(entity, Point):
      #   self.all_points.append(entity)
      # elif isinstance(entity, HalfPlane):
      #   self.all_hps.append(entity)
      # self.name2obj[entity.name] = entity
      return

    if isinstance(entity, Merge):
      self.remove(entity.from_obj)
      entity.to_obj.merge_graph[self] = entity.merge_graph
      return
        
    relation = entity
    for obj in relation.init_list:
      if obj.name not in self.name2obj:
        self.name2obj[obj.name] = obj
        if isinstance(obj, Point):
          self.all_points.append(obj)
        elif isinstance(obj, HalfPlane):
          self.all_hps.append(obj)

    # print('adding '+ self.name_map(relation))

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

  def remove(self, obj):
    # mask = [True] * len(self.relations)
    self.relations = filter(
        lambda rel: obj not in rel.init_list,
        self.relations)
  
    for t, rels in self.type2rel.items():
      self.type2rel[t] = filter(
          lambda rel: obj not in rel.init_list,
          rels)

    for val, valrels in self.val2valrel.items():
      self.val2valrel[val] = filter(
          lambda rel: obj not in rel.init_list,
          valrels)

    if obj in self.obj2valrel:
      self.obj2valrel.pop(obj)

    if obj in self.all_hps:
      self.all_hps.remove(obj)

    if obj in self.all_points:
      self.all_points.remove(obj)

    if obj.name in self.name2obj:
      self.name2obj.pop(obj.name)

    if obj in self.line2hps:
      self.line2hps.pop(obj)
    
    for line in self.line2hps:
      if obj in self.line2hps[line]:
        self.line2hps[line].remove(obj)

    if obj in self.hp2points:
      self.hp2points.pop(obj)
    
    for hp in self.hp2points:
      ps = self.hp2points[hp]
      if obj in ps:
        ps.remove(obj)

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

    if obj in self.obj2valrel:
      old_value = self.obj2valrel[obj].init_list[1]
    else:
      self.obj2valrel[obj] = relation
      if new_value not in self.val2valrel:
        self.val2valrel[new_value] = [relation]
      else:
        self.val2valrel[new_value] += [relation]

      # self.valrel2pos[relation] = len(self.relations)
      self.relations.append(relation)

      # Now, we use self.old2newvals to track to the
      # current old_value of this new_value:
      old_value = new_value
      while old_value in self.old2newvals:
        old_value = self.old2newvals[old_value]

    # merge causal dependencies
    new_value.merge_tmp_clique(state=self)
    new_value.merge(old_value, state=self)

    if old_value == new_value:
      return

    self.old2newvals[old_value] = new_value
    if new_value in self.old2newvals:
      self.old2newvals.pop(new_value)
    
    if new_value not in self.val2valrel:
      self.val2valrel[new_value] = []
    
    # Go through all the value relation for the objects
    # that has old value and update:
    for valrel in self.val2valrel[old_value]:
      # One of the valrel here is obj2valrel[obj]
      obj, _ = valrel.init_list
      # Create a new rel with old obj and new value
      new_valrel = valrel.new_rel(new_value)
      new_valrel.set_chain_position(relation.chain_position)
      new_valrel.set_critical(relation.critical)
      new_valrel.set_conclusion_position(relation.conclusion_position)

      self.val2valrel[new_value].append(new_valrel)
      self.obj2valrel[obj] = new_valrel

      # Update self.relations
      for i, rel in enumerate(self.relations):
        if rel == valrel:
          self.relations[i] = new_valrel

      # pos = self.valrel2pos[valrel]
      # self.relations[pos] = new_valrel
      # Update self.valrel2pos
      # self.valrel2pos.pop(valrel)
      # self.valrel2pos[new_valrel] = pos

    # Remove old value from self.val2valrel
    self.val2valrel.pop(old_value)
    self.name2obj.pop(old_value.name)

  def add_relations(self, relations):
    for rel in relations:
      if not isinstance(rel, Merge):
        self.add_one(rel)
    
    # This is copied for extracting proof related to transitive values
    # But will not be copied to next state.
    self.val2valrel_copy = {
      key: list(value)
      for key, value in self.val2valrel.items()
    }

    for rel in relations:
      if isinstance(rel, Merge):
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
  """The action's conclusion."""

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