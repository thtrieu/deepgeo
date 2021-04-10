from collections import defaultdict as ddict

import geometry 

from geometry import AngleXX, AngleXO, DirectionOfFullAngle
from geometry import AngleXXOfFullAngle, AngleXOOfFullAngle
from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import Distinct, DistinctLine, DistinctPoint
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, LineBordersHalfplane
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


class SpatialDistinct(object):
  """A simple hack to force everything go through state.add_one().
  So we can have state.inc() saving everything it adds
  """

  def __init__(self, rel):
    self.rel = rel


class State(object):

  def __init__(self):
    self.relations = []
    # For transitive handling:
    self.type2rel = {}
    self.obj2valrel = {}
    self.val2valrel = {}
    self.name2obj = {}
    self.old2newvals = {}

    self.all_points = []
    self.all_hps = []
    self.inc = []

    # distinct relations does not go to self.relations
    # but goes here instead, for easy removal
    # proof_distincts are for distinct in conclusions
    self.proof_distincts = {}
    # spatial_distincts are for distinct in spatial relations
    self.spatial_distincts = {}
    # everytime a distinct is added to spatial_distinct
    # its duplicate in proof_distinct (if any) is removed.

    self.line2hps = {}
    self.hp2points = {}

  def distinct_relations(self):
    return self.proof_distincts.values() + self.spatial_distincts.values()

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
    copied.proof_distincts = _copy(self.proof_distincts)
    copied.spatial_distincts = _copy(self.spatial_distincts)

    # For identifying halfplanes
    copied.line2hps = _copy(self.line2hps)
    copied.hp2points = _copy(self.hp2points)
    return copied

  def segment_between(self, p1, p2):
    if isinstance(p1, str):
      p1 = self.name2obj[p1]
    if isinstance(p2, str):
      p2 = self.name2obj[p2]
    
    point2seg = {p1: set(), p2: set()}
    for r in self.relations:
      if isinstance(r, PointEndsSegment) and r.init_list[0] in [p1, p2]:
        point2seg[r.init_list[0]].add(r.init_list[1])
    seg = list(point2seg[p1].intersection(point2seg[p2]))[0]
    return seg


  def angle_between(self, l1, l2):
    if isinstance(l1, str):
      l1 = self.name2obj[l1]
    if isinstance(l2, str):
      l2 = self.name2obj[l2]
    d1, d2 = None, None
    for rel in self.relations:
      if isinstance(rel, LineHasDirection) and rel.init_list[0] == l1:
        d1 = rel.init_list[1]
      if isinstance(rel, LineHasDirection) and rel.init_list[0] == l2:
        d2 = rel.init_list[1]
    assert d1 and d2

    fa1, fa2 = set(), set()
    for rel in self.relations:
      if isinstance(rel, DirectionOfFullAngle) and rel.init_list[0] == d1:
        fa1.add(rel.init_list[1])
      if isinstance(rel, DirectionOfFullAngle) and rel.init_list[0] == d2:
        fa2.add(rel.init_list[1])
    fa = list(fa1.intersection(fa2))[0]
    assert fa

    xx, xo = None, None
    for rel in self.relations:
      if isinstance(rel, AngleXXOfFullAngle) and rel.init_list[1] == fa:
        xx = rel.init_list[0]
      if isinstance(rel, AngleXOOfFullAngle) and rel.init_list[1] == fa:
        xo = rel.init_list[0]

    assert xx and xo
    return xx, xo

  def is_equal(self, obj1, obj2):
    if isinstance(obj1, str):
      obj1 = self.name2obj(obj1)
    if isinstance(obj2, str):
      obj2 = self.name2obj(obj2)
    
    if obj1 not in self.obj2valrel or obj2 not in self.obj2valrel:
      return False

    assert isinstance(obj1, type(obj2))
    val1 = self.obj2valrel[obj1].init_list[1]
    valrel2 = self.obj2valrel[obj2]
    return valrel2 in self.val2valrel[val1]

  def name_map(self, struct):
    if isinstance(struct, (list, tuple, set)):
      return [self.name_map(o) for o in struct]
    elif isinstance(struct, dict):
      r = {}
      for key, value in struct.items():
        key = self.name_map(key)
        if isinstance(key, list):
          key = tuple(key)
        value = self.name_map(value)
        r.update({key: value})
      return r
    else:
      if struct is None:
        return struct
      pre = str(getattr(struct, 'chain_position', 'None'))
      if pre != 'None':
        pre += '.'
      else:
        pre = ''
      if isinstance(struct, Segment):
        return pre + self.segment_name(struct)
      elif isinstance(struct, Angle):
        ang1, ang2 = self.angle_name(struct)
        return pre + ang1  # , pre + ang2
      elif isinstance(struct, HalfPlane):
        return pre + self.hp_name(struct)
      elif isinstance(struct, PointEndsSegment):
        p, s = struct.init_list
        return pre + p.name + '[' + self.segment_name(s)
      elif isinstance(struct, SegmentHasLength):
        s, l = struct.init_list
        return pre + self.segment_name(s) + '=' + l.name
      # elif isinstance(struct, HalfplaneCoversAngle):
      #   hp, a = struct.init_list
      #   return self.hp_name(hp) + '/' + self.angle_name(a)
      elif isinstance(struct, Merge):
        return pre + 'merge({}=>{})'.format(
            self.name_map(struct.from_obj), 
            self.name_map(struct.to_obj))
      elif hasattr(struct, 'name'):
        return pre + struct.name
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
    if l is None:
      return hp.name
    if p is None:
      return 'hp({}, ?)'.format(l.name)
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
    result = []

    for [hp1, hp2], [l1, l2] in self.hp_and_line_of_angle(angle):
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
        name = '<{} {} {}>'.format(angle.name, l1.name, l2.name)
        if isinstance(angle, AngleXX):
          name +=  '_xx'
        elif isinstance(angle, AngleXO):
          name += '_xo'
        else:
          raise ValueError('Angle is not XX nor XO.')
        result.append(name)
        break

          
        # import pdb; pdb.set_trace()
        # raise ValueError(
        #     'Not found intersection of {} and {}'.format(l1.name, l2.name))

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
      result.append('|{} {} {} {}|'.format(angle.name, p1, intersection, p2))
    
    if result == []:
      return [angle.name]
    return result

  def all_parallel_lines(self):
    dir2lines = ddict(lambda: [])
    for rel in self.relations:
      if isinstance(rel, LineHasDirection):
        line, dir = rel.init_list
        dir2lines[dir].append(line)
    return dir2lines

  def print_all_parallel_lines(self):
    for dir, para_lines in self.all_parallel_lines().items():
      para_lines = [l.name for l in para_lines]
      print('>> ' + dir.name + ' = ' + ' = '.join(para_lines))

  def all_equal_angles(self):
    measure2angs = ddict(lambda: [])
    for rel in self.relations:
      if isinstance(rel, AngleHasMeasure):
        angle, measure = rel.init_list
        measure2angs[measure].extend(self.angle_name(angle))
    return measure2angs

  def print_all_equal_angles(self):
    for measure, equal_angles in self.all_equal_angles().items():
      print('>> ' + measure.name + ' = ' + ' = '.join(equal_angles))

  def all_equal_segments(self):
    len2seg = ddict(lambda: [])
    for rel in self.relations:
      if isinstance(rel, SegmentHasLength):
        seg, length = rel.init_list
        len2seg[length].append(seg)
    return len2seg

  def print_all_equal_segments(self):
    for length, equal_segments in self.all_equal_segments().items():
      equal_segments = [self.segment_name(s) for s in equal_segments]
      print('>> ' + length.name + ' = ' + ' = '.join(equal_segments))

  def ends_of_segment(self, segment):
    points = []
    for p_seg in self.type2rel.get(PointEndsSegment, []):
      if segment == p_seg.init_list[1]:
        points.append(p_seg.init_list[0])
    return points

  def hp_and_line_of_angle(self, angle):

    fangle = None
    rel_type = AngleXXOfFullAngle if isinstance(angle, AngleXX) else AngleXOOfFullAngle
    for rel in self.type2rel[rel_type]:
      if angle == rel.init_list[0]:
        fangle = rel.init_list[1]

    if fangle is None:
      return

    ds = []
    for rel in self.type2rel[DirectionOfFullAngle]:
      if fangle == rel.init_list[1]:
        ds.append(rel.init_list[0])

    if len(ds) != 2:
      raise ValueError('len(ds) must = 2 but is {}'.format(len(ds)))
    d1, d2 = ds

    l1s, l2s = [], []
    for rel in self.val2valrel[d1]:
      l1s.append(rel.init_list[0])

    for rel in self.val2valrel[d2]:
      l2s.append(rel.init_list[0])

    same_sign = isinstance(angle, geometry.AngleXX) 
    for l1 in l1s:
      for l2 in l2s:
        hp1_neg, hp1_pos = self.line2hps[l1]
        hp2_neg, hp2_pos = self.line2hps[l2]
        if same_sign:
          yield [0, 0], [l1, l2]
          yield [1, 1], [l1, l2]
        else:
          yield [0, 1], [l1, l2]
          yield [1, 0], [l1, l2]

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
    self.inc += [entity]

    if isinstance(entity, SpatialDistinct):
      x, y = entity.rel.init_list
      self.spatial_distincts[(x, y)] = entity.rel
      return

    if not isinstance(entity, geometry.Relation):
      # if isinstance(entity, Point):
      #   self.all_points.append(entity)
      # elif isinstance(entity, HalfPlane):
      #   self.all_hps.append(entity)
      # self.name2obj[entity.name] = entity
      return

    if isinstance(entity, Merge):
      self.remove(entity.from_obj)
      # print('set {} merge graph'.format(entity.to_obj.name))
      entity.to_obj.merge_graph[self] = entity.merge_graph
      # entity.from_obj.merge_graph[self]['equivalents'] += [entity.to_obj]
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

    # Special treatments for distinct stuff.
    if isinstance(relation, Distinct):
      obj1, obj2 = relation.init_list
      if (obj1, obj2) in self.spatial_distincts or (obj2, obj1) in self.spatial_distincts:
        return
      if (obj1, obj2) in self.proof_distincts or (obj2, obj1) in self.proof_distincts:
        return

      if isinstance(obj1, Line):
        rel = DistinctLine(obj1, obj2)
      else:
        rel = DistinctPoint(obj1, obj2)

      self.proof_distincts[(obj1, obj2)] = rel
      return  # distinct relations does not go to self.relations or type2rel

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
          # print('{} has >= 3 hps: {}, {}, {}'.format(
          #     line.name, hp1.name, hp2.name, hp.name))
          # raise ValueError('More than 2 halfplanes.')
          # ====> this is allowed *before* merging.
        self.line2hps[line].append(hp)
      
      if hp not in self.hp2points:
        self.hp2points[hp] = []

    if isinstance(relation, HalfPlaneContainsPoint):
      hp, point = relation.init_list
      if hp not in self.hp2points:
        self.hp2points[hp] = []
      if point not in self.hp2points[hp]:
        self.hp2points[hp].append(point)

    self.relations.append(relation)
    self.type2rel[rel_type].append(relation)

  def has_relation(self, rel):
    a, b = rel.init_list
    for r in self.type2rel[type(rel)]:
      x, y = r.init_list
      try:
        if a == x or a in x.merge_graph[self]['equivalents']:
          if b == y or b in y.merge_graph[self]['equivalents']:
            return True
      except:
        continue
    return False

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
    return []

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

    for rel in relations:
      if isinstance(rel, Merge):
        self.add_one(rel)

  def add_spatial_relations(self, line2pointgroups, chain_pos=None):
    for line in line2pointgroups:
      points_neg, points_pos = line2pointgroups[line]

      # Now for `line`, we check to see if it already has 2 hps in self.
      # if not, create on demand.
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
        # self.hp2points[hp1] = []
        # self.hp2points[hp2] = []

      elif len(hps) == 1:
        hp = HalfPlane(line.name + '_hp')
        hp.set_chain_position(line.chain_position)
        hp.set_critical(line.critical)
        hp.set_conclusion_position(line.conclusion_position)
        self.add_one(LineBordersHalfplane(line, hp))
        # self.hp2points[hp] = []

      hp1, hp2 = self.line2hps[line]
      points_hp1 = self.hp2points.get(hp1, [])
      points_hp2 = self.hp2points.get(hp2, [])

      # Make sure that self.line2hps is also in order (neg, pos)
      if (any(p in points_neg for p in points_hp2) or
          any(p in points_pos for p in points_hp1)):
        points_hp1, points_hp2 = points_hp2, points_hp1
        hp1, hp2 = hp2, hp1

      self.line2hps[line] = [hp1, hp2]
      hp1.sign = -1
      hp2.sign = +1

      for p1 in points_neg:
        for rel in self.type2rel[LineContainsPoint]:
          l0, p0 = rel.init_list
          if l0 == line:  # l0 goes through p0 but not p1.
            self.add_spatial_distincts(p1, p0)
          elif p0 == p1:  # l0 goes through p1 but line does not.
            self.add_spatial_distincts(line, l0)

        if p1 in points_hp1:
          continue

        self.add_one(HalfPlaneContainsPoint(hp1, p1))
        # Add to distincts
        for p2 in points_hp2:
          self.add_spatial_distincts(p1, p2)  # p1 and p2 on diff hps
        
      # Update the value of points_hp1 after new stuff being added above.
      points_hp1 = self.hp2points.get(hp1, [])

      # Repeat the same steps for points_pos:
      for p2 in points_pos:
        for rel in self.type2rel[LineContainsPoint]:
          l0, p0 = rel.init_list
          if l0 == line:
            self.add_spatial_distincts(p2, p0)
          elif p0 == p2:
            self.add_spatial_distincts(line, l0)

        if p2 in points_hp2:
          continue
        self.add_one(HalfPlaneContainsPoint(hp2, p2))

        # Add to distincts
        for p1 in points_hp1:
          self.add_spatial_distincts(p2, p1)
  
  def add_spatial_distincts(self, x, y):
    if (x, y) in self.proof_distincts:
      self.proof_distincts.pop((x, y))
    elif (y, x) in self.proof_distincts:
      self.proof_distincts.pop((y, x))
    
    if (x, y) in self.spatial_distincts or (y, x) in self.spatial_distincts:
      return

    if isinstance(x, Line):
      rel = DistinctLine(x, y)
    else:
      rel = DistinctPoint(x, y)

    self.add_one(SpatialDistinct(rel))


class Conclusion(object):
  """The action's conclusion."""

  def __init__(self, *initial_list):
    if list(initial_list):
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