import action_chain_lib
from matplotlib import pyplot as plt
import sketch
import numpy as np
import traceback
import theorems
import theorems_utils
import pickle as pkl
import state
from collections import defaultdict as ddict

import geometry 

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import PointCentersCircle, Merge
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint



def equal(struct1, struct2):
  if not isinstance(struct1, type(struct2)):
    return False

  if not isinstance(struct1, (tuple, list, dict)):
    return struct1 == struct2

  if not len(struct1) == len(struct2):
    return False

  if isinstance(struct1, dict):
    if not all([key in struct2 for key in struct1]):
      return False

    return all([equal(struct2[key], val)
                for key, val in struct1.items()])

  for i, x in enumerate(struct1):
    y = struct2[i]
    if not equal(x, y):
      return False
  return True


def equal_dict(d1, d2):
  if len(d1) != len(d2):
    return False

  for k, v in d1.items():
    if d2.get(k, object()) != v:
      return False 

  return True 


# Then match recursively
def recursively_match_best_effort(
    query_relations,
    state_candidates,
    object_mappings,
    distinct=None,
    best=[{}],
    pdb_at=None):
  """Python generator that yields dict({premise_object: state_object}).

  Here we are trying to perform the edge-induced subgraph isomorphism
  matching algorithm. In other words, given a list of edges in the action
  premise graph, we try to see if these edges present as a subset of the
  edges in the state graph.

  In this code base, a graph edge is represented using an object of type
  geometry.Relation, with attribute self.init_list = [obj1, obj2] meaning
  this edge is connecting two nodes corresponding to obj1 and obj2. Some
  examples of edge types include PointEndsSegment, HalfplaneCoversAngle, 
  SegmentHasLength, etc. See all classes that inherit from class Relation
  in `geometry.py` to for a full list of edge types.

  Args:
    query_relations: A list of geometry.Relation objects, each represent
      an edge in the action premise graph.
    state_candidates: A dictionary {t: [list of relations with type t]}.
      examples of type t include PointEndsSegment, HalfplaneCoversAngle, 
      SegmentHasLength, etc. See `all classes that inherit from class 
      Relation in `geometry.py` to for a full list. This dictionary stores 
      all edges in the state's graph representation.
    object_mappings: A dictionary {premise_object: state_object} mapping
      the nodes in premise graph and their matched counterpart in the state 
      graph that we already know. This means the remaining job is to add to 
      this dictionary the rest of mappings such that all premise edges stored
      in premise_relations are matched with edges stored in state_candidates.
    distinct: A list of pairs (premise_obj1, premise_obj2). Each pair is 
      two nodes in the premise graph. This is to indicate that if premise_obj1
      is matched to state_obj1 and premise_obj2 is matched to state_obj2,
      then state_obj1 must be different to state_obj2, otherwise they can
      be the same. For example, in the case of two equal triangles ABC=DEF
      in the premise that got matched to ADB=ADC in the state, then AB and
      DE can be the same, while the rest must be distinct (and therefore
      presented in this `distinct` list).

      *Note*: An empty distinct list indicates that every pair of objects 
      in the premise *is distinct* - an unfortunate convention that will 
      need fixing.
    timeout: A limit on execution time of this function (in seconds).

  Yields:
    A dictionary {premise_object: state_object} that maps from nodes in
    premise graph to nodes in state graph such that all edges from
    query_relations (premise edges) is matched with some edges in
    state_candidates (state edges). If there is no successful match
    the generator should not yield any object and raise StopIteration
    right away.
  """

  if len(object_mappings) > len(best[0]):
    best[0] = dict(object_mappings)
  
  if pdb_at and equal_dict(object_mappings, pdb_at):
      import pdb; pdb.set_trace()

  if not query_relations:
    # There is not any premise edge to match:
    return [object_mappings]

  query0 = query_relations[0]
  # At this recursion level we try to match premise_rel0
  rel_type = type(query0)

  all_matches = []
  # Enumerate through possible edge match:
  for _, candidate in enumerate(state_candidates.get(rel_type, [])):
    # Now we try to match edge query0 to candidate, by checking
    # if this match will cause any conflict, if not then we proceed
    # to query1 in the next recursion depth.

    # Suppose edge query0 connects nodes a, b in premise graph
    # and edge candidate connects nodes c, d in state graph:
    (a, b), (c, d) = query0.init_list, candidate.init_list

    # Special treatment for half pi:
    # We match the exact object, not just its type
    if a == geometry.halfpi and c != a:
      continue
    if b == geometry.halfpi and d != b:
      continue

    # Now we want to match a->c, b->d without any conflict,
    # if there is conflict then candidate cannot be matched to query0.
    if (object_mappings.get(a, c) != c or
        object_mappings.get(b, d) != d or
        # Also check for inverse map if there is any:
        object_mappings.get(c, a) != a or
        object_mappings.get(d, b) != b):
      continue  # move on to the next candidate.

    new_mappings = {a: c, b: d}
    # Check for distinctiveness:
    if not distinct:  # Everything is distinct except numeric values.
      # Add the inverse mappings, so that now a <-> c and b <-> d,
      # so that in the future c cannot be matched with any other node
      # other than a, and d cannot be matched with any other node other
      # than b.
      if not isinstance(a, (SegmentLength, AngleMeasure, LineDirection)):
        new_mappings[c] = a
      if not isinstance(b, (SegmentLength, AngleMeasure, LineDirection)):
        new_mappings[d] = b
    else:
      # Check if new_mappings is going to conflict with object_mappings
      # A conflict happens if there exist a' -> c in object_mappings and
      # (a, a') presented in distinct. Likewise, if there exists b' -> d
      # in object_mappings and (b, b') presented in distinct then
      # a conflict happens. 
      conflict = False
      for distinct_pair in distinct:
        if a not in distinct_pair and b not in distinct_pair:
          continue  # nothing to check here

        x, y = distinct_pair
        x_map = object_mappings.get(x, new_mappings.get(x, None))
        y_map = object_mappings.get(y, new_mappings.get(y, None))
        # either x or y will be in new_mappings by the above "if",
        # so x_map and y_map cannot be both None
        if x_map == y_map:
          conflict = True
          break

      if conflict:
        continue  # move on to the next candidate.

    # Add {query0 -> candidate} to new_mappings
    new_mappings[query0] = candidate
    # Update object_mappings by copying all of its content
    # and then add new_mappings.
    appended_mappings = dict(object_mappings, **new_mappings)
    all_matches += recursively_match_best_effort(
        query_relations=query_relations[1:], 
        state_candidates=state_candidates,
        object_mappings=appended_mappings,
        distinct=distinct,
        best=best,
        pdb_at=pdb_at)
  return all_matches



class DebugObjects(object):
  """Saves the explore chain from root upto current point."""

  def __init__(self):
    self.state = [None] * 100
    self.canvas = [None] * 100
    self.action = [None] * 100
    self.depth = None
  
  def update(self, depth, obj=None):
    self.depth = depth
    if isinstance(obj, sketch.Canvas):
      self.canvas[depth] = obj
    elif isinstance(obj, state.State):
      self.state[depth] = obj
    elif isinstance(obj, theorems.Action):
      self.action[depth] = obj
    else:
      raise ValueError('Unrecognized type {}'.format(type(obj)))

  def save_chain(self, filename):
    i = 0
    steps = []
    while i <= self.depth:
      state, action = self.state[i+1], self.action[i]
      if action is None:
        break
      matching_command = {
          name: action.mapping[obj].name
          for name, obj in action.theorem.names.items()}

      # same as mapping, but only map from
      # theorem -> workspace, not vice versa.
      new_mapping = {}
      # to account for new objects created by
      # spatial relations and not in the matching stage.
      line2hps = {}
      for x, y in action.mapping.items():
        if y not in self.action[i].new_objects:
          continue

        new_mapping[x.name] = y.name
        # take care of the hps that are created 
        # when spatial relations are added.
        if isinstance(y, Line):
          hp1, hp2 = state.line2hps[y]
          line2hps[y.name] = [hp1.name, hp2.name]

      steps.append({
          'theorem': action.theorem.__class__.__name__,
          'command_str': ' '.join([
              '{}={}'.format(x, y) for x, y in 
              matching_command.items()
          ]),
          'mapping': new_mapping,
          'line2hps': line2hps
      })
      i += 1
    
    with open(filename, 'wb') as f:
      pkl.dump(steps, f, protocol=-1)

  def load_chain(self, filename, state, canvas):
    with open(filename, 'rb') as f:
      # old chain with old names
      old_steps = pkl.load(f)
    
    state = state.copy()
    canvas = canvas.copy()
    # new names go here:
    result_steps = []
    old_to_new_names = {}
    for i, step in enumerate(old_steps):
      theorem = theorems.theorem_from_name[step['theorem']]

      name_maps = []  # theorem to new_name for this action.
      for map_str in step['command_str'].split():
        name_in_theorem, name_in_workspace = map_str.split('=')
        if name_in_workspace in old_to_new_names:
          name_in_workspace = old_to_new_names[name_in_workspace]
        name_maps.append([name_in_theorem, name_in_workspace])

      command_str = ' '.join([
          '{}={}'.format(x, y)
          for x, y in name_maps
      ])

      try:
        mapping = dict(
            (theorem.names[a], action_chain_lib._find(state, b))
            if a in theorem.names
            else (action_chain_lib._find_premise(theorem.premise_objects, a), 
                  action_chain_lib._find(state, b))
            for a, b in name_maps)
      except Exception:
        traceback.print_exc()
        continue

      try:
        action_gen = theorem.match_from_input_mapping(
            state, mapping, randomize=False)
        action = action_gen.next()
      except StopIteration:
        print(theorem, command_str)
        best, miss = self.why_fail_to_match(theorem, state, mapping)
        import pdb; pdb.set_trace()
        raise ValueError('Matching not found {} {}'.format(theorem, command_str))

      db.update(i, state)
      db.update(i, canvas)
      db.update(i, action)
      action.set_chain_position(i)

      state.add_relations(action.new_objects)
      line2pointgroups = action.draw(canvas)
      state.add_spatial_relations(line2pointgroups)
      canvas.update_hps(state.line2hps)

      for x, y in action.mapping.items():
        if y not in action.new_objects:
          continue
        old_y_name = step['mapping'][x.name]
        old_to_new_names[old_y_name] = y.name
        if isinstance(y, Line):
          hp1, hp2 = state.line2hps[y]
          hp1_old_name, hp2_old_name = step['line2hps'][old_y_name]
          old_to_new_names[hp1_old_name] = hp1.name
          old_to_new_names[hp2_old_name] = hp2.name

      result_steps.append((theorem, command_str))
      print('Loaded {}'.format(action.to_str()))
      state.print_all_equal_angles()
      print('====')
    return result_steps

  def report(self):
    canvas_state = []
    print('\nDebug:')
    for i, (state, canvas, action) in enumerate(
        zip(self.state, self.canvas, self.action)):

      if (i > self.depth):
        break

      assert (state is None) == (canvas is None)
      if state is None:
        continue

      action_str = action.to_str() if action else 'None'
      print(i, action_str)
      canvas_state.append((canvas, state))
  
    n = len(canvas_state)
    nrows = int(np.ceil(n/3.))
    _, axes = plt.subplots(
        nrows=nrows, ncols=3, figsize=(nrows * 5, 3 * 5))
    for i, (canvas, state) in enumerate(canvas_state):
      ax = axes[i//3, i%3]
      canvas.plt_show(ax, state, [], mark_segment=0)
    plt.show(block=False)

  def why_fail_to_match(self, theorem, state, mapping={}):
    type2rel = ddict(lambda: [])
    for rel in state.relations:
      type2rel[type(rel)].append(rel)

    best = [{}]
    recursively_match_best_effort(
        theorem.premise,
        type2rel,
        mapping,
        best=best
    )
    best = best[0] 
    miss = [rel for rel in theorem.premise 
            if rel not in best]

    # recursively_match_best_effort(
    #     theorem.premise,
    #     state.type2rel,
    #     mapping,
    #     pdb_at=best
    # )
    return best, miss
      

db = None


def get_db():
  global db
  if db is None:
    db = DebugObjects()
  return db