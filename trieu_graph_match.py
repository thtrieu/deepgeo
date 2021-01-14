from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from action_chain_lib import recursively_auto_merge

import numpy as np
# import random

from numpy.lib.arraysetops import isin
import geometry
# import time
import profiling
# import theorems_utils
import debugging


from state import Conclusion
from collections import defaultdict as ddict
from profiling import Timer

import cython_graph_match
# import parallel_graph_match

from theorems import FundamentalTheorem, SamePairSkip, all_theorems

from geometry import AngleXX, FullAngle, Point, Line, Segment, Angle, HalfPlane, Circle, SelectAngle, TransitiveRelation 
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, LineBordersHalfplane
from geometry import PointCentersCircle
from geometry import Merge, Distinct
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


def strip_match_relations(premise_relations, conclusion_relations, state_relations):
  """Strip unrelated relations in state_relation and sort the relations."""
  # Sort according to number of relations, in order to minimize
  # the branching factor early on when match recursively.
  # This also help with early pruning during recursive calls.
  relevant_state_candidates = {}


  for rel in premise_relations + conclusion_relations:
    rel_type = type(rel)
    if rel_type not in relevant_state_candidates:
      relevant_state_candidates[rel_type] = []

  for state_rel in state_relations:
    if type(state_rel) in relevant_state_candidates:
      relevant_state_candidates[type(state_rel)].append(state_rel)

  rel_branch_count = []
  for rel in premise_relations:
    branch_count = len(relevant_state_candidates[type(rel)])
    if isinstance(rel, Distinct):
      branch_count *= 2
    rel_branch_count.append(branch_count)

  if premise_relations == []:
    return premise_relations, relevant_state_candidates

  # Sort according to branch_count
  premise_relations, _ = zip(*sorted(
      zip(premise_relations, rel_branch_count), 
      reverse=True))  # very important speedup!

  # Finally, we insert SamePairSkip before the first member of its group:
  optimized_order_premise_relations = []  # return this.
  all_skips = set()
  for p in premise_relations:
    skip = getattr(p, 'skip', None)
    if skip and skip not in all_skips:
      all_skips.add(skip)
      optimized_order_premise_relations.append(skip)
    optimized_order_premise_relations.append(p)

  return optimized_order_premise_relations, relevant_state_candidates


def _print_match(m, s):
  s = []
  for x, y in m:
    if x.name in s:
      s += ['{}::{}'.format(x.name, y.name)]
  print('  '.join(s))


class Timeout(Exception):
  pass


def recursively_match(
    query_relations,
    state_candidates,
    object_mappings,
    distinct=None,
    timeout=None,
    match_all=False,):  

  # profiling.enable()
  # query_relations_copy = list(query_relations)
  # state_candidates_copy = {k: list(v) for k, v in 
  #                          state_candidates.items()}
  # object_mappings_copy = {k: v for k, v in object_mappings.items()}
  # distinct_copy = list(distinct)

  # average time: 23,967,474e-8
  # with Timer('match'):
  matches = cython_graph_match.recursively_match(
      query_relations,
      state_candidates,
      object_mappings,
      distinct=distinct or [],
      return_all=match_all)


  # average time: 98,308,578e-8
  # matches = recursively_match_slow(
  #   query_relations,
  #   state_candidates,
  #   object_mappings,
  #   distinct=distinct or [],
  #   timeout=None)
  # with Timer('match'):
  #   matches = list(matches)

  # with Timer('match'):
  #   matches = parallel_graph_match.recursively_match(
  #       query_relations,
  #       state_candidates,
  #       object_mappings,
  #       distinct=distinct or [],
  #       parallel_depth=1,
  #       return_all=match_all)

  # print(len(matches), len(query_relations), [len(x) for x in matches])
  # m = matches[0]
  # keys = [(k.name, k) for k in m.keys()]
  # for _, k in sorted(keys):
  #   v = m[k]
  #   print(k.name, v.name)
  # profiling.print_records()

  # exit()
  return matches


def get_name(obj, premise_match, sign):
  sign_name = ''
  if sign:
    sign_name = '+' if sign > 0 else '-'

  name = None
  if hasattr(obj, 'name_def'):
    if isinstance(obj.name_def, list):
      if any([x not in premise_match for x in obj.name_def]):
        return None
    else:
      if obj.name_def not in premise_match:
        return None

    if isinstance(obj, HalfPlane):
      p1, p2, p = obj.name_def
      name = premise_match[p1].name.lower() + premise_match[p2].name.lower()
      if sign_name:
        name += sign_name
      else:
        name += '*' + premise_match[p].name
    elif isinstance(obj, Line):
      p1, p2 = obj.name_def
      name = premise_match[p1].name.lower() + premise_match[p2].name.lower()
    elif isinstance(obj, LineDirection):
      l = obj.name_def
      name = 'd_' + premise_match[l].name 
    elif isinstance(obj, Segment):
      p1, p2 = obj.name_def
      name = premise_match[p1].name + premise_match[p2].name
    elif isinstance(obj, SegmentLength):
      seg = obj.name_def
      name = 'l_' + premise_match[seg].name
    elif isinstance(obj, Angle):
      p1, p2, p3 = obj.name_def
      name = premise_match[p1].name + premise_match[p2].name + premise_match[p3].name
      if isinstance(obj, AngleXX):
        name += '_xx'
      else:
        name += '_xo'
    elif isinstance(obj, FullAngle):
      d1, d2 = obj.name_def
      name = '<' + premise_match[d1].name + ' ' + premise_match[d2].name + '>'
  return name



def create_new_obj_and_rels_for_conclusion(
      conclusion,
      relations,
      premise_match,
      state_candidates,
      critical,
      conclusion_position,
      val2objs,
      canvas):
  """Given a list of relations in conclusion, create corresponding ones.

  To add to the State.
  """
  new_objs_and_rels = []  # return this
  # Loop through the relation (in the conclusion) and create new objects on demand:
  for rel in relations:
    if isinstance(rel, SamePairSkip):
      continue

    new_init_list = []
    for obj in rel.init_list:
      if isinstance(obj, SelectAngle):
        obj = obj.select(premise_match)
        
      if obj not in premise_match:
        # A new object is needed to be created.
        if obj != geometry.halfpi:

          # Determine sign of hp right away.
          sign = None
          if isinstance(obj, HalfPlane) and hasattr(obj, 'def_points'):
            p1, p2, p = map(premise_match.get, obj.def_points)
            sign = canvas.calculate_hp_sign(p1, p2, p)

          new_obj = type(obj)(name=get_name(obj, premise_match, sign))
          new_objs_and_rels.append(new_obj)
          new_obj.set_critical(critical)
          new_obj.set_conclusion_position(conclusion_position)
          if sign:
            new_obj.sign = sign

        else:
          new_obj = geometry.halfpi

        # Update this new object into the premise_match
        premise_match[obj] = new_obj

      new_init_list.append(premise_match[obj])

    # Now we are ready to create a new rel.
    new_rel = type(rel)(*new_init_list)

    # We also want to add these to state_candidate, so that
    # the next construction iteration will not build the same
    # thing again because of a failed match
    if type(rel) not in state_candidates:
      state_candidates[type(rel)] = []
    state_candidates[type(rel)].append(new_rel)

    new_objs_and_rels.append(new_rel)
    new_rel.set_critical(critical)
    new_rel.set_conclusion_position(conclusion_position)

    premise_match[rel] = new_rel
    # premise_match[new_rel] = rsel

    # Finally we note that if rel is measurement relation, then the value
    # will need to update its dependency graph by adding the clique of 
    # equal nodes presented in the *current* conclusion.
    if isinstance(rel, (SegmentHasLength, AngleHasMeasure, LineHasDirection)):
      _, val = rel.init_list
      # This list is conclusion objects, 
      # will need to be mapped to state objects.
      val2objs[val] = conclusion.val2objs[val]

  return new_objs_and_rels


# def add_new_rels_from_auto_merge(
#       trigger_obj,
#       theorem, 
#       filtered_state_relations, 
#       new_rels, 
#       critical, 
#       conclusion_position,
#       current_state):
#   if theorem is None:
#     return new_rels

#   while True:
#     found = False

#     state_candidates = {}
#     for rel in filtered_state_relations + new_rels:
#       state_candidates[type(rel)] = state_candidates.get(type(rel), []) + [rel]

#     auto_merges = theorem.find_auto_merge_from_trigger(
#         state_candidates, {theorem.trigger_obj: trigger_obj})
#     # import pdb; pdb.set_trace()
#     for (obj1, obj2) in auto_merges:
#       # print('because {}, merging {} and {}'.format(trigger_obj.name, obj1.name, obj2.name))
#       # import pdb; pdb.set_trace()
#       new_rels += create_new_rels_from_merge(
#           obj1, obj2, 
#           filtered_state_relations + new_rels,
#           critical,
#           conclusion_position,
#           current_state)
#       found = False
#     if not found:
#       break
#   return new_rels


def other_obj(rel, obj):
  assert obj in rel.init_list
  return rel.init_list[1-rel.init_list.index(obj)]



def create_new_rels_from_merge(obj1, obj2, 
                               state_relations,
                               critical,
                               conclusion_position,
                               current_state):
  """Algorithm to merge obj1 & obj2.

  Merge (obj1, obj2):
    0. Merge obj2.merge_graph into obj1.merge_graph.
    1. Copy all edges with obj2 to obj1, except for transitive
      value edges, use merge_transitive_value(rel1, rel2) instead.
    2. 
      if isinstance(obj1, Point):
        while SameSegment.match(state_relations, {point=obj1}):
          Merge(seg1, seg2)  # recursive call.
      if isinstance(obj1, Line):
        while SameHalfPlane.match(state_relations, {line=obj1}):
          Merge(hp1, hp2)  # recursive call.
      if isinstance(obj1, Halfplane):
        while SameAngle.match(state_relations, {hp=obj1}):
          Merge(angle1, angle2)  # recursive call.
    3. Add info that obj1 is alias of obj2.
  """
  if not isinstance(obj1, type(obj2)):
    raise ValueError('Cannot merge {} ({}) and {} ({})'.format(
        obj1, type(obj1), obj2, type(obj2)))

  # if not isinstance(obj1, (Point, Segment, Line, Angle, Circle,
  #                          SegmentLength, AngleMeasure, LineDirection)):
  #   raise ValueError('Cannot merge {} and {} of type {}'.format(
  #       obj1, obj2, type(obj1)))

  obj_type = type(obj1)

  # Now we add new relations to new_rels
  # by looping through state_relations and 
  # copy any relation involving obj2 for obj1
  # then delete obj2.

  # Step 0. Merge obj2.merge_graph into obj1.merge_graph
  merge_graph1 = obj1.get_merge_graph(current_state, {
      other_obj(rel, obj1): {obj1: rel} 
      for rel in state_relations 
      if not isinstance(rel, Merge) and obj1 in rel.init_list})

  merge_graph2 = obj2.get_merge_graph(current_state, {
      other_obj(rel, obj2): {obj2: rel} 
      for rel in state_relations 
      if not isinstance(rel, Merge) and obj2 in rel.init_list})

  assert obj2 not in merge_graph1
  assert obj1 not in merge_graph2

  # Copy merge_graph1 to merge_graph before adding info from merge_graph2
  merge_graph = {k: dict(v) for k, v in merge_graph1.items()}
  merge_graph[obj2] = {obj1: None}
  merge_graph[obj1] = merge_graph.get(obj1, {})
  merge_graph[obj1].update({obj2: None})

  merge_graph['equivalents'] = list(merge_graph1['equivalents'])
  merge_graph['equivalents'] += list(merge_graph2['equivalents'])
  merge_graph['equivalents'] += [obj2]

  # Copy info from merge_graph2
  for obj_a, obj_b_dict in merge_graph2.items():
    if obj_a == 'equivalents':
      continue
    if isinstance(obj_a, type(obj1)):
      obj3 = obj_a
      if obj3 in merge_graph:
        assert isinstance(merge_graph[obj3][obj1], Distinct)
        merge_graph[obj3].update(dict(obj_b_dict))
      else:
        merge_graph[obj3] = dict(obj_b_dict)
    else:
      if obj_a in merge_graph:
        for obj_x, rel in obj_b_dict.items():
          assert obj_x not in merge_graph[obj_a]
          merge_graph[obj_a][obj_x] = rel
      else:
        merge_graph[obj_a] = dict(obj_b_dict)

  # Step 1. create new rels
  new_rels = []  # return this
  val_rels = {obj1: None, obj2: None}
  for rel in state_relations:
    if isinstance(rel, Merge):
      continue

    if isinstance(rel, Distinct) and obj1 in rel.init_list and obj2 in rel.init_list:
      raise ValueError('Trying to merge distinct {}s {}'.format(obj_type, rel.name))
    
    # We defer transitive relations to later (see Step 1b.)
    if (isinstance(rel, (SegmentHasLength, LineHasDirection, AngleHasMeasure)) and 
        rel.init_list[0] in [obj1, obj2]):
      val_rels[rel.init_list[0]] = rel
      continue

    if obj2 in rel.init_list and other_obj(rel, obj2) not in merge_graph1:
      # add the rel that invovles obj2, but not obj1 to new_rels
      new_rel = rel.replace(obj2, obj1)
      new_rel.set_critical(critical)
      new_rel.set_conclusion_position(conclusion_position)
      new_rels.append(new_rel)

  if val_rels[obj1] is None and val_rels[obj2] is None:
    pass  # nothing to do.

  elif val_rels[obj1] and val_rels[obj2] is None:
    pass

  elif val_rels[obj1] is None and val_rels[obj2]:
    # If obj1 has no val, but obj2 has val
    rel = val_rels[obj2]
    val = rel.init_list[1]
    # Then after merging obj1 and obj2, obj1 will has the same val
    # with a lot of objs in val.edges, and the reason is that obj1==obj2:
    val.add_new_clique([obj1, obj2])
    new_rel = rel.replace(obj2, obj1)
    new_rel.set_critical(critical)
    new_rel.set_conclusion_position(conclusion_position)
    new_rels.append(new_rel)
  else:  # both has values.
    val1 = val_rels[obj1].init_list[1]
    val2 = val_rels[obj2].init_list[1]

    if val1 is val2:
      val1.add_new_clique([obj1, obj2])
      # this need to be in new rels, for set_chain_pos to reach it:
      new_rels.append(val_rels[obj1])  
    else:
      val2.add_new_clique([obj1, obj2])
      val2.update_edges_tmp(val1.edges[current_state])

      new_rel = val_rels[obj2].replace(obj2, obj1)
      new_rel.set_critical(critical)
      new_rel.set_conclusion_position(conclusion_position)
      new_rels.append(new_rel)
  
  # Now filter obj2 out of state_relations to recursively
  # seek for consequently triggered merges.
  # filtered_state_relations = filter(
  #     lambda x: isinstance(x, Merge) or obj2 not in x.init_list, state_relations)

  # theorem = None
  # if isinstance(obj1, Point):
  #   theorem = all_theorems['auto_seg']
  # elif isinstance(obj1, Line):
  #   theorem = all_theorems['auto_hp']
  # elif isinstance(obj1, HalfPlane):
  #   theorem = all_theorems['auto_angle']
  
  # new_rels = add_new_rels_from_auto_merge(
  #   trigger_obj=obj1,
  #   theorem=theorem, 
  #   filtered_state_relations=filtered_state_relations, 
  #   new_rels=new_rels, 
  #   critical=critical, 
  #   conclusion_position=conclusion_position,
  #   current_state=current_state)
  
  # Finally remove obj2.
  # But first remove obj2 in new_rels
  # new_rels = filter(
  #     lambda rel: isinstance(rel, Merge) or obj2 not in rel.init_list, 
  #     new_rels)

  new_rel = Merge(obj2, obj1, merge_graph)
  new_rel.set_critical(critical)
  new_rel.set_conclusion_position(conclusion_position)

  new_rels.append(new_rel)
  return new_rels


def match_conclusions(conclusion, state_candidates, 
                      premise_match, state_relations, 
                      distinct=None, state=None, canvas=None):
  """Given that premise is matched, see if the conclusion is already there.

  Args:
    conclusion: Conclusion() object
    state_candidates: A dictionary {t: [list of relations with type t]}.
      examples of type t include PointEndsSegment, HalfplaneCoversAngle, 
      SegmentHasLength, etc. See `all classes that inherit from class 
      Relation in `geometry.py` to for a full list. This dictionary stores 
      all edges in the state's graph representation.
    premise_match: A dictionary {premise_object: state_object} that maps 
      from nodes in premise graph to nodes in state graph. This describes
      the successful match between premise and our current graph
    state_relations: A list of all relations currently in graph.
    distinct: A list of pairs (premise_obj1, premise_obj2). Each pair is 
      two nodes in the premise graph. This is to indicate that if premise_obj1
      is matched to state_obj1 and premise_obj2 is matched to state_obj2,
      then state_obj1 must be different to state_obj2, otherwise they can
      be the same. For example, in the case of two equal triangles ABC=DEF
      in the premise that got matched to ADB=ADC in the state, then AB and
      DE can be the same, while the rest must be distinct (and therefore
      presented in this `distinct` list).

  Returns:
    matched_conclusion: A Conclusion() object, the same as the argument
      `conclusion`, except all nodes and edges are replaced with ones
      in the graph, instead of one in the theorem.
    premise_match: updated argument `premise_match` as new edges and nodes
      mapping are added.
  """
  matched_conclusion = Conclusion()
  conclusion_position = 0

  # new value to objects, new ones created in this conclusion
  # will be used to create a clique & add to dependency graph.
  concl_val2objs = ddict(lambda: [])  # val -> conclusion nodes

  # Loop through relation
  for relations, critical in conclusion:
    # For each of the construction step in the conclusion
    # we check if it is already in the state
    total_match = True
    try:
      match = recursively_match(query_relations=relations,
                                state_candidates=state_candidates,
                                object_mappings=premise_match,
                                distinct=distinct)
      if isinstance(match, list):
        match = match[0]
      else:
        match = match.next()
    except:
      total_match = False

    if total_match:  # if yes, move on, nothing to do here.
      premise_match = match
      continue

    # Otherwise, we need to add new objects into the state.
    if isinstance(relations[0], Merge):
      # Case 1: Merging two objects
      assert len(relations) == 1
      obj1, obj2 = relations[0].init_list
      assert obj1 in premise_match and obj2 in premise_match
      new_objs_and_rels = create_new_rels_from_merge(
          premise_match[obj1], 
          premise_match[obj2], 
          state_relations=state_relations,
          critical=critical,
          conclusion_position=conclusion_position,
          current_state=state)

      # Now we remove all relations related to merged_objs
      all_merged_objs = [
          rel.from_obj for rel in new_objs_and_rels 
          if isinstance(rel, Merge)]

      # Filter them out, except TransitiveRelations
      new_objs_and_rels = filter(
          lambda rel: (isinstance(rel, Merge) or
                       isinstance(rel, TransitiveRelation) or
                       rel.init_list[0] not in all_merged_objs and 
                       rel.init_list[1] not in all_merged_objs),
          new_objs_and_rels
      )
    else:
      # Case 2: Create new objects and relations
      new_objs_and_rels = create_new_obj_and_rels_for_conclusion(
          conclusion=conclusion,
          relations=relations,
          premise_match=premise_match,
          state_candidates=state_candidates,
          critical=critical,
          conclusion_position=conclusion_position,
          val2objs=concl_val2objs,
          canvas=canvas
      )

    if critical:
      matched_conclusion.add_critical(*new_objs_and_rels)
    else:
      matched_conclusion.add(*new_objs_and_rels)
    conclusion_position += 1

  # Finally we add dependency cliques:
  for val, objs in concl_val2objs.items():
    # Map both val & objs into state space.
    premise_match[val].add_new_clique(map(premise_match.get, objs))

  return matched_conclusion, premise_match


def match_relations(premise_relations, 
                    state,
                    # reverse_premise=True,
                    conclusion=None,
                    randomize=False,
                    distinct=None,
                    mapping=None,
                    timeout=None,
                    match_all=False,
                    canvas=None):
  """Yield list of matched list of relations in state_relation.
  
  Args:
    premise_relations: A list of objects of type geometry.Relation, each
      represents an edge in the action's premise graph.
    state_relations: A list of objects of type geometry.Relation, each
      represents an edge in the state graph
    augmented_relations: A list of objects of type geometry.Relation, each
      represents an edge to be added in to the state graph. This is needed
      for example, when adding the self-equality relation AD = AD into the
      graph, so that the premise for triangle ADB = ADC is completely 
      presented in the premise.
    conclusion: An object of type Conclusion
  """
  state_relations = list(state.relations + state.distinct_relations())

  if conclusion:
    conclusion_relations = sum(conclusion.topological_list, [])
  else:
    conclusion_relations = []

  # Rearrage relations to optimize recursion branching
  with Timer('action/prepare'):
    sorted_premise_relations, state_candidates = strip_match_relations(
        premise_relations, conclusion_relations, state_relations)

    if randomize:
      for rel_type in state_candidates:
        np.random.shuffle(state_candidates[rel_type])

  with Timer('action/premise_match'):
    premise_matches = recursively_match(
        query_relations=sorted_premise_relations,
        state_candidates=state_candidates,
        object_mappings=mapping or {},
        distinct=distinct,
        timeout=timeout,
        match_all=match_all)

  for premise_match in premise_matches:
    if not conclusion:
      yield premise_match
      continue

    # Copy state_candidates over
    # So that different premise matches won't tamper with each other.
    # Because each of them will put in a different set of new objects 
    # into state_candidates.
    state_candidates_copy = {x: list(y) for x, y in state_candidates.items()}
    with Timer('action/conclusion_match'):
      matched_conclusion, all_match = match_conclusions(
          conclusion=conclusion, 
          state_candidates=state_candidates_copy, 
          premise_match=premise_match, 
          state_relations=state_relations,
          # Distinct is needed to avoid rematching the same premise
          # by rotating the match.
          distinct=distinct,
          state=state,  # needed for merging merge_graphs
          canvas=canvas  # needed for knowing the sign of new hps right away.
      )
    if any(matched_conclusion.critical):
    # if matched_conclusion.topological_list:
      yield matched_conclusion, all_match
