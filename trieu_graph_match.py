from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import geometry
import time
import profiling
import theorems_utils

from state import Conclusion
from collections import defaultdict as ddict
from profiling import Timer

import cython_graph_match
# import parallel_graph_match

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle 
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import PointCentersCircle
from geometry import Merge
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


def strip_match_relations(premise_relations, conclusion_relations, state_relations):
  """Strip unrelated relations in state_relation and sort the relations."""

  # Sort according to number of relations, in order to minimize
  # the branching factor early on when match recursively.
  # This also help with early pruning during recursive calls.
  state_candidates = {}

  if premise_relations == []:
    return premise_relations, state_candidates

  for rel in premise_relations + conclusion_relations:
    rel_type = type(rel)
    if rel_type not in state_candidates:
      state_candidates[rel_type] = []

  for state_rel in state_relations:
    if type(state_rel) in state_candidates:
      state_candidates[type(state_rel)].append(state_rel)

  rel_branch_count = []
  for rel in premise_relations:
    branch_count = len(state_candidates[rel_type])
    rel_branch_count.append(branch_count)

  # sort according to branch_count
  premise_relations, _ = zip(*sorted(
      zip(premise_relations, rel_branch_count), 
      reverse=True))  # very important speedup!

  return list(premise_relations), state_candidates


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


def create_new_obj_and_rels_for_conclusion(
      conclusion,
      relations,
      premise_match,
      state_candidates,
      critical,
      conclusion_position,
      val2objs):
  """Given a list of relations in conclusion, create corresponding ones.

  To add to the State.
  """

  new_objs_and_rels = []  # return this
  # Loop through the relation and create new objects on demand:
  for rel in relations:
    new_init_list = []

    for obj in rel.init_list:
      if obj not in premise_match:
        # A new object is needed to be created.
        # For example in EqualAnglesBecauseIntersectingCords,
        # 2 new lines are created, although they are not critical.
        # (i.e. their creation doesnt need the full premise.)
        if obj != geometry.halfpi:
          new_obj = type(obj)()  # automatic naming
          new_objs_and_rels.append(new_obj)
          # Critical for example, the new AngleMeasure created
          # for the two equal angles in EqualAnglesBecauseIntersectingCords
          new_obj.set_critical(critical)
          new_obj.set_conclusion_position(conclusion_position)
        else:
          new_obj = geometry.halfpi

        # Update this new object into the premise_match
        premise_match[obj] = new_obj

      new_init_list.append(premise_match[obj])

    # Now we are ready to create a new init list.
    new_rel = type(rel)(*new_init_list)

    # We also want to add these to state_candidate, so that
    # the next construction iteration will not build the same
    # thing again because of a failed match (happens e.g. ASA, SAS)
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


def create_new_rels_from_merge(obj1, obj2, 
                               state_relations,
                               critical,
                               conclusion_position):
  if not isinstance(obj1, type(obj2)):
    raise ValueError('Cannot merge {} ({}) and {} ({})'.format(
        obj1, type(obj1), obj2, type(obj2)))

  if not isinstance(obj1, (Point, Segment, Line, Angle, Circle)):
    raise ValueError('Cannot merge {} and {} of type {}'.format(
        obj1, obj2, type(obj1)))

  new_rels = []  # return this
  # Now we add new relations to new_rels
  # by looping through state_relations and 
  # copy any relation involving obj1 for obj2 & vice versa.

  obj_type = type(obj1)

  for rel in state_relations:
    if obj_type is Line and isinstance(rel, LineBordersHalfplane):
      pass
      

    if obj1 in rel.init_list:
      new_rel = rel.replace(obj1, obj2)
    elif obj2 in rel.init_list:
      new_rel = rel.replace(obj2, obj1)
    else:
      continue

    new_rels.append(new_rel)
    new_rel.set_critical(critical)
    new_rel.set_conclusion_position(conclusion_position)

    if isinstance(new_rel, (SegmentHasLength, AngleHasMeasure, LineHasDirection)):
      val = new_rel.init_list[1]
      val.add_new_clique([obj1, obj2])

  return new_rels


def match_conclusions(conclusion, state_candidates, 
                      premise_match, state_relations, 
                      distinct=None):
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
      new_objs_and_rels = create_new_rels_from_merge(
          obj1, obj2, 
          state_relations=state_relations,
          critical=critical,
          conclusion_position=conclusion_position)
    else:
      # Case 2: Create new objects and relations
      new_objs_and_rels = create_new_obj_and_rels_for_conclusion(
          conclusion=conclusion,
          relations=relations,
          premise_match=premise_match,
          state_candidates=state_candidates,
          critical=critical,
          conclusion_position=conclusion_position,
          val2objs=concl_val2objs
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
                    state_relations,
                    augmented_relations=None,
                    # reverse_premise=True,
                    conclusion=None,
                    randomize=False,
                    distinct=None,
                    mapping=None,
                    timeout=None,
                    match_all=False):
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

  if conclusion:
    conclusion_relations = sum(conclusion.topological_list, [])
  else:
    conclusion_relations = []

  augmented_relations = augmented_relations or []
  # Rearrage relations to optimize recursion branching
  with Timer('action/prepare'):
    sorted_premise_relations, state_candidates = strip_match_relations(
        premise_relations, conclusion_relations, 
        state_relations + augmented_relations)

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

  if augmented_relations:
    # We build state candidates without the augmented relations.
    conclusion_state_candidates = {rel_type: [] for rel_type in state_candidates}
    for relation in state_relations:
      rel_type = type(relation)
      if not rel_type in state_candidates:
        continue 
      conclusion_state_candidates[rel_type].append(relation)
    state_candidates = conclusion_state_candidates

  for premise_match in premise_matches:
    if not conclusion:
      yield premise_match
      continue

    # Copy state_candidates over
    # So that different conclusion matches won't tamper with each other.
    # Because each of them will put in new objects into state_candidates.
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
      )
    if matched_conclusion.topological_list:
      yield matched_conclusion, all_match
    # else:
    #   import pdb; pdb.set_trace()
