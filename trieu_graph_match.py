
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import geometry
import time

import pyximport; pyximport.install()

import theorems_utils
from theorems_utils import Conclusion
from collections import defaultdict as ddict

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle 
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import PointCentersCircle
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


def strip_match_relations(premise_relations, conclusion_relations, state_relations):
  """Strip unrelated relations in state_relation and sort the relations."""

  # Sort according to number of relations, in order to minimize
  # the branching factor early on when match recursively.
  # This also help with early pruning during recursive calls.
  state_candidates = {}

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
      zip(premise_relations, rel_branch_count), reverse=True))

  return list(premise_relations), state_candidates


def _print_match(m, s):
  s = []
  for x, y in m:
    if x.name in s:
      s += ['{}::{}'.format(x.name, y.name)]
  print('  '.join(s))


class Timeout(Exception):
  pass


# Then match recursively
def recursively_match(query_relations,
                      state_candidates,
                      object_mappings,
                      distinct=None,
                      timeout=None):
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
  in `geometry.py` to for a full list.

  Args:
    query_relations: A list of geometry.Relation objects, each represent
      an edge in the action premise graph.
    state_candidates: A dictionary {t: [list of relations with type t]}.
      examples of type t include PointEndsSegment, HalfplaneCoversAngle, 
      SegmentHasLength, etc. See `all classes that inherit from class 
      Relation in `geometry.py` to for a full list. This dictionary stores 
      all relations in the state's graph representation.
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
  if timeout and time.time() > timeout[0]:
    raise Timeout

  if not query_relations:
    # There is not any premise edge to match:
    yield object_mappings
    return

  query0 = query_relations[0]
  # At this recursion level we try to match premise_rel0
  rel_type = type(query0)

  # Enumerate through possible edge match:
  for i, candidate in enumerate(state_candidates.get(rel_type, [])):
    # Now we try to match edge query0 to candidate, by checking
    # if this match will cause any conflict, if not then we proceed
    # to query1 in the next recursion depth.

    # Suppose edge query0 connects nodes a, b in premise graph
    # and edge candidate connects nodes c, d in state graph:
    (a, b), (c, d) = query0.init_list, candidate.init_list

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

    # Move on to the next recursion depth:
    next_matches = recursively_match(
        query_relations=query_relations[1:], 
        state_candidates=state_candidates,
        object_mappings=appended_mappings,
        distinct=distinct,
        timeout=timeout)

    for match in next_matches:
      yield match


def match_conclusions(conclusion, state_candidates, 
                      premise_match, distinct=None):
  matched_conclusion = Conclusion()
  conclusion_position = 0

  # new value to objects, new ones created in this conclusion,
  # will be used to create dependency path.
  val2objs = ddict(lambda: [])
  for relations, critical in conclusion:
    # For each of the construction step in the conclusion
    # we check if it is already in the premise
    total_match = True
    try:
      match = recursively_match(query_relations=relations,
                                state_candidates=state_candidates,
                                object_mappings=premise_match,
                                distinct=distinct).next()
    except StopIteration:
      total_match = False

    if total_match:  # if yes, move on.
      premise_match = match
      # Collect objects into value buckets
      continue

    # Otherwise, we need to add new objects into the state.
    new_constructions = []
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
            new_constructions.append(new_obj)
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

      new_constructions.append(new_rel)
      new_rel.set_critical(critical)
      new_rel.set_conclusion_position(conclusion_position)

      premise_match[rel] = new_rel
      # premise_match[new_rel] = rsel

      if isinstance(rel, (SegmentHasLength, AngleHasMeasure, LineHasDirection)):
        _, val = rel.init_list
        # Mark that val is involved in a new relation
        val2objs[val] = conclusion.val2objs[val]

    if critical:
      matched_conclusion.add_critical(*new_constructions)
    else:
      matched_conclusion.add(*new_constructions)
    conclusion_position += 1

  # Finally we map val2objs into state space.
  for val, objs in val2objs.items():
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
                    timeout=None):
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
  sorted_premise_relations, state_candidates = strip_match_relations(
      premise_relations, conclusion_relations, 
      state_relations + augmented_relations)

  if randomize:
    for rel_type in state_candidates:
      np.random.shuffle(state_candidates[rel_type])

  premise_matches = recursively_match(
      query_relations=sorted_premise_relations,
      state_candidates=state_candidates,
      object_mappings=mapping or {},
      distinct=distinct,
      timeout=timeout)

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
    state_candidates_ = {x: list(y) for x, y in state_candidates.items()}
    matched_conclusion, all_match = match_conclusions(
        conclusion=conclusion, 
        state_candidates=state_candidates_, 
        premise_match=premise_match, 
        # Distinct is needed to avoid rematching the same premise
        # by rotating the match.
        distinct=distinct,
    )
    if matched_conclusion.topological_list:
      yield matched_conclusion, all_match
