
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
  if timeout and time.time() > timeout[0]:
    raise Timeout

  if not query_relations:
    yield object_mappings
    return

  # At this recursion level we try to match premise_rel0
  query0 = query_relations[0]
  rel_type = type(query0)

  # Enumerate through possible relation match:
  for i, candidate in enumerate(state_candidates.get(rel_type, [])):
    # All mappings with prefix object_mappings
    # and additional mapping from the match (premise_rel0 & candidate)
    (a, b), (c, d) = query0.init_list, candidate.init_list
    # Now we want to match a->c, b->d without any conflict
    # if there is conflict then candidate cannot match.
    if (object_mappings.get(a, c) != c or
        object_mappings.get(c, a) != a or
        object_mappings.get(b, d) != d or
        object_mappings.get(d, b) != b):
      continue

    new_mappings = {a: c, b: d}
    # Special treatment
    if not distinct:  # everything is distinct except numeric values.
      if not isinstance(a, (SegmentLength, AngleMeasure, LineDirection)):
        new_mappings[c] = a
      if not isinstance(b, (SegmentLength, AngleMeasure, LineDirection)):
        new_mappings[d] = b
    else:
      # check if new_mappings is going to conflict with object_mappings
      conflict = False
      for distinct_pair in distinct:
        if a not in distinct_pair and b not in distinct_pair:
          continue
        x, y = distinct_pair
        x_map = object_mappings.get(x, new_mappings.get(x, None))
        y_map = object_mappings.get(y, new_mappings.get(y, None))
        # either x or y will be in new_mappings by the above "if",
        # so x_map and y_map cannot be both None
        if x_map == y_map:
          conflict = True
          break

      if conflict:
        continue

    new_mappings[query0] = candidate
    appended_mappings = dict(object_mappings, **new_mappings)
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
    premise_match[val].add(map(premise_match.get, objs))

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
  """Return list of matched list of relations in state_relation."""

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
