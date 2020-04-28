from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import geometry
import theorems_utils
import parallelism

from collections import defaultdict as ddict
from geometry import SegmentLength, AngleMeasure, LineDirection
  

# Then match recursively
def recursively_match(
    query_relations,
    state_candidates,
    object_mappings,
    distinct=None,
    parallel_depth=1,
    return_all=False):
  """Python generator that yields dict({premise_object: state_object})."""
  if not query_relations:
    # There is not any premise edge to match:
    return [object_mappings]

  query0 = query_relations[0]
  # At this recursion level we try to match premise_rel0
  rel_type = type(query0)

  # Enumerate through possible edge match:
  fns, args = [], []
  all_matches = []

  for i, candidate in enumerate(state_candidates.get(rel_type, [])):
    fns.append(recursively_match_branch)
    args.append((query0,
                 candidate, 
                 query_relations,
                 state_candidates,
                 object_mappings, 
                 distinct,
                 parallel_depth,
                 return_all))

  result = []
  if parallel_depth > 0 and len(fns) > 1:
    if return_all:
      return parallelism.parallelize(fns, args)
    else:
      return parallelism.parallelize_return_on_first_finish(fns, args)\

  else:
    for fn, arg in zip(fns, args):
      r = fn(*arg)
      if r is not None:
        if return_all:
          result += r
        else:
          return r
    return result


def recursively_match_branch(query0, 
                             candidate,
                             query_relations,
                             state_candidates,
                             object_mappings, 
                             distinct, 
                             parallel_depth,
                             return_all):
  # Now we try to match edge query0 to candidate, by checking
  # if this match will cause any conflict, if not then we proceed
  # to query1 in the next recursion depth.

  # Suppose edge query0 connects nodes a, b in premise graph
  # and edge candidate connects nodes c, d in state graph:
  (a, b), (c, d) = query0.init_list, candidate.init_list

  # Special treatment for half pi:
  if a == geometry.halfpi and c != a:
    return []
  if b == geometry.halfpi and d != b:
    return []

  # Now we want to match a->c, b->d without any conflict,
  # if there is conflict then candidate cannot be matched to query0.
  if (object_mappings.get(a, c) != c or
      object_mappings.get(b, d) != d or
      # Also check for inverse map if there is any:
      object_mappings.get(c, a) != a or
      object_mappings.get(d, b) != b):
    return []

  new_mappings = {_wrap(a): _wrap(c), _wrap(b): _wrap(d)}
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
      return []

  # Add {query0 -> candidate} to new_mappings
  new_mappings[query0] = candidate
  # Update object_mappings by copying all of its content
  # and then add new_mappings.
  appended_mappings = dict(object_mappings, **new_mappings)

  # Move on to the next recursion depth:
  return recursively_match(
      query_relations=query_relations[1:], 
      state_candidates=state_candidates,
      object_mappings=appended_mappings,
      distinct=distinct,
      parallel_depth=parallel_depth-1,
      return_all=return_all)
