"""Cython dict or np array?.

n = 500

cdef np.ndarray[int, ndim=1, mode='c'] a = numpy.arange(n, dtype=numpy.int32)
cdef list access = range(n)
cdef int i

numpy.random.shuffle(access)
t = time.time()
for i in access:
  _ = a[i]
print(time.time()-t)

cdef dict d = {object():object() for i in range(n)}
access = list(d.keys())
numpy.random.shuffle(access)

cdef object o
t = time.time()
for o in access:
  _ = d[o]
print(time.time()-t)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import geometry
import numpy
import time
from cython.parallel import prange


cimport numpy as np
cimport cython

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle 
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane
from geometry import PointCentersCircle
from geometry import Merge
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list recursively_match(
    list query_relations,
    dict state_candidates,
    dict object_mappings,
    list distinct=[],
    int return_all=0,
    int depth=0):  # ,
    # list counter=[0]):
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

  if not query_relations:
    # There is not any premise edge to match:
    return [object_mappings]

  query0 = query_relations[0]
  a, b = query0.init_list

  # At this recursion level we try to match premise_rel0
  rel_type = type(query0)

  cdef list all_matches = []
  cdef dict new_mappings
  cdef int conflict
  cdef dict appended_mappings
  cdef list match

  # for i in prange(100, nogil=True):
  #   numpy.random.rand(50, 50).dot(numpy.random.rand(50, 50))

  # Enumerate through possible edge match:
  for candidate in state_candidates.get(rel_type, []):
    # counter[0] += 1
    # Now we try to match edge query0 to candidate, by checking
    # if this match will cause any conflict, if not then we proceed
    # to query1 in the next recursion depth.

    # Suppose edge query0 connects nodes a, b in premise graph
    # and edge candidate connects nodes c, d in state graph:
    c, d = candidate.init_list

    # Special treatment for half pi:
    if a == geometry.halfpi and c != a:
      continue
    if b == geometry.halfpi and d != b:
      continue

    # Now we want to match a->c, b->d without any conflict,
    # if there is conflict then candidate cannot be matched to query0.
    # if (object_mappings.get(a, c) != c or
    #     object_mappings.get(b, d) != d or
    #     # Also check for inverse map if there is any:
    #     object_mappings.get(c, a) != a or
    #     object_mappings.get(d, b) != b):
    if (a in object_mappings and object_mappings[a] != c or
        b in object_mappings and object_mappings[b] != d or
        c in object_mappings and object_mappings[c] != a or
        d in object_mappings and object_mappings[d] != b):
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
      conflict = 0
      for distinct_pair in distinct:
        if a not in distinct_pair and b not in distinct_pair:
          continue  # nothing to check here

        x, y = distinct_pair[0], distinct_pair[1]
        x_map = object_mappings.get(x, new_mappings.get(x, None))
        y_map = object_mappings.get(y, new_mappings.get(y, None))
        # either x or y will be in new_mappings by the above "if",
        # so x_map and y_map cannot be both None
        if x_map == y_map:
          conflict = 1
          break

      if conflict:
        continue  # move on to the next candidate.

    # Add {query0 -> candidate} to new_mappings
    new_mappings[query0] = candidate
    # Update object_mappings by copying all of its content
    # and then add new_mappings.

    appended_mappings = dict(object_mappings, **new_mappings)

    # Move on to the next recursion depth:
    match = recursively_match(
        query_relations=query_relations[1:], 
        state_candidates=state_candidates,
        object_mappings=appended_mappings,
        distinct=distinct,
        return_all=return_all,
        depth=depth+1)  #,
        # counter=counter)

    if match == []:
      continue

    if not return_all:
      return match
    else:
      all_matches.extend(match)

  # if depth == 0:
  #   print(counter[0])
  return all_matches