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


# edge_types = [
#   SegmentHasLength, AngleHasMeasure, LineHasDirection,
#   PointEndsSegment, HalfplaneCoversAngle, LineBordersHalfplane,
#   PointCentersCircle,
#   Merge,
#   LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint
# ]

# edge_types = {t:i for i, t in enumerate(edge_types)}

# """
# query_relations:
#   list[num_edge, 3]

# dict state_candidates,
#   2d-list[num_edge_type, num_edge, 3]

# dict object_mappings,
#   list node_id
#   list node_id

# list distinct=[],
#   list node_id
#   list node_id

# int return_all=0
# """

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef list recursively_match_np_array(
#     list query_relations,
#     dict state_candidates,
#     dict object_mappings,
#     list distinct,
#     int return_all=0):

#   cdef dict node_to_id = {geometry.halfpi: 0}
#   cdef list all_nodes = [geometry.halfpi]

#   cdef list query_relations_l = []
#   for rel in query_relations:
#     a, b = rel.init_list

#     if a not in node_to_id:
#       node_to_id[a] = len(all_nodes)
#       all_nodes.append(a)

#     if b not in node_to_id:
#       node_to_id[b] = len(all_nodes)
#       all_nodes.append(b)

#     a, b = node_to_id[a], node_to_id[b]
#     qa = int(isinstance(a, (SegmentLength, AngleMeasure, LineDirection)))
#     qb = int(isinstance(b, (SegmentLength, AngleMeasure, LineDirection)))

#     query_relations_l.append([a, b, qa, qb, edge_types[type(rel)]])

#   query_relations_np = numpy.array(
#       query_relations_l, dtype=numpy.int32)

#   cdef list state_candidates_l = []
#   for i in range(len(edge_types)):
#     state_candidates_l.append([])

#   for rel_type, rels in state_candidates.items():
#     i = edge_types[rel_type]
#     for rel in rels:
#       c, d = rel.init_list

#       if c not in node_to_id:
#         node_to_id[c] = len(all_nodes)
#         all_nodes.append(c)

#       if d not in node_to_id:
#         node_to_id[d] = len(all_nodes)
#         all_nodes.append(d)

#       c, d = node_to_id[c], node_to_id[d]
#       state_candidates_l[i].append([c, d])

#   max_len = max([len(x) for _, x in 
#                   state_candidates.items()])
#   for edges in state_candidates_l:
#     l = len(edges)
#     if l < max_len:
#       edges += [[-1, -1]] * (max_len - l)

#   state_candidates_np = numpy.array(
#       state_candidates_l, dtype=numpy.int32)

#   cdef list distinct_l = []
#   for x, y in distinct:
#     if x not in node_to_id:
#       node_to_id[x] = len(all_nodes)
#       all_nodes.append(x)

#     if y not in node_to_id:
#       node_to_id[y] = len(all_nodes)
#       all_nodes.append(y)

#     x, y = node_to_id[x], node_to_id[y]
#     distinct_l.append([x, y])
#   distinct_np = numpy.array(distinct_l, dtype=numpy.int32)

#   num_nodes = len(all_nodes)
#   all_nodes += query_relations[::-1]
#   object_mappings_np = numpy.array(
#       [-1] * len(all_nodes), dtype=numpy.int32)

#   t = time.time()
#   cdef np.ndarray[
#       int, ndim=2, mode='c'] all_matches_np = recursively_match_np_array_cython(
#           query_relations_np,
#           state_candidates_np,
#           object_mappings_np,
#           distinct_np,
#           depth=0,
#           return_all=int(return_all)
#       )
#   print(time.time()-t)

#   cdef list all_matches = []
#   for i in range(all_matches_np.shape[0]):
#     match = {}

#     for j, v in enumerate(all_matches_np[i, :]):
#       if v == -1:
#         continue
#       map_from = all_nodes[j]

#       if j >= num_nodes:
#         map_to = state_candidates[type(map_from)][v]
#       else:
#         map_to = all_nodes[v]
#       match[map_from] = map_to
#     all_matches.append(match)

#   return all_matches


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[int, ndim=2, mode='c'] recursively_match_np_array_cython( 
#     # query_relations: [num_query, 5]
#     np.ndarray[np.int32_t, ndim=2, mode='c'] query_relations,

#     # state_candidates
#     np.ndarray[np.int32_t, ndim=3, mode='c'] state_candidates,

#     # object_mappings
#     np.ndarray[np.int32_t, ndim=1, mode='c'] object_mappings,

#     # distinct
#     np.ndarray[np.int32_t, ndim=2, mode='c'] distinct,

#     int depth=0,

#     # return_all 
#     int return_all=0):  #,

#     # list counter=[0]):

#   if query_relations.shape[0] == 0:
#     # There is not any premise edge to match:
#     return object_mappings[None, :]

#   # At this recursion level we try to match query_relations[0]
#   # which is an edge with type rel_type 
#   # that connect node a to node b
#   # where qa and qb indicates if a or b is quantity measurement.
#   cdef int a, b, qa, qb, rel_type
#   a, b, qa, qb, rel_type = query_relations[0, :]

#   cdef int conflict

#   cdef np.ndarray[np.int32_t, ndim=2, mode='c'] candidates
#   candidates = state_candidates[rel_type, :, :]

#   cdef int num_candidate = candidates.shape[0]
#   cdef int num_distinct = distinct.shape[0]
#   cdef int num_node = object_mappings.shape[0]

#   cdef np.ndarray[np.int32_t, ndim=1, mode='c'] candidate
#   cdef int c, d  # candidate connect c -> d

#   cdef np.ndarray[
#       np.int32_t, ndim=1, mode='c'] new_mappings = -numpy.ones([num_node], dtype=numpy.int32)
#   cdef np.ndarray[np.int32_t, ndim=1, mode='c'] appended_mappings

#   cdef np.ndarray[
#       np.int32_t, ndim=2, mode='c'] all_matches = -numpy.ones([0, num_node], dtype=numpy.int32)
#   cdef np.ndarray[np.int32_t, ndim=2, mode='c'] match

#   cdef int x, y, x_map, y_map  # to iterate over distinct pair.

#   # Enumerate through possible edge match:
#   for candidate_count in range(num_candidate):
#     # Now we try to match edge query0 to candidate, by checking
#     # if this match will cause any conflict, if not then we proceed
#     # to query1 in the next recursion depth.

#     # Suppose edge query0 connects nodes a, b in premise graph
#     # and edge candidate connects nodes c, d in state graph:
#     c, d = candidates[candidate_count, 0], candidates[candidate_count, 1] 
#     if c == -1 or d == -1:
#       break

#     # counter[0] += 1

#     # Special treatment for half pi:
#     if a == 0 and c != a:
#       continue
#     if b == 0 and d != b:
#       continue

#     # Now we want to match a->c, b->d without any conflict,
#     # if there is conflict then candidate cannot be matched to query0.
#     if (object_mappings[a] != -1 and object_mappings[a] != c or
#         object_mappings[b] != -1 and object_mappings[b] != d or
#         # Also check for inverse map if there is any:
#         object_mappings[c] != -1 and object_mappings[c] != a or
#         object_mappings[d] != -1 and object_mappings[d] != b):
#       continue  # move on to the next candidate.

#     for node_count in range(num_node):
#       new_mappings[node_count] = -1
#     new_mappings[a] = c
#     new_mappings[b] = d

#     # Check for distinctiveness:
#     if num_distinct == 0:  # Everything is distinct except numeric values.
#       # Add the inverse mappings, so that now a <-> c and b <-> d,
#       # so that in the future c cannot be matched with any other node
#       # other than a, and d cannot be matched with any other node other
#       # than b.
#       if qa == 0:
#         new_mappings[c] = a
#       if qb == 0:
#         new_mappings[d] = b
#     else:
#       # Check if new_mappings is going to conflict with object_mappings
#       # A conflict happens if there exist a' -> c in object_mappings and
#       # (a, a') presented in distinct. Likewise, if there exists b' -> d
#       # in object_mappings and (b, b') presented in distinct then
#       # a conflict happens. 
#       conflict = 0
#       for dictinct_count in range(num_distinct):
#         x, y = distinct[dictinct_count, 0], distinct[dictinct_count, 1]
#         if (x != a and y != a and x != b and y != b):
#           continue  # nothing to check here

#         x_map = object_mappings[x]
#         if x_map == -1:
#           x_map = new_mappings[x]

#         y_map = object_mappings[y]
#         if y_map == -1:
#           y_map = new_mappings[y]

#         # either x or y will be in new_mappings by the above "if",
#         # so x_map and y_map cannot be both None
#         if x_map == y_map:
#           conflict = 1
#           break

#       if conflict == 1:
#         continue  # move on to the next candidate.

#     # Add {query0 -> candidate} to new_mappings
#     new_mappings[num_node - depth - 1] = candidate_count

#     # Update object_mappings by copying all of its content
#     # and then add new_mappings.
#     appended_mappings = -numpy.ones([num_node], dtype=numpy.int32)
#     for node_count in range(num_node):
#       node_map = new_mappings[node_count]
#       if node_map == -1:
#         node_map = object_mappings[node_count]
#       appended_mappings[node_count] = node_map

#     # Move on to the next recursion depth:
#     match = recursively_match_np_array_cython(
#         query_relations=query_relations[1:, :], 
#         state_candidates=state_candidates,
#         object_mappings=appended_mappings,
#         distinct=distinct,
#         depth=depth+1,
#         return_all=return_all)  #,
#         # counter=counter)

#     if match.shape[0] == 0:
#       continue

#     if return_all == 0:
#       return match[None, :]
#     else:
#       all_matches = numpy.concatenate(
#           [all_matches, match], axis=0
#       )

#   # if depth == 0:
#   #   print(counter[0])
#   return all_matches


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

  # n = 500

  # cdef np.ndarray[int, ndim=1, mode='c'] a = numpy.arange(n, dtype=numpy.int32)
  # cdef list access = range(n)
  # cdef int i

  # numpy.random.shuffle(access)
  # t = time.time()
  # for i in access:
  #   _ = a[i]
  # print(time.time()-t)

  # cdef dict d = {object():object() for i in range(n)}
  # access = list(d.keys())
  # numpy.random.shuffle(access)

  # cdef object o
  # t = time.time()
  # for o in access:
  #   _ = d[o]
  # print(time.time()-t)
  # exit()

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