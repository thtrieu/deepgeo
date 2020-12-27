from geometry import CausalValue, Relation, SegmentLength, AngleMeasure, LineDirection, TransitiveRelation
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import Merge
import geometry
import traceback

from collections import defaultdict as ddict


VALUE_ENTITIES = (
    AngleMeasure, SegmentLength, LineDirection
)

VALUE_RELATIONS = (
    AngleHasMeasure, SegmentHasLength, LineHasDirection
)


def extract_all_valuing_goals(final_state, prev_state):
  """Yields pairs (problem_queue, proof_queue)."""
  prev_equal_objs = {}
  for _, valrels in prev_state.val2valrel.items():
    objs = [rel.init_list[0] for rel in valrels]
    for obj in objs:
      prev_equal_objs[obj] = objs

  for val, rels in final_state.val2valrel.items():
    if len(rels) < 2:
      continue
    for i, rel1 in enumerate(rels[:-1]):
      for rel2 in rels[i+1:]:
        obj1, obj2 = rel1.init_list[0], rel2.init_list[0]

        if obj2 in prev_equal_objs.get(obj1, []) or obj1 in prev_equal_objs.get(obj2, []):
          continue
        
        problem_queue = [obj1, obj2]  # e.g. [a, b] or [a, e]
        proof_queue = [(val, rel1, rel2)]
        yield problem_queue, proof_queue


def new_relations_from_merge(prev_state, merge_rel):
  # What are the new relations?
  # Suppose obj2 were merged into obj1

  from_obj, to_obj = merge_rel.from_obj, merge_rel.to_obj
  from_graph = from_obj.get_merge_graph(prev_state)
  to_graph = to_obj.get_merge_graph(prev_state)

  # import pdb; pdb.set_trace()

  for obj in from_graph:
    if obj in to_graph:
      continue

    if from_obj not in from_graph[obj]:
      continue

    rel = from_graph[obj][from_obj]
    if isinstance(rel, TransitiveRelation):
      continue
      
    if isinstance(rel, int):
      continue

    new_rel = rel.replace(from_obj, to_obj)
    yield new_rel.init_list, [(new_rel, to_obj, to_obj)]

    for to_obj_equiv in to_graph['equivalents']:
      new_rel = rel.replace(from_obj, to_obj_equiv)
      yield new_rel.init_list, [(new_rel, to_obj_equiv, to_obj)]
  
  for obj in to_graph:
    if obj in from_graph:
      continue
    if to_obj not in to_graph[obj]:
      continue
    rel = to_graph[obj][to_obj]
    if isinstance(rel, TransitiveRelation):
      continue
    if isinstance(rel, int):
      continue
    
    new_rel = rel.replace(to_obj, from_obj)
    yield new_rel.init_list, [(new_rel, from_obj, to_obj)]
    
    for from_obj_equiv in from_graph['equivalents']:
      new_rel = rel.replace(to_obj, from_obj_equiv)
      yield new_rel.init_list, [(new_rel, from_obj_equiv, to_obj)]


def extract_all_proof_goals(action_chain, final_state):
  last_action = action_chain[-1]
  prev_state = last_action.state

  # First extract all transitive relations:
  for problem_queue, proof_queue in extract_all_valuing_goals(
      final_state, prev_state):
    yield problem_queue, proof_queue

  for merge_rel in last_action.merges:
    for problem_queue, proof_queue in new_relations_from_merge(
        prev_state, merge_rel):
      yield problem_queue, proof_queue

  # for obj in last_action.new_objects:
  #   if not isinstance(obj, TransitiveRelation) and not isinstance(obj, Merge):
  #     if isinstance(obj, Relation):
  #       rel = obj
  #       yield rel.init_list, [rel]


def get_state_and_proof_objects(last_action, state):
  """Yields pairs (problem_queue, proof_queue).

  Last action essentially add new equalities into state.
  Suppose they are a=b=v1, c=d=v2

  So if the state has (a=e, c=f), we have just observe
  the following 4 proofs:
    * a = b
      => yield (a, b), (v1, a=v1, b=v1)
    * e = b
      => yield (e, b), (v1, e=v1, b=v1)
    * c = d
      => ..
    * f = d
      => ..

  The pair (problem_queue, proof_queue) will be used
  as input to the backtracking algorithm that figures out
  what is the minimal set of relevant actions that lead
  to the corresponding discovery.
  """

  # Collect new value to rel in conclusion:
  new_objs, val2objs = [], {}
  for rel in last_action.new_objects:
    if isinstance(rel, VALUE_RELATIONS):
      obj, val = rel.init_list

      # if val == state.halfpi_val():
      #   continue

      new_objs.append(obj)

      if not val in state.val2valrel:
        val = state.obj2valrel[obj].init_list[1]
      val2objs[val] = state.val2valrel[val]
      
      # Old code, run into error. See debug_001.pkl
      # try:
      #   val2objs[val] = state.val2valrel[val]
      # except:
      #   import pdb; pdb.set_trace()
      #   raise Exception
  
  # At this point:
  # new_objs = [b, d]
  # val2objs = {v1: [a, b, e], v2: [c, d, f]}
  # Loop through values correspond to new objects
  for val, rels in val2objs.items():
    # if there are < 2 objs associated with this val
    # then we move on, nothing to prove here.
    if len(rels) < 2:
      continue

    # Loop through all distinct pair of rels
    for i, rel1 in enumerate(rels[:-1]):
      for rel2 in rels[i+1:]:
        # both objects are not new, move on,
        obj1, obj2 = rel1.init_list[0], rel2.init_list[0]
        if obj1 not in new_objs and obj2 not in new_objs:
          continue  # e.g. obj1 = a, obj2 = e
        # Else yield the state and proof queues
        problem_queue = [obj1, obj2]  # e.g. [a, b] or [a, e]
        proof_queue = (val, rel1, rel2)
        yield problem_queue, proof_queue


def whittle_from(final_state, queue, action_chain, 
                 goal_objects=None, whittled_state=None):
  # Whittled info will be put into here:
  whittled = [[] for _ in range(len(action_chain))]
  # We expect 
  # empty [] if the corresponding action in action_chain is not relevant, 
  # True, if the whole action is needed
  # and a list of constructions, if action is not needed but only
  # part of its conclusion.topological_list.

  # Keep track of the head of the queue 
  # we don't pop things from queue but move this pointer ahead.
  i = 0
  non_critical_count = 0  # count when the whole premise is not needed.

  # The idea is to look at the head of the queue,
  # see what are its dependents, add those to the queue tail, and move on.
  while i < len(queue):
    query = queue[i]
    i += 1  # move on.

    # Case 1: the query is a tuple (val, rel1, rel2)
    if isinstance(query, tuple) and isinstance(query[0], VALUE_ENTITIES):
      val, rel1, rel2 = query  # what are the dependent of this 3-tuple?
      
      # obj1 and obj2 are the first two dependents
      obj1, obj2 = rel1.init_list[0], rel2.init_list[0]

      # Then there are also others that connects why obj1 == obj2
      # through transitivity
      dependents = val.dependency_path(obj1, obj2, final_state) + [obj1, obj2]
      # dependents are now [int, int, int, ..., obj1, obj2]

      # Make sure such a path of transitivity exists:
      # if not all([d is not None for d in dependents]):
      #   import pdb; pdb.set_trace()
      #   raise ValueError('Path not found between {} and {} in {}'.format(
      #       obj1.name, obj2.name,
      #       {x.name: {a.name: b for a, b in y.items()} for x, y in val.edges.items()}))

      # Add these dependents to the queue.
      queue.extend(dependents)
      continue
    
    if isinstance(query, tuple):
      rel, obj, source_obj = query
      # import pdb; pdb.set_trace()
      assert obj in rel.init_list
      dependents = source_obj.find_merge_path(rel, obj, final_state)
      queue.extend(dependents)
      continue

    # For the remaining cases, there are two information that is needed:
    # 1. The position of `query` in action_chain
    # 2. Whether the construction of this `query` is critical.

    # Case 2: An integer in the queue
    # This means the WHOLE action_chain[pos].state.premise_objs is needed.
    if isinstance(query, int):
      critical = True
      pos = query
    else:  # not an integer but an obj or rel.
      if isinstance(query, list):
        import pdb; pdb.set_trace()
      pos = query.chain_position
      if pos is None:    # at init state already
        continue
      critical = query.critical

    # the whole action and its premise is already visited.
    if (whittled_state and whittled_state[pos] == True  # by a prev whittle_from()
        or whittled[pos] == True):   # or by this one.
      continue
    # except:
    #   import pdb; pdb.set_trace()

    action = action_chain[pos]
    state = action.state
    # When the whole action is not needed and there is still
    # critical query, we defer this query to the end
    # This optimizes running time because it maximizes
    # the hit `if whittled[pos] == True` above.
    if not critical and len(queue) - (i-1) > non_critical_count:
      non_critical_count += 1
      queue.append(query)
      continue
    elif critical:
      # The whole action is needed.
      whittled[pos] = True

      # Now we add the whole premise to the dependents:
      dependents = []

      # premise of type VALUE_RELATIONS is post-processed here:
      valrels = {}
      merges = ddict(lambda: [])

      for obj in action.theorem.premise_objects:
        if not isinstance(obj, VALUE_ENTITIES + VALUE_RELATIONS):
          rel = action.mapping[obj]
          # dependents.append(rel)

          if not hasattr(rel, '_init_list'):
            if rel not in dependents:
              dependents.append(rel)
            continue

          x, y = rel.init_list

          y_merge_graph = y.get_merge_graph(state)
          if x in y_merge_graph and x not in y_merge_graph['equivalents']:
            merges[y].append(x)

          x_merge_graph = x.get_merge_graph(state)
          if y in x_merge_graph and y not in x_merge_graph['equivalents']:
            merges[x].append(y)

          if y not in merges[x] and x not in merges[y]:
            if rel not in dependents:
              dependents.append(rel)
          
        elif isinstance(obj, VALUE_RELATIONS):
          val = obj.init_list[1]
          if val not in valrels:
            valrels[val] = []
          valrels[val].append(obj)

      # This format (val, rel1, rel2) is for a later call
      # val.dependency_path(rel1, rel2)
      for val, (rel1, rel2) in valrels.items():
        val, rel1, rel2 = map(action.mapping.get, (val, rel1, rel2))
        if rel1 != rel2:
          dependents.append((val, rel1, rel2))
        else:
          dependents.append(rel1)

      merge_positions = set()  # all positions of necessary merge actions
      
      for obj, others in merges.items():
        merge_graph = obj.get_merge_graph(state)
        equivs = set()
        for other in others:
          # TODO(thtrieu): for now we just pick the first equiv.
          equiv = merge_graph[other].keys()[0]

          rel = merge_graph[other][equiv]
          if rel not in dependents:
            dependents.append(rel)

          equivs.add(equiv)
        merge_positions.update(obj.find_min_span_subgraph(equivs, state))
      
      for pos in merge_positions:
        if pos not in dependents:
          dependents.append(pos)

    else:  # Non critical
      # try:
      found = action.matched_conclusion.topological_list[
          query.conclusion_position]
      # except:
      #   traceback.print_exc()
      #   import pdb; pdb.set_trace()
      whittled[pos].append(found)
      # Here we ignore the relations in `found` themselves
      # because we know that they are created at chain_pos = pos
      # there is no need to go further. Only init_list are concerned.
      dependents = sum([c.init_list for c in found
                        if hasattr(c, 'init_list')], tuple())
      non_critical_count -= 1

    # Push dependents into queue.
    for dep in dependents:
      if dep not in queue:
        queue.append(dep)
      if hasattr(dep, 'init_list'):
        a, b = dep.init_list
        if a not in queue:
          queue.append(a)
        if b not in queue:
          queue.append(b)

  return whittled