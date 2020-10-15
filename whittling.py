from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection

import traceback


VALUE_ENTITIES = (
    AngleMeasure, SegmentLength, LineDirection
)

VALUE_RELATIONS = (
    AngleHasMeasure, SegmentHasLength, LineHasDirection
)


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
    if isinstance(rel, (SegmentHasLength, LineHasDirection, AngleHasMeasure)):
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
    if isinstance(query, tuple):
      val, rel1, rel2 = query  # what are the dependent of this 3-tuple?
      
      # obj1 and obj2 are the first two dependents
      obj1, obj2 = rel1.init_list[0], rel2.init_list[0]

      # Then there are also others that connects why obj1 == obj2
      # through transitivity
      dependents = val.dependency_path(obj1, obj2, final_state) + [obj1, obj2]
      # dependents are now [int, int, int, ..., obj1, obj2]

      # Make sure such a path of transitivity exists:
      if not all([d is not None for d in dependents]):
        import pdb; pdb.set_trace()
        raise ValueError('Path not found between {} and {} in {}'.format(
            obj1.name, obj2.name,
            {x.name: {a.name: b for a, b in y.items()} for x, y in val.edges.items()}))

      # Add these dependents to the queue.
      queue.extend(dependents)
      continue
    
    # For the remaining cases, there are two information that is needed:
    # 1. The position of `query` in action_chain
    # 2. Whether the construction of this `query` is critical.

    # Case 2: An integer in the queue
    # This means the WHOLE action_chain[pos].topological list is needed.
    if isinstance(query, int):
      critical = True
      pos = query
    else:
      pos = query.chain_position  # at init state already
      if pos is None:
        continue
      critical = query.critical

    # the whole action and its premise is visited.
    try:  # pos = 9, len(whittled_state) = 8
      if (whittled_state and whittled_state[pos] == True 
          or whittled[pos] == True): 
        continue
    except:
      import pdb; pdb.set_trace()

    action = action_chain[pos]
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

      for obj in action.theorem.premise_objects:
        if not isinstance(obj, VALUE_ENTITIES + VALUE_RELATIONS):
          dependents.append(action.mapping[obj])
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

    else:  # Non critical
      try:
        found = action.matched_conclusion.topological_list[
            query.conclusion_position]
      except:
        traceback.print_exc()
        import pdb; pdb.set_trace()
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