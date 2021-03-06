import sketch
import geometry
import theorems
import debugging
import profiling

from theorems_utils import collinear, concyclic, in_halfplane
from theorems_utils import divides_halfplanes, line_and_halfplanes
from theorems_utils import have_length, have_measure, have_direction
from theorems_utils import segment_def, fangle_def
from theorems_utils import diff_side, same_side
from theorems_utils import distinct

from state import State, Conclusion

from geometry import Point, Line, Segment, Angle, HalfPlane, Circle
from geometry import SegmentLength, AngleMeasure, LineDirection
from geometry import SegmentHasLength, AngleHasMeasure, LineHasDirection
from geometry import PointEndsSegment, LineBordersHalfplane
from geometry import LineContainsPoint, CircleContainsPoint, HalfPlaneContainsPoint


def add_action(state, action, full_state):
  assert False
  for obj in action.conclusion_objects:
    state.add_one(obj)
    if isinstance(obj, Line):
      hp1, hp2 = full_state.line2hps[obj]
      state.add_one(LineBordersHalfplane(obj, hp1))
      state.add_one(LineBordersHalfplane(obj, hp2))

  for hp in state.all_hps:
    if hp not in full_state.hp2points:
      continue
    for p in full_state.hp2points[hp]:
      if p in state.all_points:
        state.add_one(HalfPlaneContainsPoint(hp, p))


def _is_numeric(string):
  try:
    _ = int(string)
    return True
  except:
    return False


def _find(state, name):

  if name in state.name2obj:
    return state.name2obj[name]

  names = [n.split('_') for n in state.name2obj.keys()]
  names = [n for n in names if (
              len(n) == 2 and
              n[0] == name and
              _is_numeric(n[1]) 
           )]
  if len(names) != 1:
    import pdb; pdb.set_trace()
    raise ValueError('Failed looking for {}'.format(name))

  name = '_'.join(names[0])
  return state.name2obj[name]


def _find_premise(premise_objects, name):
  for obj in premise_objects:
    if obj.name.startswith(name):
      return obj
  return None


def mapping_from_command(command, theorem, state):  

  name_maps = [c.split('=') for c in command.split()]

  if name_maps == []:
    return {}

  if '=' not in command:
    return {
        theorem_obj: _find(state, command_obj_name)
        for theorem_obj, command_obj_name
        in zip(theorem.args, command.split())
        if command_obj_name != 'None'
    }

  mapping = dict(
      (theorem.names[a], _find(state, b))
      if a in theorem.names
      else (_find_premise(theorem.premise_objects, a), _find(state, b))
      for a, b in name_maps)
  return mapping


def recursively_auto_merge(action, state, chain_position):
  """A merge could lead to a chain of merges."""
  if not action.merges:
    return

  # for merge_rel in action.merges:
  #   print('Merge {} ==> {}'.format(
  #       merge_rel.from_obj.name, 
  #       merge_rel.to_obj.name))
  
  exhausted = False  # is all auto merges exhausted?
  while not exhausted:
    exhausted = True  # set to False immediately once a match is found.
    for merge_rel, _ in action.merges:
      trigger_obj = merge_rel.to_obj

      for theorem in theorems.auto_merge_theorems_from_trigger_obj(trigger_obj):
        for next_merge_action in theorem.match_from_input_mapping(
            state, {theorem.trigger_obj: trigger_obj}):
          exhausted = False
          next_merge_action.set_chain_position(chain_position)
          state.add_relations(next_merge_action.new_objects)
          recursively_auto_merge(next_merge_action, state, chain_position)
          
          # append action with auto merged's matched_conclusion & new objects
          action.update(next_merge_action)


def extract_theorem_mapping(theorem_command, state):
  theorem, command = None, None
  if isinstance(theorem_command, tuple):
    theorem, command = theorem_command
  elif isinstance(theorem_command, str):
    theorem, command = theorem_command.split(':')
    theorem = theorems.theorem_from_short_name[theorem.strip()]
    command = command.strip()
  
  mapping = mapping_from_command(command, theorem, state)
  return theorem, mapping



def execute_steps(steps, state, canvas, verbose=False, init_action_chain=None):
  init_action_chain = init_action_chain or []
  action_chain = []

  for i, theorem_command in enumerate(steps):
    theorem, mapping = extract_theorem_mapping(theorem_command, state)

    with profiling.Timer(theorem.__class__.__name__):
      action_gen = theorem.match_from_input_mapping(
          state, mapping, randomize=False, canvas=canvas)

      pos = i + len(init_action_chain)

      try:
        action = action_gen.next()
      except StopIteration:
        best, miss = debugging.why_fail_to_match(
            theorem, state, mapping=mapping)
        import pdb; pdb.set_trace()
        # Use state.name_map(best) or state.name_map(miss)
        # to investigate why this matching fail.
        raise ValueError('Matching not found {} {}'.format(
            theorem, theorem_command))

    print('{:2d}. {}'.format(pos, action.to_str()))
    action.set_chain_position(pos)
    action_chain.append(action)

    if verbose:
      print('\tAdd : {}'.format(
          [obj.name for obj in action.new_objects]))
    
    # import pdb; pdb.set_trace()
    # print(state)
    state = state.copy()
    canvas = canvas.copy()
    
    # add conclusion
    state.add_relations(action.new_objects)

    # add spatial relations:
    line2pointgroups = action.draw(canvas)
    state.add_spatial_relations(line2pointgroups, pos)
    canvas.update_hps(state.line2hps)
    
    # add auto equalities
    state, auto_merge_directions = action.action_eliminate_angle(state, canvas, pos)
    state, auto_merge_points = action.action_eliminate_distance(state, canvas, pos)
    
    # add auto merges
    if auto_merge_points + auto_merge_directions:
      # Loop through actions and update state.
      for theorem, mapping, auto_pos in auto_merge_points + auto_merge_directions:
        for merge_action in theorem.match_from_input_mapping(state, mapping, canvas=canvas):
          merge_action.set_chain_position(pos, auto_pos)
          break
        state.add_relations(merge_action.new_objects)
        recursively_auto_merge(merge_action, state, pos)
        action.update(merge_action)
    # else:
    recursively_auto_merge(action, state, pos)

  return state, canvas, init_action_chain+action_chain


def init_by_normal_triangle():
  geometry.reset()
  canvas = sketch.Canvas()
  state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, BC, CA = map(Segment, 'AB BC CA'.split())

  state.add_relations(
      # [A, B, C, ab, bc, ca, AB, BC, CA] +
      segment_def(AB, A, B) +
      segment_def(BC, B, C) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      distinct(A, B, C) +
      distinct(ab, bc, ca)
  )

  state.add_spatial_relations(canvas.add_triangle(A, B, C, ab, bc, ca))
  canvas.update_hps(state.line2hps)
  return state, canvas, [(theorems.all_theorems['right'], '')]


def init_by_isosceles_triangle():
  geometry.reset()
  canvas = sketch.Canvas()
  state = State()

  A, B, C = map(Point, 'ABC')
  ab, bc, ca = map(Line, 'ab bc ca'.split())
  AB, BC, CA = map(Segment, 'AB BC CA'.split())

  state.add_relations(
      # [A, B, C, ab, bc, ca, AB, BC, CA] +
      segment_def(AB, A, B) +
      segment_def(BC, B, C) +
      segment_def(CA, C, A) +
      collinear(ab, A, B) +
      collinear(bc, B, C) +
      collinear(ca, C, A) +
      have_length('1m', AB, CA)
  )

  state.add_spatial_relations(canvas.add_triangle(A, B, C, ab, bc, ca))
  canvas.update_hps(state.line2hps)
  return state, canvas, [(theorems.all_theorems['right'], '')]


def init_by_thales():
  geometry.reset()
  init_state, init_canvas, _ = init_by_normal_triangle()
  state, canvas = init_state.copy(), init_canvas.copy()

  steps = [
      (theorems.all_theorems['mid'], 'A=A B=B'),  # P1
      (theorems.all_theorems['parallel'], 'A=P1 l=bc'),  # l1
      (theorems.all_theorems['seg_line'], 'l=l1 A=A B=C'),  # P1
      (theorems.all_theorems['parallel'], 'A=C l=ab'),  # l2
      (theorems.all_theorems['line'], 'A=P1 B=C'),  # l3
  ]
  return state, canvas, steps


def init_by_debug_001():
  geometry.reset()
  init_state, init_canvas, _ = init_by_normal_triangle()
  state, canvas = init_state.copy(), init_canvas.copy()

  steps = [
      (theorems.all_theorems['parallel'], 'A=B B=ca'),  # l1
      (theorems.all_theorems['bisect'], 'hp1=ab_hp2 hp2=l1_hp1'),  # l2
      (theorems.all_theorems['seg_line'], 'l=l2 A=C B=A'),  # P1
      (theorems.all_theorems['eq'], 'l=l2 l1=l1 l2=ca'),  # l2
      (theorems.all_theorems['mirror'], 'A=P1 B=B'),  # l2
      (theorems.all_theorems['asa'], 'A=P1 B=A C=B D=B F=P1 de=ab ef=ca'),  # bug??
  ]
  return state, canvas, steps