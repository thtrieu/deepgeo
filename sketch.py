from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from numpy.testing._private.utils import _assert_no_gc_cycles_context
import theorems
import numpy as np
import math

import geometry
import elimination

from collections import OrderedDict as odict

try:
  import matplotlib
  from matplotlib import pyplot as plt
except:
  pass

ATOM = 1e-12


class Circle(object):

  def __init__(self, center, radius):
    self.center = center
    self.radius = radius


class Point(object):

  def __init__(self, x, y):
    self.x = x
    self.y = y  

  def __add__(self, p): 
    return Point(self.x + p.x, self.y + p.y)

  def __sub__(self, p): 
    return Point(self.x - p.x, self.y - p.y)

  def __mul__(self, f): 
    return Point(self.x * f, self.y * f)

  def __rmul__(self, f):
    return self * f

  def __truediv__(self, f):
    return Point(self.x / f, self.y / f)

  def __floordiv__(self, f):
    div = self / f  # true div
    return Point(int(div.x), int(div.y))

  def midpoint(self, p):
    return Point(0.5*(self.x + p.x), 0.5*(self.y + p.y))
  
  def distance(self, p):
    dx = self.x - p.x
    dy = self.y - p.y
    return np.sqrt(dx*dx + dy*dy)


class Line(object):

  def __init__(self, p1=None, p2=None, coefficients=None):
    if coefficients:
      self.coefficients = coefficients
    else:
      self.coefficients = (p1.y - p2.y,
                           p2.x - p1.x,
                           p1.x * p2.y - p2.x * p1.y)
    a, b, c = self.coefficients
    # Make sure a is always positive (or always negative for that matter)
    if a < 0:
      self.coefficients = (-a, -b, -c)

    if a == 0.:
      # With a being always positive,
      # Assuming a = +epsilon > 0
      # Then b such that ax + by = 0 with y>0 should be negative. 
      if b > 0:
        self.coefficients = (0., -b, -c)

  def parallel_line(self, p):
    a, b, _ = self.coefficients
    return Line(coefficients=(a, b, -a*p.x-b*p.y))

  def perpendicular_line(self, p):
    a, b, _ = self.coefficients
    return Line(p, p + Point(a, b))

  def greater_than(self, other):
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a > y/x
    return b*x > a*y
  
  def equal(self, other):
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a > y/x
    return b*x == a*y
  
  def less_than(self, other):
    a, b, _ = self.coefficients
    x, y, _ = other.coefficients
    # b/a > y/x
    return b*x < a*y

  def intersect(self, line):
    return line_line_intersection(self, line)


class Segment(object):

  def __init__(self, p1=None, p2=None):
    self.p1 = p1
    self.p2 = p2


def circle_intersection(o1, o2):
  c1, c2 = o1.center, o2.center
  dc = (c2 - c1).evalf()
  d = math.sqrt(dc.x * dc.x + dc.y * dc.y)

  r0, r1, d = map(float, [o1.radius, o2.radius, d])
  r02 = r0*r0
  a = (r02 - r1*r1 + d*d) / 2.0 / d
  h = math.sqrt(r02 - a*a)
  p2 = o1.center + dc * a / d

  s1 = Point(p2.x + h * dc.y / d, p2.y - h * dc.x / d)
  s2 = Point(p2.x - h * dc.y / d, p2.y + h * dc.x / d)
  return s1, s2


def solve_quad(a, b, c):
  a = 2 * a

  d = b*b - 2*a*c
  
  if d < 0:
    return None  # the caller should expect this result.

  y = math.sqrt(d)
  return (-b - y)/a, (-b + y)/a


def line_circle_intersection(line, circle):
  a, b, c = line.coefficients
  r = float(circle.radius)
  center = circle.center
  p, q = center.x, center.y

  if b == 0:
    x = -c / a
    x_p = x - p
    x_p2 = x_p * x_p
    y1, y2 = solve_quad(1, -2*q, q*q + x_p2 - r*r)
    return (Point(x, y1), Point(x, y2))

  if a == 0:
    y = -c / b
    y_q = y - q
    y_q2 = y_q * y_q
    x1, x2 = solve_quad(1, -2*p, p*p + y_q2 - r*r)
    return (Point(x1, y), Point(x2, y))

  c_ap = c + a * p
  a2 = a * a
  y1, y2 = solve_quad(
      a2 + b*b,
      2 * (b * c_ap - a2*q),
      c_ap * c_ap + a2 * (q*q - r*r)
  )

  return Point(-(b * y1 + c)/a, y1), Point(-(b * y2 + c)/a, y2)


def circle_segment_intersect(circle, p):
  c = circle.center
  r = float(circle.radius)
  p_c = (p - c).evalf()
  d = math.sqrt(p_c.x*p_c.x + p_c.y*p_c.y)
  return c + p_c * r/d


def line_segment_intersection(l, A, B):
  a, b, c = l.coefficients
  x1, y1, x2, y2 = A.x, A.y, B.x, B.y
  dx, dy = x2-x1, y2-y1
  alpha = (-c - a * x1 - b * y1) / (a * dx + b * dy)
  return Point(x1 + alpha * dx, y1 + alpha * dy)


class InvalidLineIntersect(BaseException):
  pass


def line_line_intersection(l1, l2):
  a1, b1, c1 = l1.coefficients
  a2, b2, c2 = l2.coefficients
  # a1x + b1y + c1 = 0
  # a2x + b2y + c2 = 0
  d = a1 * b2 - a2 * b1
  if d == 0:
    raise InvalidLineIntersect
  return Point((c2 * b1 - c1 * b2) / d, 
               (c1 * a2 - c2 * a1) / d)


def bisector(l1, l2, point, same_sign):
  # if same_sign, the bisector bisects (+, +) and (-, -)
  # else the bisector bisects (+, -) and (-, +)
  a1, b1, _ = l1.coefficients
  a2, b2, _ = l2.coefficients

  d1 = math.sqrt(a1*a1 + b1*b1)
  d2 = math.sqrt(a2*a2 + b2*b2)
  a = b1*d2 + b2*d1
  b = a1*d2 + a2*d1

  # Now the two bisectors are:
  # 1. ax - by = 0 => normal vector: (a, -b)
  n1 = (a, -b)
  # 2. ay + bx = 0 => normal vector: (b, a)
  n2 = (b, a)

  # Now we identify which bisector bisects which angle
  # by checking to see the relative positions of
  # the normal vectors n1 and n2 with normal vectors of l1 and l2
  # (recall that the normal vector always points to the positive side)
  cos11 = a * a1 - b * b1  # n1 dot (a1, b1) 
  cos12 = a * a2 - b * b2  # n1 dot (a2, b2)
  if cos11 * cos12 > 0:
    # normal vec of bisector 1 lies in (+, +) or (-, -)
    # => bisector 1 lies in (+, -) halfplane intersection
    # => bisector 2 lies in (+, +) or (-, -) halfplane intersection
    n = n2 if same_sign else n1
  else:
    # normal vec of bisector 1 lies in (+, -)
    # => bisector 1 lies in (+, +) or (-, -) halfplane intersection
    # => bisector 2 lies in (+, -) or (-, +) halfplane intersection
    n = n1 if same_sign else n2

  a, b = n  # bisector: ax + by + c = 0
  return Line(point, point + Point(-b, a))


def _copy(structure):
  if not isinstance(structure, (list, tuple, dict, odict)):
    return structure
  elif isinstance(structure, list):
    return [_copy(x) for x in structure]
  elif isinstance(structure, tuple):
    return tuple(_copy(x) for x in structure)
  else:
    return odict((_copy(key), _copy(val))
                 for (key, val) in structure.items())


def highlight_segment(ax, p1, p2, color, alpha, mark_segment=False):
  x, y = (p1.x, p2.x), (p1.y, p2.y)
  if not mark_segment:
    ax.plot(x, y, color=color, linewidth=7, alpha=alpha)
  else:
    x, y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
    ax.scatter(x, y, color=color, alpha=1.0, marker='o', s=50)


def highlight_point(ax, p, color, alpha):
  ax.scatter(
      p.x, p.y, color=color, edgecolors='none', 
      s=300, alpha=alpha)


def highlight_angle(ax, hps, lines, color, alpha):
  hp1, hp2 = hps
  l1, l2 = lines
  head = line_line_intersection(l1, l2)

  c_head = Circle(head, 0.5)

  p11, p12 = line_circle_intersection(l1, c_head)
  a, b, c = l2.coefficients
  v = a * float(p11.x) + b * float(p11.y) + c
  if v > 0:
    p11, p12 = p12, p11
  p1 = [p11, p12][hp2]
  # ax.scatter(p1.x, p1.y)

  p21, p22 = line_circle_intersection(l2, c_head)
  a, b, c = l1.coefficients
  v = a * float(p21.x) + b * float(p21.y) + c
  if v > 0:
    p21, p22 = p22, p21
  p2 = [p21, p22][hp1]
  # ax.scatter(p2.x, p2.y)

  d1 = p1 - head
  d2 = p2 - head

  a1 = np.arctan2(float(d1.y), float(d1.x)) 
  a2 = np.arctan2(float(d2.y), float(d2.x))
  a1, a2 = a1 * 180/np.pi, a2 * 180/np.pi
  a1, a2 = a1 % 360, a2 % 360

  if a1 > a2:
    a1, a2 = a2, a1

  if a2 - a1 > 180:
    a1, a2 = a2, a1

  b1, b2 = a1, a2
  if b1 > b2:
    b2 += 360
  d = b2 - b1
  if d >= 90: 
    return

  scale = min(2.0, 90 /d)
  scale = max(scale, 0.4)

  fov = matplotlib.patches.Wedge(
      (float(head.x), float(head.y)), 
      0.3 * scale, 
      a1, a2, 
      color=color, 
      alpha=alpha)
  ax.add_artist(fov)


class PiDirection(object):

  def __init__(self):
    self.name = 'pi'
    self.line = Line(coefficients=(0, 1, 0))


class Canvas(object):

  def __init__(self, angle_engine=None, distance_engine=None):
    self.points = odict()  # graph Point -> sym Point *and* vice versa
    # self.points_inverse = odict()
    self.lines = odict()  # graph Line -> sym Line
    self.line2points = odict()  # graph Line -> ([sym p for p in hp1], 
                                #                [sym p for p in hp2])
    self.line2hps = odict()
    self.circles = odict()  # graph Circle -> sym Circle

    # The following matrices help with fast calculating betweenness.
    self.line_matrix = np.zeros((0, 3))
    # point_matrix[:, i] = point[i].x, point[i].y
    self.point_matrix = np.zeros((3, 0))

    if angle_engine == None:
      self.pi = PiDirection()
      angle_engine = elimination.AngleEngine(self.pi)
    
    if distance_engine is None:
      distance_engine = elimination.DistanceEngine()
    
    self.angle_engine = angle_engine
    self.distance_engine = distance_engine

  def show_now(self):
    fig, ax = plt.subplots()
    self.plt_show(ax, None)
    plt.axis('equal')
    plt.show()

  def plt_show(self, ax, state, obj_hightlights=[], mark_segment=False):
    lines = self.lines.keys()
    all_points = self.points.keys()

    mult = np.matmul(  # n_line, n_point
        self.line_matrix,  # [n_line, 3]
        self.point_matrix)  # [3, n_point]
    mult = np.abs(mult) < ATOM

    line_ends = {}
    for line, are_on_line in zip(self.lines.keys(), mult):
      line_name = line.name
      graph_points = [p for p, is_on_line in 
                      zip(all_points, are_on_line)
                      if is_on_line]

      p_names = [p.name for p in graph_points]
      sym_points = [self.points[p] for p in graph_points]

      # import pdb; pdb.set_trace()
      if len(sym_points) > 1:
        all_x = [p.x for p in sym_points]
        p1 = sym_points[np.argmin(all_x)]
        p2 = sym_points[np.argmax(all_x)]
      else:
        p = sym_points[0]
        try:
          p1, p2 = line_circle_intersection(
              self.lines[line],
              Circle(p, 1.0))
        except:
          import pdb; pdb.set_trace()

      if p1 == p2:  # vertical line
        all_y = [p.y for p in sym_points]
        p1 = sym_points[np.argmin(all_y)]
        p2 = sym_points[np.argmax(all_y)]

      lp1 = p1 + (p1 - p2) * 0.5
      lp2 = p2
      line_ends[line] = lp1, lp2

      lx, ly = (lp1.x, lp2.x), (lp1.y, lp2.y)
      ax.plot(lx, ly, color='black')
      ax.annotate(line_name, (lp1.x + 0.01, lp1.y + 0.01))

      for name, p in zip(p_names, sym_points):
        ax.scatter(p.x, p.y, color='black')
        ax.annotate(name, (p.x + 0.01, p.y + 0.01))

    for obj, color, alpha in obj_hightlights:
      if isinstance(obj, geometry.Point):
        if obj not in self.points:
          import pdb; pdb.set_trace()
        highlight_point(ax, self.points[obj], color, alpha)
      elif isinstance(obj, geometry.Segment):
        p1, p2 = state.ends_of_segment(obj)
        p1, p2 = self.points[p1], self.points[p2]
        highlight_segment(ax, p1, p2, color, alpha, mark_segment)
      elif isinstance(obj, geometry.Line):
        p1, p2 = line_ends[obj]
        highlight_segment(ax, p1, p2, color, alpha)
      elif isinstance(obj, geometry.Angle):
        hps, lines = state.hp_and_line_of_angle(obj)
        lines = map(self.lines.get, lines)
        highlight_angle(ax, hps, lines, color, alpha)
      elif isinstance(obj, geometry.HalfPlane):
        line = state.line_of_hp(obj)
        p1, p2 = line_ends[line]
        highlight_segment(ax, p1, p2, color, alpha)

  def copy(self):
    new_canvas = Canvas(
        self.angle_engine.copy(), 
        self.distance_engine.copy()
    )

    new_canvas.points = _copy(self.points)
    # new_canvas.points_inverse = _copy(self.points_inverse)
    new_canvas.lines = _copy(self.lines)
    new_canvas.line2points = _copy(self.line2points)
    new_canvas.line2hps = _copy(self.line2hps)
    new_canvas.circles = _copy(self.circles)

    new_canvas.line_matrix = np.array(self.line_matrix)
    new_canvas.point_matrix = np.array(self.point_matrix)
    return new_canvas

  def update_hps(self, state_line2hps):
    self.line2hps = {
        l: list(hps) for l, hps in
        state_line2hps.items()
    }

  def update_line(self, line, sym_line):
    if line in self.lines:
      return
    
    line.sym = sym_line
    # sym_line.name = line.name
    self.lines[line] = sym_line
    a, b, c = sym_line.coefficients 
    line.sym_line = sym_line

    line_vector = [[a, b, c]]
    self.line_matrix = np.concatenate(
        [self.line_matrix, line_vector], 0)

    mult = np.matmul(line_vector, self.point_matrix)[0, :]
    halfplane_neg = []
    halfplane_pos = []

    # for (p_sym, p_node), v in zip(self.points_inverse.items(), mult):
    for point, v in zip(self.points, mult):
      if v > ATOM:
        halfplane_pos.append(point)
      elif v < -ATOM:
        halfplane_neg.append(point)

    self.line2points[line] = (halfplane_neg, halfplane_pos)

  def update_point(self, node_point, sym_point):
    if node_point in self.points:
      return 

    node_point.x = sym_point.x
    node_point.y = sym_point.y
    self.points[node_point] = sym_point
    # different sympy Points with the same coordinate
    # get the same hash id, so we need to use id here.
    # self.points_inverse[id(sym_point)] = node_point

    x, y = float(sym_point.x), float(sym_point.y)
    point_vector = [[x], [y], [1.]]
    self.point_matrix = np.concatenate(
        [self.point_matrix, point_vector], 1)

    mult = np.matmul(self.line_matrix, point_vector)[:, 0]

    for line, v in zip(self.line2points, mult):
      if v > ATOM:
        self.line2points[line][1].append(node_point)
      elif v < -ATOM:
        self.line2points[line][0].append(node_point)

  def _get_bound(self, l, hp):
    hp = self.line2hps[l].index(hp)  # 0 if neg, 1 if pos
    l = self.lines[l]
    a, b, c = l.coefficients  # ax + by + c = 0
    if hp == 0:  # or a*p.x + b*p.y + c < 0
      a, b, c = -a, -b, -c
    # Now ax + by + c > ATOM, i.e.:
    # by > (ATOM-c) - ax
    # ax > (ATOM-c) - by

    lower_x, upper_x = -np.inf, np.inf
    lower_y, upper_y = -np.inf, np.inf

    if b == 0:  # ax > ATOM-c
      if a > 0:
        lower_x = (ATOM-c)/a
      else:
        upper_x = (ATOM-c)/a
      return [lower_x, upper_x], [lower_y, upper_y]

    if a == 0:  # by > ATOM-c
      if b > 0:
        lower_y = (ATOM-c)/b
      else:
        upper_y = (ATOM-c)/b
      return [lower_x, upper_x], [lower_y, upper_y]
    
    # Now a != 0 and b != 0
    if a > 0:
      lower_x = -b/a, ATOM-c  # x > -b/a * y + ATOM-c
    else:
      upper_x = -b/a, ATOM-c
    return [lower_x, upper_x], [lower_y, upper_y]

  def _sample_from(self, lower, upper):
    if lower == -np.inf:
      if upper == np.inf:
        y = np.random.rand()*2.0 - 1.0
      else:
        y = upper - 2*np.random.rand()
    else:
      if upper == np.inf:
        y = lower + 2*np.random.rand()
      else:
        y = lower + np.random.rand() * (upper-lower)
    return y

  def _add_free_point_l_and_hps(self, p, l_and_hps):

    lower_xs, upper_xs = [], []
    lower_y, upper_y = -np.inf, np.inf

    for l, hp in l_and_hps:
      [lx, ux], [ly, uy] = self._get_bound(l, hp)
      lower_xs.append(lx)
      upper_xs.append(ux)
      lower_y = max(lower_y, ly)
      upper_y = min(upper_y, uy)
    
    y = self._sample_from(lower_y, upper_y)

    lower_x, upper_x = -np.inf, np.inf
    for lx in lower_xs:
      if isinstance(lx, tuple):
        a, b = lx
        lx = a * y + b
      lower_x = max(lower_x, lx)

    for ux in upper_xs:
      if isinstance(ux, tuple):
        a, b = ux
        ux = a * y + b
      upper_x = min(upper_x, ux)

    x = self._sample_from(lower_x, upper_x)
    self.update_point(p, Point(x, y))
    return self.line2points

  def add_free_point_1hp(self, p, l, hp):
    return self._add_free_point_l_and_hps(p, [(l, hp)])

  def add_free_point_2hp(self, p, l1, hp1, l2, hp2):
    return self._add_free_point_l_and_hps(
        p, [
            (l1, hp1),
            (l2, hp2)])

  def add_free_point_3hp(self, p, l1, hp1, l2, hp2, l3, hp3):
    return self._add_free_point_l_and_hps(
        p, [
            (l1, hp1),
            (l2, hp2),
            (l3, hp3)])

  def add_free_point_4hp(self, p, l1, hp1, l2, hp2, l3, hp3, l4, hp4):
    return self._add_free_point_l_and_hps(
        p, [
            (l1, hp1),
            (l2, hp2),
            (l3, hp3),
            (l4, hp4)])
  
  def add_free_point_on_line(self, point, line):
    a, b, c = self.lines[line].coefficients
    # ax + by = -c
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)

    if a == 0:
      y = -c/b
    elif b == 0:
      x = -c/a
    else:
      y = (-c-a*x)/b

    symp = Point(x, y)
    self.update_point(point, symp)
    return self.line2points

  def add_trapezoid(self, A, B, C, D, ab, bc, cd, da):
    sA = Point(np.random.uniform(-1, 1),
              np.random.uniform(-1, 1))
    sB = Point(np.random.uniform(-1, 1),
              np.random.uniform(-1, 1))
    sC = Point(np.random.uniform(-1, 1),
              np.random.uniform(-1, 1))

    lAB = np.random.uniform(0.5, 1.5)
    lCA = np.random.uniform(0.5, 1.5)
    lCD = np.random.uniform(0.5, 1.5)

    diff = lCD - lAB
    if np.abs(diff) < 0.2:
      sign = -1 if diff < 0 else 1
      lCD += sign * 0.2

    sB = sA + (sB-sA) / sB.distance(sA) * lAB
    sC = sA + (sC-sA) / sC.distance(sA) * lCA
    sD = sC + (sA-sB) / sB.distance(sA) * lCD

    sab = Line(sA, sB)
    sbc = Line(sB, sC)
    scd = Line(sC, sD)
    sda = Line(sD, sA)
    
    self.update_point(A, sA)
    self.update_point(B, sB)
    self.update_point(C, sC)
    self.update_point(D, sD)
    self.update_line(ab, sab)
    self.update_line(bc, sbc)
    self.update_line(cd, scd)
    self.update_line(da, sda)
    return self.line2points
    
  def add_random_angle(self, point, l1, l2):
    ang1 = np.random.uniform(0, 2 * np.pi)
    ang2 = ang1 + .1 * np.pi + np.random.uniform(0, 1.8 * np.pi)
    a1, b1 = np.sin(ang1), np.cos(ang1)
    a2, b2 = np.sin(ang2), np.cos(ang2)
    c1 = np.random.uniform(-1, 1)
    c2 = np.random.uniform(-1, 1)

    syml1 = Line(coefficients=(a1, b1, c1))
    syml2 = Line(coefficients=(a2, b2, c2))
    symp = line_line_intersection(syml1, syml2)
    self.update_point(point, symp)
    self.update_line(l1, syml1)
    self.update_line(l2, syml2)
    return self.line2points

  def add_angle_bisector(self, new_line, point, l1, hp1, l2, hp2):
    # Here we try to figure out the Line that bisects angle
    # created by intersection of hp1 and hp2
    # where hp1 is bordered by l1 and hp2 is bordered by l2

    # Given any two lines l1 and l2, there are two bisectors
    # of the 4 angles created by them. Why?
    # Recall that any line divides the plane into 1 positive & 1 negative side
    # i.e. two halfplanes, one negative (-) and one positive (+).
    # So there are 4 angles created by the intersection of l1 and l2

    # Denote the 4 angles by the pair of halfplanes that define each, 
    # using the sign of the halfplane: (+, +), (+, -), (-, +), (-, -)
    # then one of the bisector bisects (+, +) and (-, -) simultaneously
    # the other bisects (+, -) and (-, +) simultaneously.

    # First we see if hp1 and hp2 is positive or negative
    hp1 = self.line2hps[l1].index(hp1)  # 0 if neg, 1 if pos
    hp2 = self.line2hps[l2].index(hp2)  # 0 if neg, 1 if pos

    # This bool uniquely define the bisector we are looking for:
    same_sign = hp1 == hp2
    # if same_sign, the bisector bisects (+, +) and (-, -)
    # else the bisector bisects (+, -) and (-, +)
    
    l1, l2 = self.lines[l1], self.lines[l2]
    point = self.points[point]
    self.update_line(new_line, bisector(l1, l2, point, same_sign))
    return {new_line: self.line2points[new_line]}

  def remove_line(self, line):
    line_idx = self.lines.keys().index(line)
    self.lines.pop(line)
    self.line2points.pop(line)
    self.line2hps.pop(line)
    self.line_matrix = np.delete(self.line_matrix, line_idx, 0)

  def add_perp_line_from_point_on(self, new_line, point, line):
    point, line = self.points[point], self.lines[line]
    self.update_line(new_line, line.perpendicular_line(point))
    return {new_line: self.line2points[new_line]}

  def add_perp_line_from_point_out(self, new_line, new_point, point, line):
    point, line = self.points[point], self.lines[line]
    perp_line = line.perpendicular_line(point)
    self.update_line(new_line, perp_line)
    self.update_point(new_point, line_line_intersection(line, perp_line))
    return self.line2points

  def add_intersect_line_line(self, new_point, line1, line2):
    line1, line2 = self.lines[line1], self.lines[line2]
    sym_point = line_line_intersection(line1, line2)
    if sym_point is None:
      return None

    self.update_point(new_point, sym_point)
    return self.line2points

  def add_intersect_seg_line(self, new_point, line, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    line = self.lines[line]
    self.update_point(new_point, 
                      line_segment_intersection(line, p1, p2))
    return self.line2points

  def add_midpoint(self, mid, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    self.update_point(mid, p1.midpoint(p2))
    return self.line2points

  def add_mirrorpoint(self, mirror, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    self.update_point(mirror, p1 + (p2 - p1) * 2.0)
    return self.line2points

  def add_parallel_line(self, new_line, p, line):
    p = self.points[p]
    line = self.lines[line]
    self.update_line(new_line, line.parallel_line(p))
    return {new_line: self.line2points[new_line]}

  def add_normal_triangle(self, p1, p2, p3, l12, l23, l31):
    # A standard normal triangle
    a = np.pi/2
    A = Point(0., 1.)
    
    d = np.pi/45.

    b = np.random.uniform(np.pi/2 + d, 3 * np.pi/2)
    Bx = np.cos(b)
    By = np.sin(b)
    B = Point(Bx, By)

    if a+d >= b-d:
      gap = 2*d - (b-a)
      c = np.random.uniform(2*np.pi - gap)
      if c >= a+d:
        c += gap
    else:
      c = np.random.uniform(2*np.pi - 4 * d)
      if (a-d <= c and c <= a+d or
          b-d <= c and c <= b+d):
        c += 2 * d
    
    C = Point(np.cos(c), np.sin(c))

    shift = Point(np.random.uniform(-1, 1),
                  np.random.uniform(-1, 1))
    A = A + shift
    B = B + shift
    C = C + shift

    self.update_point(p1, A)
    self.update_point(p2, B)
    self.update_point(p3, C)
    self.update_line(l12, Line(A, B))
    self.update_line(l23, Line(B, C))
    self.update_line(l31, Line(C, A))
    return self.line2points

  def add_right_triangle(self, p1, p2, p3, l12, l23, l31):
    # A standard normal triangle
    a = np.pi/2
    A = Point(0., 1.)
    b = 3*a
    B = Point(0., -1)
    
    d = np.pi/45.

    c = np.random.uniform(2*np.pi - 4 * d)
    if (a-d <= c and c <= a+d or
        b-d <= c and c <= b+d):
      c += 2 * d
    
    C = Point(np.cos(c), np.sin(c))

    shift = Point(np.random.uniform(-1, 1),
                  np.random.uniform(-1, 1))
    A = A + shift
    B = B + shift
    C = C + shift

    self.update_point(p1, A)
    self.update_point(p2, B)
    self.update_point(p3, C)
    self.update_line(l12, Line(A, B))
    self.update_line(l23, Line(B, C))
    self.update_line(l31, Line(C, A))
    return self.line2points

  def add_isosceles_triangle(self, p1, p2, p3, l12, l23, l31):
    A = Point(0., 1.)
    d = np.pi/45.
    b = np.random.uniform(np.pi/2 + d, 3 * np.pi/2 - d)
    Bx = np.cos(b)
    By = np.sin(b)

    B = Point(Bx, By)
    C = Point(-Bx, By)

    shift = Point(np.random.uniform(-1, 1),
                  np.random.uniform(-1, 1))
    A = A + shift
    B = B + shift
    C = C + shift

    self.update_point(p1, A)
    self.update_point(p2, B)
    self.update_point(p3, C)
    self.update_line(l12, Line(A, B))
    self.update_line(l23, Line(B, C))
    self.update_line(l31, Line(C, A))
    return self.line2points

  def add_equilateral_triangle(self, p1, p2, p3, l12, l23, l31):
    # A standard normal triangle
    A, B, C = Point(0., 1.), Point(3., 0.), Point(0.5, 2.5)

    self.update_point(p1, A)
    self.update_point(p2, B)
    self.update_point(p3, C)
    self.update_line(l12, Line(A, B))
    self.update_line(l23, Line(B, C))
    self.update_line(l31, Line(C, A))
    return self.line2points

  def add_line(self, new_line, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    self.update_line(new_line, Line(p1, p2))
    return {new_line: self.line2points[new_line]}

  def calculate_hp_sign(self, p1, p2, p):
    # calculate the sign of hp that border line L
    # that goes through points lp1, lp2 and the hp contains point p
    p1, p2, p = map(self.points.get, [p1, p2, p])
    a, b, c =  (p1.y - p2.y,
                p2.x - p1.x,
                p1.x * p2.y - p2.x * p1.y)
    if a <= 0.:
      a, b, c = -a, -b, -c
    
    if a * p.x + b * p.y + c > 0:
      return 1
    return -1

  def add_free_directions(self, *vars):
    self.angle_engine.add_free(*vars)

  def eliminate_angle(self, pair1, pair2, same_sign, chain_pos):
    da, db = pair1
    dx, dy = pair2

    if db.line.greater_than(da.line):
      da, db = db, da

    if dy.line.greater_than(dx.line):
      dx, dy = dy, dx

    if same_sign:
      return self.angle_engine.add_eq(da, db, dx, dy, chain_pos)
    else:
      return self.angle_engine.add_sup(da, db, dx, dy, chain_pos)
  
  def add_free_points(self, l, *free_points):
    self.distance_engine.add_free(l, *free_points)

  def add_points(self, l, *points):
    self.distance_engine.add(l, *points)

  def eliminate_distance(self, pair1, pair2, chain_pos):
    pa, pb = pair1
    px, py = pair2

    if elimination.point_greater(pb, pa):
      pa, pb = pb, pa
    if elimination.point_greater(py, px):
      px, py = py, px
    return self.distance_engine.add_eq(pa, pb, px, py, chain_pos)
    

    

  # def add_circumscribe_circle(self, new_circle, p1, p2, p3):
  #   raise NotImplementedError('Not implemented Circle')
  #   p1, p2, p3 = map(self.points.get, [p1, p2, p3])
  #   self.circles[new_circle] = Circle(p1, p2, p3)
  #   return {}

  # def add_point_finish_cord(self, new_point, p1, p2, c):
  #   p1 = self.points[p1]
  #   p2 = self.points[p2]
  #   c = self.circles[c]

  #   # s = time.time()
  #   a, b, c, d, x, y = [p1.x, p1.y, p2.x, p2.y, 
  #                       c.center.x, c.center.y]
  #   d1, d2 = c - a, d - b
  #   alpha = -2 * (d1 * (a-x) + d2 * (b-y)) / (d1 * d1 + d2 * d2)
  #   sym_new_point = Point(a + alpha * d1, b + alpha * d2)
  #   self.update_point(new_point, sym_new_point)
  #   return self.line2points

#   def add_intersecting_point_line_circle(
#       self, new_point, line1, line2, circle, other_side_point):

#     line1 = self.lines[line1]
#     line2 = self.lines[line2]
#     circle = self.circles[circle]
#     point = self.points[other_side_point]

#     p1, p2 = line_circle_intersection(line1, circle)
#     # p1, p2 = line1.intersection(circle)
#     if _is_different_side(line2, p1, point):
#       self.update_point(new_point, p1)
#     else:
#       self.update_point(new_point, p2)
#     return self.line2points

#   def add_intersecting_point_circle_circle(
#       self, new_point, line, circle1, circle2, other_side_point):
#     line = self.lines[line]
#     circle1 = self.circles[circle1]
#     circle2 = self.circles[circle2]
#     point = self.points[other_side_point]

#     p1, p2  = circle_intersection(circle1, circle2)
#     # p1, p2 = circle1.intersection(circle2)
#     if _is_different_side(line, p1, point):
#       self.update_point(new_point, p1)
#     else:
#       self.update_point(new_point, p2)
#     return self.line2points

#   def add_twice_intersecting_circle(
#       self, new_c, new_p1, new_p2, new_line, center, p1, p2):
#     center = self.points[center]
#     p1 = self.points[p1]
#     p2 = self.points[p2]

#     d1 = center.distance(p1)
#     d2 = center.distance(p2)
#     d = min(d1, d2)
#     radius = d/3.0

#     c = Circle(center, radius)
#     self.circles[new_c] = c

#     # print(c.center, c.radius, p1.x, p1.y)
#     sym_newp1 = circle_segment_intersect(c, p1)
#     sym_newp2 = circle_segment_intersect(c, p2)
#     self.update_point(new_p1, sym_newp1)
#     self.update_point(new_p2, sym_newp2)
#     self.update_line(new_line, Line(sym_newp1, sym_newp2))
#     return self.line2points


# def _is_different_side(line, point1, point2):
#   a, b, c = line.coefficients
#   d1 = p1.x * a + p1.y * b + c
#   d2 = p2.x * a + p2.y * b + c
#   return d1 * d2 < 0



