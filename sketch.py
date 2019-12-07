from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import theorems
import numpy as np
import math

from collections import OrderedDict as odict
from sympy import Point, Line, Segment, Circle, Ray

import matplotlib
# matplotlib.use('MacOSX')
from matplotlib import pyplot as plt


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
  x = - b
  try:
    y = math.sqrt(b * b - 2 * a * c)
  except:
    import pdb; pdb.set_trace()
  return (x - y)/a, (x + y)/a


def line_circle_intersection(line, circle):
  a, b, c = map(float, line.coefficients)
  r = float(circle.radius)
  center = circle.center
  p, q = map(float, (center.x, center.y))

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
  a, b, c = map(float, l.coefficients)
  x1, y1, x2, y2 = map(float, (A.x, A.y, B.x, B.y))
  dx, dy = x2-x1, y2-y1
  alpha = (-c - a * x1 - b * y1) / (a * dx + b * dy)
  return Point(x1 + alpha * dx, y1 + alpha * dy)


class InvalidLineIntersect(BaseException):
  pass


def line_line_intersection(l1, l2):
  a1, b1, c1 = map(float, l1.coefficients)
  a2, b2, c2 = map(float, l2.coefficients)
  # a1x + b1y + c1 = 0
  # a2x + b2y + c2 = 0
  d = a1 * b2 - a2 * b1
  if d == 0:
    raise InvalidLineIntersect
  x = (c2 * b1 - c1 * b2) / d
  y = (c1 * a2 - c2 * a1) / d
  return Point(x, y)


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


class Canvas(object):

  def __init__(self):
    self.points = odict()  # graph Point -> sym Point *and* vice versa
    # self.points_inverse = odict()
    self.lines = odict()  # graph Line -> sym Line
    self.line2hps = odict()  # graph Line -> ([sym p for p in hp1], 
                                #                [sym p for p in hp2])
    self.circles = odict()  # graph Circle -> sym Circle

    # The following matrices help with fast calculating betweenness.
    self.line_matrix = np.zeros((0, 3))
    # point_matrix[:, i] = point[i].x, point[i].y
    self.point_matrix = np.zeros((3, 0))

  def show(self):
    points = self.points.keys()
    lines = self.lines.keys()

    mult = np.matmul(  # n_line, n_point
        self.line_matrix,  # [n_line, 3]
        self.point_matrix)  # [3, n_point]
    mult = np.abs(mult) < 1e-12


    all_points = self.points.keys()
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

        if p1 == p2:  # vertical line
          all_y = [p.y for p in sym_points]
          p1 = sym_points[np.argmin(all_y)]
          p2 = sym_points[np.argmax(all_y)]

        lp1 = p1 + (p1 - p2) * 0.5
        lp2 = p2

        lx, ly = (lp1.x, lp2.x), (lp1.y, lp2.y)
      else:
        p = sym_points[0]
        x, y = float(p.x), float(p.y)
        a, b, c = map(float, self.lines[line].coefficients)

        slope = 0 if a == 0. else (b/a)
        lx = (x - 2.0, x + 2.0)
        ly = (y + 2.0 * slope, y - 2.0 * slope)

      plt.plot(lx, ly, color='black')
      plt.annotate(line_name, (lp1.x + 0.1, lp1.y + 0.1))

      for name, p in zip(p_names, sym_points):
        plt.scatter(p.x, p.y, color='black')
        plt.annotate(name, (p.x + 0.1, p.y + 0.1))

    file_name = raw_input('Save sketch to file name: ')
    if file_name:
      print('Saving to {}.png'.format(file_name))
      plt.savefig('{}.png'.format(file_name), dpi=1000)
    plt.clf()

  def copy(self):
    new_canvas = Canvas()

    new_canvas.points = _copy(self.points)
    # new_canvas.points_inverse = _copy(self.points_inverse)
    new_canvas.lines = _copy(self.lines)
    new_canvas.line2hps = _copy(self.line2hps)
    new_canvas.circles = _copy(self.circles)

    new_canvas.line_matrix = np.array(self.line_matrix)
    new_canvas.point_matrix = np.array(self.point_matrix)
    return new_canvas

  def update_line(self, line, sym_line):
    if line in self.lines:
      return

    self.lines[line] = sym_line
    a, b, c = map(float, sym_line.coefficients)
    line_vector = [[a, b, c]]
    self.line_matrix = np.concatenate(
        [self.line_matrix, line_vector], 0)

    mult = np.matmul(line_vector, self.point_matrix)[0, :]
    halfplane_neg = []
    halfplane_pos = []

    # for (p_sym, p_node), v in zip(self.points_inverse.items(), mult):
    for point, v in zip(self.points, mult):
      if v > 1e-12:
        halfplane_pos.append(point)
      elif v < -1e-12:
        halfplane_neg.append(point)

    self.line2hps[line] = (halfplane_neg, halfplane_pos)

  def update_point(self, node_point, sym_point):
    if node_point in self.points:
      return 

    self.points[node_point] = sym_point
    # different sympy Points with the same coordinate
    # get the same hash id, so we need to use id here.
    # self.points_inverse[id(sym_point)] = node_point

    x, y = float(sym_point.x), float(sym_point.y)
    point_vector = [[x], [y], [1.]]
    self.point_matrix = np.concatenate(
        [self.point_matrix, point_vector], 1)

    mult = np.matmul(self.line_matrix, point_vector)[:, 0]

    for line, v in zip(self.line2hps, mult):
      if v > 1e-12:
        self.line2hps[line][1].append(node_point)
      elif v < -1e-12:
        self.line2hps[line][0].append(node_point)

  def add_intersect_line_line(self, new_point, line1, line2):
    line1, line2 = self.lines[line1], self.lines[line2]
    sym_point = line_line_intersection(line1, line2)
    if sym_point is None:
      return None

    self.update_point(new_point, sym_point)
    return self.line2hps

  def add_intersect_seg_line(self, new_point, line, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    line = self.lines[line]
    self.update_point(new_point, 
                      line_segment_intersection(line, p1, p2))
    return self.line2hps

  def add_midpoint(self, mid, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    self.update_point(mid, p1.midpoint(p2))
    return self.line2hps

  def add_mirrorpoint(self, mirror, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    self.update_point(mirror, p1 + (p2 - p1) * 2.0)
    return self.line2hps

  def add_parallel_line(self, new_line, p, line):
    p = self.points[p]
    line = self.lines[line]
    self.update_line(new_line, line.parallel_line(p))
    return {new_line: self.line2hps[new_line]}

  def add_triangle(self, p1, p2, p3, l12, l23, l31):
    # A standard normal triangle
    b, c, a = Point(0., 0.), Point(3., 0.), Point(0.5, 2.5)

    self.update_point(p1, a)
    self.update_point(p2, b)
    self.update_point(p3, c)
    self.update_line(l12, Line(a, b))
    self.update_line(l23, Line(b, c))
    self.update_line(l31, Line(c, a))
    return self.line2hps

  def add_perp_bisector_line(self, new_line, mid, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    s = Segment(p1, p2)
    self.update_line(new_line, s.perpendicular_bisector())
    self.update_point(mid, s.midpoint)
    return self.line2hps

  def add_line(self, new_line, p1, p2):
    p1, p2 = self.points[p1], self.points[p2]
    self.update_line(new_line, Line(p1, p2))
    return {new_line: self.line2hps[new_line]}

  def add_circumscribe_circle(self, new_circle, p1, p2, p3):
    p1, p2, p3 = map(self.points.get, [p1, p2, p3])
    self.circles[new_circle] = Circle(p1, p2, p3)
    return {}

  def add_point_finish_cord(self, new_point, p1, p2, c):
    p1 = self.points[p1]
    p2 = self.points[p2]
    c = self.circles[c]

    # s = time.time()
    a, b, c, d, x, y = map(float, 
                           [p1.x, p1.y, p2.x, p2.y, c.center.x, c.center.y])
    d1, d2 = c - a, d - b
    alpha = -2 * (d1 * (a-x) + d2 * (b-y)) / (d1 * d1 + d2 * d2)
    sym_new_point = Point(a + alpha * d1, b + alpha * d2)
    # print(' > ', time.time() - s)
    # s = time.time()
    self.update_point(new_point, sym_new_point)
    # print(' >>> ', time.time() - s)
    return self.line2hps

  def add_intersecting_point_line_circle(
      self, new_point, line1, line2, circle, other_side_point):

    line1 = self.lines[line1]
    line2 = self.lines[line2]
    circle = self.circles[circle]
    point = self.points[other_side_point]

    p1, p2 = line_circle_intersection(line1, circle)
    # p1, p2 = line1.intersection(circle)
    if _is_different_side(line2, p1, point):
      self.update_point(new_point, p1)
    else:
      self.update_point(new_point, p2)
    return self.line2hps

  def add_intersecting_point_circle_circle(
      self, new_point, line, circle1, circle2, other_side_point):
    line = self.lines[line]
    circle1 = self.circles[circle1]
    circle2 = self.circles[circle2]
    point = self.points[other_side_point]

    p1, p2  = circle_intersection(circle1, circle2)
    # p1, p2 = circle1.intersection(circle2)
    if _is_different_side(line, p1, point):
      self.update_point(new_point, p1)
    else:
      self.update_point(new_point, p2)
    return self.line2hps

  def add_twice_intersecting_circle(
      self, new_c, new_p1, new_p2, new_line, center, p1, p2):
    center = self.points[center]
    p1 = self.points[p1]
    p2 = self.points[p2]

    d1 = center.distance(p1)
    d2 = center.distance(p2)
    d = min(d1, d2)
    radius = d/3.0

    c = Circle(center, radius)
    self.circles[new_c] = c

    # print(c.center, c.radius, p1.x, p1.y)
    sym_newp1 = circle_segment_intersect(c, p1)
    sym_newp2 = circle_segment_intersect(c, p2)
    self.update_point(new_p1, sym_newp1)
    self.update_point(new_p2, sym_newp2)
    self.update_line(new_line, Line(sym_newp1, sym_newp2))
    return self.line2hps


def _is_different_side(line, point1, point2):
  a, b, c = map(float, line.coefficients)
  p1, p2 = point1.evalf(), point2.evalf()
  d1 = p1.x * a + p1.y * b + c
  d2 = p2.x * a + p2.y * b + c

  if d1 * d2 < 0:
    return True
  return False



