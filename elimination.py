"""
Example:

Step 1. Create Isosceles Triangle ABC where angle B = angle C
  Canvas register:
    free: [d_ab, d_bc]
    dependent: [d_ca]
    equal angles: 
        d_ab - d_bc = pi - (d_ca - d_bc)
    Angle def: angleA, angleB, angleC
    
Step 2. Create bisector AD.
  Canvas register:
    free: []
    dependent: d_ad
    equal angles: 
        d_ad - d_ab = d_ca - d_ad
    Angle def:  angleD
    Reduce: angleD = pi/2
"""
from geometry import LineDirection
import numpy as np
import time

from collections import defaultdict as ddict


def rref(matrix):
  A = np.array(matrix, dtype=np.float64)
  m, n = A.shape
  h, k = 0, 0

  while h < m and k < n:
    i_max = np.argmax(np.abs(A[:, k]))
    if A[i_max, k] == 0:
      k += 1  # nothing to do
      continue
    # Else, swap h and i_max
    A[h, :], A[i_max, :] = A[i_max, :], A[h, :]
    A[h, :] /= A[h, k]
    # For each of the other rows:
    A[h+1:, k:] -= A[h:h+1, k:] * A[h+1:, k:k+1]/ A[h, k]
    h += 1
    k += 1

  return A, None


def parse(s):
  r = ['']
  for c in s:
    if c in '+-':
      r.append(c)
      r.append('')
    else:
      r[-1] += c
  return [c for c in r if c]


def set_v(vname, v, s, i):
  sign = 1.0
  for c in parse(s):
    if c == '+':
      sign = 1
    elif c == '-':
      sign = -1
    else:
      assert c in vname
      v[i, vname.index(c)] += sign


def to_mat(vars1, vars2, equations):
  vars1 = vars1.split(',')
  vars2 = vars2.split(',')

  v1 = np.zeros((len(equations), len(vars1)), dtype=np.int32)
  v2 = np.zeros((len(equations), len(vars2)), dtype=np.int32)

  for i, eq in enumerate(equations):
    l, r = eq.replace(' ', '').split('=')
    set_v(vars1, v1, l, i)
    set_v(vars2, v2, r, i)

  return np.concatenate([v1, v2], -1)


def to_eqs(mat, vars1, vars2):
  vars1 = vars1.split(',')
  vars2 = vars2.split(',')

  v1 = mat[:, :len(vars1)]
  v2 = mat[:, len(vars1):]

  eqs = []
  for i in range(mat.shape[0]):
    l = ''
    for j, x in enumerate(v1[i, :]):
      if x != 0:
        if x == int(x):
          x = int(x)
        if x > 0:
          l += '+' if l != '' else ''
        else:
          l += '-'
        if np.abs(x) != 1:
          l += str(abs(x))
        l += vars1[j]

    r = ''
    for j, x in enumerate(v2[i, :]):
      if x != 0:
        if x == int(x):
          x = int(x)
        if x > 0:
          r += '+' if r != '' else ''
        else:
          r += '-'
        if np.abs(x) != 1:
          r += str(abs(x))
        r += vars2[j]
    
    eqs.append(l + '=' + r)
  
  return eqs


def test_isosceles_angle_bisect():
  # Step 1: list all knowledges in the following order:
  #   A. Equal angles (in terms of differences of directions),
  #   In the order where they are created:
  #     1.  d1 - d4 = pi - (d3 - d4)
  #     2.  d1 - d2 = d2 - d3
  #   B. Definition of angles:
  #     3.  m = pi - (d2 - d4)
  #     4.  n = d4 - d2
  #     5.  x = d1 - d2
  #     6.  y = d2 - d3

  # Step 2: identify dependent variables:
  vars1 = 'd1,d2,x,y,m,n'  # ,p,q,r,s,t,u,v,w,z,e'
  # and free variables (always pi & some d)
  vars2 = 'pi,d4,d3'

  # Step 3: write down equations where
  # LHS = dependent and RHS = free variables
  equations = [
      # Equal angles
      'd1 = pi + d4 + d4 - d3',
      'd1 - d2 - d2 = - d3',
      # Angle definition, in the order declared in vars1.
      'x - d1 + d2 =', 
      'y + x = pi',
      'm + d2  = pi + d4',
      'n + m  = pi ',
      # 'p - d2  = - d3',
      # 'q + p = pi',
      # 'r = d3 - d4',
      # 's + r = pi',
      # 't + d1 = d4',
      # 'u + t = pi',
      # 'v -d1 = d3',
      # 'w + v = pi',
  ]

  a = to_mat(vars1, vars2, equations)
  ra, _ = rref(a)
  
  eqs = to_eqs(ra, vars1, vars2)
  eqs = dict([s.split('=') for s in eqs])

  assert eqs['m'] == '0.5pi'
  assert eqs['n'] == '0.5pi'


def test_isosceles_perp():

  vars1 = 'd1,d2,x,y,m,n'
  vars2 = 'pi,d4,d3'

  equations = [
      # Equal angles
      'd1 = pi + d4 + d4 - d3',
      'd2 + d2 = pi + d4 + d4',  # d2 - d4 = 0.5pi
      # Angle definition, in the order declared in vars1.
      'x - d1 + d2 =', 
      'y - d2 = -d3',
      'm + d2 = pi + d4',
      'n + m = pi ',
  ]

  a = to_mat(vars1, vars2, equations)
  ra, _ = rref(a)
  eqs = dict([s.split('=') for s in 
              to_eqs(ra, vars1, vars2)])

  assert eqs['m'] == '0.5pi'
  assert eqs['n'] == '0.5pi'
  assert eqs['x'] == eqs['y']


def add(d1, d2):
  for k, v in d2.items():
    d1[k] += v
    if d1[k] == 0:
      d1.pop(k)
  return d1


def substract(d1, d2):
  for k, v in d2.items():
    d1[k] -= v
    if d1[k] == 0:
      d1.pop(k)
  return d1


class Engine(object):

  def __init__(self, pi):
    self.free = []
    self.deps = {}

    self.pi = pi
    self.add_free(pi)
    self.constants = []

    self.v2a = ddict(lambda: [])
    self.a2v = ddict(lambda: None)

  def supplement_of(self, angle):
    if len(angle) == 2:
      return (self.pi,) + angle
    return angle[1:]

  def is_equal(self, a1, a2):
    return a1 in self.a2v and self.a2v[a1] == self.a2v[a2]

  def merge(self, a1, a2):
    a1_ = self.supplement_of(a1)
    a2_ = self.supplement_of(a2)

    v_a1 = self.a2v[a1]
    v_a1_ = self.a2v[a1_]
    v_a2 = self.a2v[a2]
    v_a2_ = self.a2v[a2_]

    if v_a1 is None and v_a2 is None:
      v_a1 = object()
      v_a1_ = object()
      self.a2v[a1] = v_a1
      self.a2v[a2] = v_a1
      self.v2a[v_a1] += [a1, a2]
      self.a2v[a1_] = v_a1_
      self.a2v[a2_] = v_a1_
      self.v2a[v_a1_] += [a1_, a2_]
    elif v_a1 is None:
      self.a2v[a1] = v_a2
      self.a2v[a1_] = v_a2_
      self.v2a[v_a2] += [a1]
      self.v2a[v_a2_] += [a1_]
    elif v_a2 is None:
      self.a2v[a2] = v_a1
      self.a2v[a2_] = v_a1_
      self.v2a[v_a1] += [a2]
      self.v2a[v_a1_] += [a2_]
    else:
      if v_a1 != v_a2:
        for x in self.v2a[v_a2]:
          self.a2v[x] = v_a1
        self.v2a[v_a1] += self.v2a[v_a2]
        self.v2a.pop(v_a2)
      if v_a1_ != v_a2_:
        for x in self.v2a[v_a2_]:
          self.a2v[x] = v_a1_
        self.v2a[v_a1_] += self.v2a[v_a2_]
        self.v2a.pop(v_a2_)

  
  def add_free(self, *vars):
    for v in list(vars):
      self.deps[v] = {v: 1.0}

      for v0 in self.free:
        if v0.line.greater_than(v.line):
          a, b = v0, v
        else:
          a, b = v, v0

        self.calculate(a, b)
        self.calculate_sup(a, b)

      self.free.append(v)
      
  def add_eq(self, a, b, x, y):
    # a - b = x - y = > a - b - x + y = 0
    if self.is_equal((a, b), (x, y)):
      return []

    result = ddict(lambda: 0.)

    dep, div = None, 0.
    for t, s in zip([a, b, x, y], [1., -1., -1., 1.]):
      if t in self.deps:
        result = add(result, {k: v * -s for k, v in self.deps[t].items()})
      else:
        dep = t
        div += s 

    if dep:
      self.deps[dep] = {k: v / div for k, v in result.items()}
      self.merge((a, b), (x, y))
      return self.auto(dep)
    return []

  def add_sup(self, a, b, x, y):
    # a - b = pi - (x - y) => a - b + x - y = pi
    if self.is_equal((a, b), (self.pi, x, y)):
      return []

    result = ddict(lambda: 0)
    result[self.pi] = 1.0

    dep, div = None, 0.
    for t, s in zip([a, b, x, y], [1., -1., 1., -1.]):
      if t in self.deps:
        result = add(result, {k: v * -s for k, v in self.deps[t].items()})
      else:
        dep = t
        div += s 

    if dep:
      self.deps[dep] = {k: v / div for k, v in result.items()}
      self.merge((a, b), (self.pi, x, y))
      return self.auto(dep)
    return []

  def name(self, x):
    if len(x) == 2:
      d1, d2 = x
      return d1.name + '-' + d2.name
    else:
      p, d1, d2 = x
      return p.name + '+' + d2.name + '-' + d1.name

  def auto(self, v):
    # find vars that is not v:
    vars = [v0 for v0 in self.deps 
            if v0 not in [v, self.pi] and
            not isinstance(v0, tuple)]

    val2angs = ddict(lambda: [])
    for v0 in vars:
      if v0.line.greater_than(v.line):
        a, b = v0, v
      else:
        a, b = v, v0

      val2angs[self.calculate(a, b)].append((a, b))
      val2angs[self.calculate_sup(a, b)].append((self.pi, a, b))
    
    connected = set()
    new_eqs = []  
    for val, angs in val2angs.items():
      if len(angs) <= 1:
        continue

      for i, ang1 in enumerate(angs[:-1]):
        for ang2 in angs[i+1:]:
          
          if self.is_equal(ang1, ang2):
            continue

          val1, val2 = self.a2v[ang1], self.a2v[ang2]
          if (val1, val2) in connected or (val2, val1) in connected:
            continue

          new_val = (val1, val2)
          connected.add(new_val)
          if len(val) == 1 and val[0][0] == self.pi:
            new_val = val[0][1]
          
          new_eqs.append((new_val, ang1, ang2))
          # print(new_val, self.name(ang1), self.name(ang2))

    for _, x1, x2 in new_eqs:
      self.merge(x1, x2)
    return new_eqs
  
  def calculate(self, a, b):
    # m = a - b
    if (a, b) not in self.deps:
      result = ddict(lambda: 0)
      result.update(self.deps[a])
      self.deps[(a, b)] = substract(result, self.deps[b])

    val = self.deps[(a, b)]
    hash = []
    for f in self.free:
      if f in val:
        hash.append((f, val[f]))
    return tuple(hash)

  def calculate_sup(self, a, b):
    if (self.pi, a, b) not in self.deps:
      result = ddict(lambda: 0)
      result[self.pi] = 1.0
      self.deps[(self.pi, a, b)] = substract(
          result, self.deps[(a, b)])

    val = self.deps[(self.pi, a, b)]
    hash = []
    for f in self.free:
      if f in val:
        hash.append((f, val[f]))
    return tuple(hash)



if __name__ == '__main__':
  s = time.time()
  test_isosceles_angle_bisect()
  test_isosceles_perp()
  print('\n\t[OK!] {}s'.format(time.time()-s))