"""
Example:
"""

import profiling
from profiling import Timer

from collections import defaultdict


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


class AngleEngine(object):

  def __init__(self, pi):
    self.free = []
    self.deps = {}
    self.pos = {}  # which chain pos explain the value?
    
    self.eqs = []
    self.const2a = defaultdict(lambda: [])

    self.pi = pi
    self.add_free(pi)

    self.v2a = defaultdict(lambda: [])
    self.a2v = defaultdict(lambda: None)

  def copy(self):
    e = AngleEngine(self.pi)
    e.free = list(self.free)
    e.deps = {x: dict(y) for x, y in self.deps.items()}
    e.pos = {x: set(y) for x, y in self.pos.items()}
    self.eqs = list(self.eqs)

    const2a = defaultdict(lambda: [])
    for c, a in self.const2a.items():
      const2a[c] = list(a)
    e.const2a = const2a
    
    v2a = defaultdict(lambda: [])
    for v, a in self.v2a.items():
      v2a[v] = list(a)
    e.v2a = v2a

    a2v = defaultdict(lambda: None)
    for a, v in self.a2v.items():
      a2v[a] = v
    e.a2v = a2v
    return e

  def is_equal(self, a1, a2):
    return a1 in self.a2v and self.a2v[a1] == self.a2v[a2]

  def add_free(self, *vars):
    for v in list(vars):
      if v in self.deps:
        continue
      self.deps[v] = {v: 1.0}
      self.pos[v] = set()

      for v0 in self.free:
        if v0.line.greater_than(v.line):
          a, b = v0, v
        else:
          a, b = v, v0

        self.calculate(a, b)
        self.calculate_sup(a, b)

      self.free.append(v)
      
  def add_eq(self, a, b, x, y, chain_pos):
    # a - b = x - y = > a - b - x + y = 0
    # if self.is_equal((a, b), (x, y)):
    if {(a, b), (x, y)} in self.eqs:
      return []

    self.eqs.append({(a, b), (x, y)})
    self.eqs.append({(self.pi, a, b), (self.pi, x, y)})

    pos = set([chain_pos])
    result = defaultdict(lambda: 0.)

    dep, div = None, 0.
    for t, s in zip([a, b, x, y], [1., -1., -1., 1.]):
      if t in self.deps:
        result = add(result, {k: v * -s for k, v in self.deps[t].items()})
        pos.update(self.pos[t])
      else:
        dep = t
        div += s 

    if dep:
      self.deps[dep] = {k: v / div for k, v in result.items()}
      self.pos[dep] = pos
      return self.auto(dep)
    return []

  def add_sup(self, a, b, x, y, chain_pos):
    # a - b = pi - (x - y) => a - b + x - y = pi
    # if self.is_equal((a, b), (self.pi, x, y)):
    if {(a, b), (self.pi, x, y)} in self.eqs:
      return []

    self.eqs.append({(a, b), (self.pi, x, y)})
    self.eqs.append({(self.pi, a, b), (x, y)})

    pos = set([chain_pos])
    result = defaultdict(lambda: 0)
    result[self.pi] = 1.0

    dep, div = None, 0.
    for t, s in zip([a, b, x, y], [1., -1., 1., -1.]):
      if t in self.deps:
        result = add(result, {k: v * -s for k, v in self.deps[t].items()})
        pos.update(self.pos[t])
      else:
        dep = t
        div += s 

    if dep:
      self.deps[dep] = {k: v / div for k, v in result.items()}
      self.pos[dep] = pos
      return self.auto(dep)
    return []

  def ang_name(self, x):
    if len(x) == 2:
      d1, d2 = x
      return d1.name + '-' + d2.name
    else:
      p, d1, d2 = x
      return p.name + '+' + d2.name + '-' + d1.name

  def val_name(self, x):
    if isinstance(x, float):
      return str(x) + 'pi'
    name = ''
    for d, v in x:
      if v > 0:
        if name != '':
          name += '+'
      else:
        name += '-'
      if abs(v) != 1:
        name += str(abs(v))
      name += d.name
    return name

  def sup_of(self, ang):
    if len(ang) == 2:
      return (self.pi,) + ang
    return ang[1:]

  def is_const(self, hash):
    return len(hash) ==  1 and hash[0][0] == self.pi

  def auto(self, v):
    # find vars that is not v:
    vars = [v0 for v0 in self.deps 
            if v0 not in [v, self.pi] and
            not isinstance(v0, tuple)]

    new_ang_hash = []
    for v0 in vars:
      if v0.line.greater_than(v.line):
        a, b = v0, v
      else:
        a, b = v, v0
      
      hash = self.calculate(a, b)
      new_ang_hash.append(((a, b), hash))
      hash = self.calculate_sup(a, b)
      new_ang_hash.append(((self.pi, a, b), hash))

    new_eqs = []  
    for ang1, hash in new_ang_hash:
      if self.is_const(hash):
        if ang1 in self.const2a[hash]:
          continue
        self.const2a[hash].append(ang1)
        const = hash[0][1]
        new_eqs.append((const, ang1, self.pos[ang1]))
        # print('{}pi = {}'.format(const, self.ang_name(ang1)))
        continue

      for ang2 in self.v2a[hash]:
        if ang2 == ang1:
          continue
        
        if {ang1, ang2} in self.eqs:
          continue
          
        self.eqs.append({ang1, ang2})
        self.eqs.append({self.sup_of(ang1), self.sup_of(ang2)})
        
        new_eqs.append((ang1, ang2, self.pos[ang1].union(self.pos[ang2])))
        # print('{} = {} = {}'.format(
        #     self.val_name(hash), self.ang_name(ang1), self.ang_name(ang2)))

    return new_eqs
  
  def calculate(self, a, b):
    # m = a - b
    if (a, b) not in self.deps:
      result = defaultdict(lambda: 0)
      result.update(self.deps[a])
      self.deps[(a, b)] = substract(result, self.deps[b])
      self.pos[(a, b)] = self.pos[a].union(self.pos[b])

      val = self.deps[(a, b)]
      hash = []
      for f in self.free:
        if f in val:
          hash.append((f, val[f]))
      
      hash = tuple(hash)
      self.a2v[(a, b)] = hash
      self.v2a[hash].append((a, b))
    
    return self.a2v[(a, b)]

  def calculate_sup(self, a, b):
    if (self.pi, a, b) not in self.deps:
      result = defaultdict(lambda: 0)
      result[self.pi] = 1.0
      self.deps[(self.pi, a, b)] = substract(
          result, self.deps[(a, b)])
      self.pos[(self.pi, a, b)] = self.pos[a].union(self.pos[b])

      val = self.deps[(self.pi, a, b)]
      hash = []
      for f in self.free:
        if f in val:
          hash.append((f, val[f]))

      hash = tuple(hash)
      self.a2v[(self.pi, a, b)] = hash
      self.v2a[hash].append((self.pi, a, b))

    return self.a2v[(self.pi, a, b)]


def point_greater(p1, p2):
  try:
    if round(p1.y, 12) > round(p2.y, 12):
      return True
    if round(p1.y, 12) < round(p2.y, 12):
      return False
    if round(p1.x, 12) > round(p2.x, 12):
      return True
    if round(p1.x, 12) < round(p2.x, 12):
      return False
    return p2.name > p1.name
  except AttributeError:
    return p1 > p2


class LineEngine(object):

  def __init__(self, lineobj):
    self.lineobj = lineobj
    self.points = set()

    self.free = []
    self.deps = {}
    self.pos = {}
    
    self.eqs = []

    self.v2s = defaultdict(lambda: [])
    self.s2v = {}  #defaultdict(lambda: None)

  def __getitem__(self, point):
    return self.deps[point]

  def copy(self):
    e = LineEngine(self.lineobj)
    e.points = set(self.points)
    e.free = list(self.free)
    e.deps = {x: dict(y) for x, y in self.deps.items()}
    e.pos = {x: set(y) for x, y in self.pos.items()}
    self.eqs = list(self.eqs)
    
    v2s = defaultdict(lambda: [])
    for v, s in self.v2s.items():
      v2s[v] = list(s)
    e.v2s = v2s

    s2v = defaultdict(lambda: None)
    for s, v in self.s2v.items():
      s2v[s] = v
    e.s2v = s2v
    return e

  def is_equal(self, s1, s2):
    return s1 in self.s2v and self.s2v[s1] == self.s2v[s2]

  def add(self, *ps):
    for p in list(ps):
      self.points.add(p)

  def add_free(self, *vars):
    autos = {}
    for v in list(vars):
      self.deps[v] = defaultdict(lambda: 0.0)
      self.deps[v][(self, v)] = 1.0
      self.pos[v] = set()
      autos.update(self.auto(v))

      self.free.append(v)
      self.points.add(v)
    return autos

  def add_seg(self, a, b, d, pos_d=set()):
    if a in self.deps and b not in self.deps:
      result = defaultdict(lambda: 0.0)
      result.update(self.deps[a])
      self.deps[b] = substract(result, d)
      self.pos[b] = pos_d.union(self.pos[a])
      return self.auto(b)

    elif a not in self.deps and b in self.deps:
      result = defaultdict(lambda: 0.0)
      result.update(self.deps[b])
      self.deps[a] = add(result, d)
      self.pos[a] = pos_d.union(self.pos[b])
      return self.auto(a)

    elif a not in self.deps and b not in self.deps:
      autos = self.add_free(b)
      result = defaultdict(lambda: 0.0)
      result.update(self.deps[b])
      self.deps[a] = add(result, d)
      self.pos[a] = pos_d.union(self.pos[b])
      autos.update(self.auto(a))
      return autos

    return {}
      
  def add_eq(self, a, b, x, y, chain_pos=-1):
    # a - b = x - y = > a - b - x + y = 0
    # if self.is_equal((a, b), (x, y)):
    if {(a, b), (x, y)} in self.eqs:
      return []

    pos = set([chain_pos])
    result = defaultdict(lambda: 0.)

    dep2div = defaultdict(lambda: 0.0)
    for t, s in zip([a, b, x, y], [1., -1., -1., 1.]):
      if t in self.deps:
        result = add(result, {k: v * -s for k, v in self.deps[t].items()})
        pos.update(self.pos[t])
      else:
        dep2div[t] += s

    autos = {}
    for t, s in dep2div.items():
      if s == 0:
        autos.update(self.add_free(t))
    dep2div = dict((t, s) for t, s in dep2div.items() if s != 0)

    if dep2div:
      if len(dep2div) == 1:
        dep, div = dep2div.items()[0]
        self.deps[dep] = {k: v / div for k, v in result.items()}
        self.pos[dep] = pos
        autos.update(self.auto(dep))
      elif len(dep2div) > 1:
        alldeps = list(dep2div.keys())
        for d in alldeps[:-1]:
          autos.update(self.add_free(d))
        autos.update(self.add_eq(a, b, x, y, chain_pos))

    self.eqs.append({(a, b), (x, y)})
    return autos

  def seg_name(self, s):
    p1, p2 = s
    return p1.name + p2.name

  def val_name(self, x):
    if isinstance(x, float):
      return str(x)
    name = ''
    for d, v in x:
      if v > 0:
        if name != '':
          name += '+'
      else:
        name += '-'
      if abs(v) != 1:
        name += str(abs(v))
      name += d.name
    return name

  def is_const(self, hash):
    return len(hash) == 0

  def auto(self, v):
    # find vars that is not v:
    vars = [v0 for v0 in self.deps 
            if v0 not in [v] and
            not isinstance(v0, tuple)]  # not a segment.

    new_len_hash = {}
    for v0 in vars:
      if point_greater(v0, v):
        a, b = v0, v
      else:
        a, b = v, v0
      
      hash1 = self.calculate(a, b)
      new_len_hash.update({(a, b): (hash1, self.pos[(a, b)])})
      # hash2 = self.calculate(b, a)
      # new_len_hash.update({(b, a): hash2})
    
    # for seg, v in new_len_hash.items():
    #   a, b = seg
    #   v = [str(n) + lp for lp, n in v]
    #   print(a.name+b.name, '=', v)

    return new_len_hash
  
  def calculate(self, a, b):

    if (a, b) not in self.deps:
      result = defaultdict(lambda: 0)
      result.update(self.deps[a])
      self.deps[(a, b)] = substract(result, self.deps[b])
      self.pos[(a, b)] = self.pos[a].union(self.pos[b])

      val = self.deps[(a, b)]

      hash = sorted([
          (getattr(l.lineobj, 'name', l.lineobj) + '.' + 
           getattr(p, 'name', p), n) 
          for (l, p), n in val.items()
      ])
      
      hash = tuple(hash)
      self.s2v[(a, b)] = hash
      self.v2s[hash].append((a, b))
    
    return self.s2v[(a, b)]


class ddict(dict):

  def __init__(self, fn=lambda: None):
    # self.d = {}
    self.fn = fn

  def __getitem__(self, k):
    if k not in self:
      self[k] = self.fn(k)

    return self.get(k)


class DistanceEngine(object):
  
  def __init__(self):
    self.lines = ddict(lambda x: LineEngine(x))
    self.v2s = {}

  def copy(self):
    e = DistanceEngine()
    e.lines.update({lname: lengine.copy() for lname, lengine in 
                    self.lines.items()})
    e.v2s = {x: set(y) for x, y in self.v2s.items()}
    return e

  def __getitem__(self, obj):
    if isinstance(obj, tuple):
      p1, p2 = obj
      if point_greater(p2, p1):
        p1, p2 = p2, p1
      l = self.line_of(p1, p2)
      return l.s2v[(p1, p2)]
    return self.lines[obj]

  def line_of(self, p1, p2):
    for l in self.lines.values():
      if p1 in l.points and p2 in l.points:
        return l
    return None

  def add(self, l, *ps):
    self.lines[l].add(*ps)

  def add_free(self, l, *v):
    autos = self.lines[l].add_free(*v)
    self.add_autos(autos)

  def add_eq(self, a, b, x, y, chain_pos=-1):

    if point_greater(b, a):
      a, b = b, a

    if point_greater(y, x):
      x, y = y, x

    lab = self.line_of(a, b)
    lxy = self.line_of(x, y)
    if lab == lxy:
      autos = lab.add_eq(a, b, x, y, chain_pos)
    else:
      if (x, y) in lxy.deps:
        autos = lab.add_seg(a, b, lxy.deps[(x, y)], lxy.pos[(x, y)] | {chain_pos})
      elif (a, b) in lab.deps:
        autos = lxy.add_seg(x, y, lab.deps[(a, b)], lab.pos[(a, b)] | {chain_pos})
      else:
        autos = lab.add_free(a, b)
        autos.update(lxy.add_seg(
            x, y, lab.deps[(a, b)], lab.pos[(a, b)] | {chain_pos}))

    v = lab.s2v[(a, b)]
    if v not in self.v2s:
      self.v2s[v] = set()
    self.v2s[v].add((a, b))
    self.v2s[v].add((x, y))

    return self.add_autos(autos)
  
  def add_autos(self, autos):
    new_eqs = []
    for seg, (v, chain_pos) in autos.items():
      if v in self.v2s:
        if v != ():
          if seg in self.v2s[v]:
            continue
          for x in self.v2s[v]:
            new_eqs.append((v, x, seg, chain_pos))

        self.v2s[v].add(seg)
      else:
        self.v2s[v] = {seg}

      if v == ():
        new_eqs.append((0.0, seg, chain_pos))

    return new_eqs


def test_mid_point():
  print('Test mid point')
  e = DistanceEngine()
  e.add_free('az', 'A')
  e.add_free('az', 'Z')
  e.add('az', 'M')
  r = e.add_eq('A', 'M', 'M', 'Z')
  assert r == []
  e.add('az', 'N')
  r = e.add_eq('Z', 'N', 'N', 'A')
  assert r == [(0.0, ('N', 'M'), set([-1]))]

  assert e['az']['M'] == e['az']['N']


def test_equal_plus():
  print('Test adding vectors')
  e = DistanceEngine()
  e.add_free('ab', 'A')
  e.add_free('ab', 'B')
  e.add_free('mn', 'M')
  e.add('mn', 'N')
  r = e.add_eq('A', 'B', 'M', 'N')
  assert r == []

  e.add_free('mn', 'P')
  e.add('ab', 'C')

  r = e.add_eq('M', 'P', 'A', 'C')
  assert r == [
      ((('ab.A', 1.0), ('ab.B', -1.0), 
        ('mn.M', -1.0), ('mn.P', 1.0)), ('P', 'N'), ('C', 'B'), set([-1]))
  ]

  assert e[('B', 'C')] == e[('N', 'P')]


def test_equal_bisects():
  print('test equal bisects')
  e = DistanceEngine()
  e.add_free('az', 'A', 'Z')
  e.add('az', 'M')
  r = e.add_eq('A', 'M', 'Z', 'M')
  assert r == []

  e.add_free('by', 'B')
  e.add('by', 'Y')
  r = e.add_eq('A', 'Z', 'B', 'Y')
  assert r == []
  e.add('by', 'N')
  r = e.add_eq('Y', 'N', 'B', 'N')
  assert r == []

  assert e[('B', 'N')] == (('az.A', -0.5), ('az.Z', 0.5))
  assert e[('B', 'N')] == e[('Y', 'N')]
  assert e[('B', 'N')] == e[('M', 'Z')]
  assert e[('A', 'M')] == e[('M', 'Z')]


def test_equal_dist():
  print('test equal dist')
  e = DistanceEngine()
  e.add('az', 'A', 'B', 'Z')
  r = e.add_eq('A', 'Z', 'B', 'Z')
  assert r == [(0.0, ('B', 'A'), set([-1]))]


def test_three_medians():
  e = DistanceEngine()
  e.add_free('ab', 'A', 'B')
  e.add_free('bc', 'B', 'C')
  e.add_free('ca', 'C', 'A')
  e.add('ab', 'A0')
  e.add('bc', 'B0')
  e.add('ca', 'C0')
  e.add_eq('A', 'A0', 'A0', 'B')
  e.add_eq('B', 'B0', 'B0', 'C')
  e.add_eq('C', 'C0', 'C0', 'A')
  e.add('ab0', 'A', 'G1', 'G2', 'B0')
  e.add('bc0', 'B', 'G1', 'C0')
  e.add('ca0', 'C', 'G2', 'A0')
  e.add_eq_ratio('')


if __name__ == '__main__':
  profiling.enable()
  with Timer('test'):
    # test_mid_point()
    # test_equal_plus()
    # test_equal_bisects()
    # test_equal_dist()
    test_three_medians()
  
  profiling.print_records()