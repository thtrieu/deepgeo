"""
Example:
"""

from elimination import DistanceEngine
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


def val_name(x):
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


def is_const(hash):
  return len(hash) == 0


class RatioEngine(object):

  def __init__(self, name):
    self.free = []
    self.deps = {}
    self.pos = {}  # which chain pos explain the value?
    
    self.eqs = []
    self.const2a = defaultdict(lambda: [])

    self.name = name

    self.v2a = defaultdict(lambda: [])
    self.a2v = defaultdict(lambda: None)

  def copy(self):
    e = RatioEngine(self.name)
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
      self.deps[v] = {v: 1.0}
      self.pos[v] = set()

      for v0 in self.free:
        self.calculate(v, v0)
        self.calculate(v0, v)

      self.free.append(v)
      
  def add_eq_ratio(self, a, b, x, y, chain_pos):
    # a - b = x - y = > a - b - x + y = 0
    # if self.is_equal((a, b), (x, y)):
    if {(a, b), (x, y)} in self.eqs:
      return []

    self.eqs.append({(a, b), (x, y)})

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

  def is_const(self, hash):
    return len(hash) == 0

  def auto(self, v):
    # find vars that is not v:
    vars = [v0 for v0 in self.deps 
            if v0 != v and
            not isinstance(v0, tuple)]

    new_ang_hash = []
    for v0 in vars:
      if v0.line.greater_than(v.line):
        a, b = v0, v
      else:
        a, b = v, v0
      
      hash = self.calculate(a, b)
      new_ang_hash.append(((a, b), hash))

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


if __name__ == '__main__':
  profiling.enable()
  with Timer('test'):
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
    pass
  
  profiling.print_records()   