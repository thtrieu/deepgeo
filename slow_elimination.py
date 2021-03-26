from symengine import symbols
# from sympy import symbols
from sympy import factor, simplify



class LineEngine(object):
  
  def __init__(self, name):
    self.name = name
    self.points = {}
    self.segments = {}

  def add_free(self, p):
    self.points[p] = symbols(self.name + '.' + p)
    return self.auto(p)

  def add(self, p, v=None):
    self.points[p] = v
    if v is not None:
      return self.auto(p)

  def know(self, p):
    return p in self.points and self.points[p] != None

  def add_free_seg(self, a, b):
    if self.know(a):
      return self.add_free(b)
    elif self.know(b):
      return self.add_free(a)
    else:
      autos = self.add_free(a)
      autos.update(self.add_free(b))
      return autos

  def add_seg(self, a, b, r, e=1):
    if e < 0:
      r = 1./r
      e = -e

    if e != 1:
      r = r ** (1./e)

    self.segments[(a, b)] = r
    self.segments[(b, a)] = -r

    if self.know(a):
      return self.add(b, self.points[a] - r)
    elif self.know(b):
      return self.add(a, self.points[b] + r)
    else:
      autos = self.add_free(a)
      autos.update(self.add(b, self.points[a] - r))
      return autos

  def add_eq_seg(self, a, b, x, y, m=1):
    # (a-b) = m(x - y)
    if (a, b) in self.segments:
      self.segments[(x, y)] = self.segments[(a, b)]/m
      self.segments[(y, x)] = -self.segments[(x, y)]
    if (x, y) in self.segments:
      self.segments[(a, b)] = self.segments[(x, y)] * m
      self.segments[(b, a)] = -self.segments[(a, b)]

    r = 0
    deps = defaultdict(lambda: 0)
    for t, s in zip([a, b, x, y], [1, -1, -m, m]):
      if self.know(t):
        r -= s * self.points[t]
      else:
        deps[t] += s
    
    if len(deps) == 0:

      t2s = defaultdict(lambda: 0)
      for t, s in zip([a, b, x, y], [1, -1, -m, m]):
        t2s[t] += s

      t0 = None
      for t in t2s:
        if self.is_free(t):
          t0 = t
          break

      if t0 is None:
        print('todo no deps no free')
        return

      for t, s in t2s.items():
        if t != t0:
          r -= s * self.points[t]
      
      self.add(t0, r/t2s[t0])
      return self.force_auto(t0)
    else:
      return self.add_one_point(deps.items(), r)

  def add_abxy_pqrs(self, a, b, x, y, p, q, r, s):
    # (a-b)/(x-y) = (p-q)/(r-s)
    # (a-b)*(r-s) = (p-q)*(x-y)
    if (a-b) in self.segments:
      return self.add_abxy_mpq(p, q, x, y, r, s, self.segments[(a, b)])
    elif (p-q) in self.segments:
      return self.add_abxy_mpq(a, b, r, s, x, y, self.segments[(p, q)])
    elif (x-y) in self.segments:
      return self.add_abxy_mpq(a, b, r, s, p, q, self.segments[(x, y)])
    elif (r-s) in self.segments:
      return self.add_abxy_mpq(p, q, x, y, a, b, self.segments[(r, s)])
    
    autos = self.add_free_seg(a, b)
    autos.update(self.add_abxy_mpq(p, q, x, y, r, s, self.segments[(a, b)]))
    return autos

  def add_abxy_mpq(self, a, b, x, y, p, q, m):
    # (a-b)(x-y) = m(p-q)
    if (a, b) in self.segments:
      return self.add_eq_seg(x, y, p, q, m / self.segments[(a, b)])
    elif (x, y) in self.segments:
      return self.add_eq_seg(a, b, p, q, m / self.segments[(x, y)])
    elif (p, q) in self.segments:
      return self.add_abxy_m(a, b, x, y, m * self.segments[(p, q)])
    
    autos = self.add_free_seg(a, b)
    autos.update(self.add_eq_seg(x, y, p, q, m / self.segments[(a, b)]))
    return autos
  
  def add_abxy_m(self, a, b, x, y, m):
    # (a-b)(x-y) = m
    if (a, b) in self.segments:
      return self.add_seg(x, y, m / self.segments[(a, b)])
    elif (x, y) in self.segments:
      return self.add_seg(a, b, m / self.segments[(x, y)])
    
    autos = self.add_free_seg(a, b)
    autos.update(self.add_seg(x, y, m / self.segments[(a, b)]))
    return autos

  def add_one_point(self, t_and_s, r):
    autos = {}

    for t, s in t_and_s[:-1]:
      autos.update(self.add_free(t))
      r -= s * self.points[t]
    
    t, s = t_and_s[-1]
    autos.update(self.add(t, r/s))
    return autos

  def is_free(self, p):
    return p not in self.points or self.points[p] == symbols(self.name + '.' + p)
  
  def auto(self, p):
    added = {}
    # return added

    for p0 in self.points:
      if p0 == p:
        continue

      if (p, p0) in self.segments or (p0, p) in self.segments:
        continue

      added[(p, p0)] = factor(self.points[p] - self.points[p0])
      added[(p0, p)] = -added[(p, p0)]
        
    self.segments.update(added)
    return added

  def force_auto(self, p):
    added = {}

    for p0 in self.points:
      if p0 == p:
        continue
      added[(p, p0)] = factor(self.points[p] - self.points[p0])
      added[(p0, p)] = -added[(p, p0)]
    
    self.segments.update(added)
    return added

  def __getitem__(self, p):
    return self.points[p]


class ddict(dict):

  def __init__(self, fn=lambda: None):
    # self.d = {}
    self.fn = fn

  def __getitem__(self, k):
    if k not in self:
      self[k] = self.fn(k)

    return self.get(k)

  # def values(self):
  #   return self.d.values()
  

class DistanceEngine(object):
  
  def __init__(self):
    self.lines = ddict(lambda x: LineEngine(x))
    self.v2seg = {}

  def __getitem__(self, obj):
    if isinstance(obj, tuple):
      p1, p2 = obj
      l = self.line_of(p1, p2)
      return l.segments[(p1, p2)]
    return self.lines[obj]

  def line_of(self, p1, p2):
    for l in self.lines.values():
      if p1 in l.points and p2 in l.points:
        return l
    return None

  def add_eq_seg(self, a, b, x, y):
    lab = self.line_of(a, b)
    lxy = self.line_of(x, y)
    if lab == lxy:
      autos = lab.add_eq_seg(a, b, x, y)
    else:
      if (x, y) in lxy.segments:
        return lab.add_seg(a, b, lxy.segments[(x, y)])
      elif (a, b) in lab.segments:
        return lxy.add_seg(x, y, lab.segments[(a, b)])
      else:
        autos = lab.add_free_seg(a, b)
        autos.update(lxy.add_seg(x, y, lab.segments[(a, b)]))
  
    v = lab.segments[(a, b)]
    if v not in self.v2seg:
      self.v2seg[v] = set()
    self.v2seg[v].add((a, b))
    self.v2seg[v].add((x, y))

    if -v not in self.v2seg:
      self.v2seg[-v] = set()
    self.v2seg[-v].add((b, a))
    self.v2seg[-v].add((y, x))

    self.add_autos(autos)

  def _check(self, vars, la, p1, p2, p3, p4, lb, p5, p6, p7, p8):
    if p2 == p3 and p6 == p7:
      if not la.know(p2) or not lb.know(p6):
        del vars[:]
        vars.extend([p1, p2, p1, p4, p5, p6, p5, p8])

  def check(self, p1, p2, p3, p4, p5, p6, p7, p8):
    vars = [p1, p2, p3, p4, p5, p6, p7, p8]
    l12 = self.line_of(p1, p2)
    l34 = self.line_of(p3, p4)
    l56 = self.line_of(p5, p6)
    l78 = self.line_of(p7, p8)

    # return vars, l12, l34, l56, l78

    if l12 == l34 and l56 == l78:
      pass
    elif l12 == l56 and l34 == l78:
      p1, p2, p3, p4, p5, p6, p7, p8 = (
          p1, p2, p5, p6, p3, p4, p7, p8)
      l34, l56 = l56, l34
    else:
      return vars, l12, l34, l56, l78

    self._check(vars, l12, p1, p2, p3, p4, l56, p5, p6, p7, p8)
    self._check(vars, l12, p2, p1, p3, p4, l56, p6, p5, p7, p8)
    self._check(vars, l12, p1, p2, p4, p3, l56, p5, p6, p8, p7)
    self._check(vars, l12, p2, p1, p4, p3, l56, p6, p5, p8, p7)
    return vars, l12, l34, l56, l78

  def add_eq_ratio(self, p1, p2, p3, p4, p5, p6, p7, p8):
    # p1-p2/p3-p4=p5-p6/p7-p8
    # or
    # p1-p2/p5-p6=p3-p4/p7-p8

    [p1, p2, p3, p4, 
     p5, p6, p7, p8], l12, l34, l56, l78 = self.check(
        p1, p2, p3, p4, p5, p6, p7, p8)

    result = 1
    deps = defaultdict(lambda: 0)

    for t, l, s in zip([(p1, p2), (p3, p4), (p5, p6), (p7, p8)], 
                       [l12, l34, l56, l78], [1, -1, -1, 1]):
      if t in l.segments:
        if s > 0:
          result /= l.segments[t]
        else:
          result *= l.segments[t]
      else:
        deps[t] += s

    if len(deps) == 0:
      print('todo eq ratio no deps')
      return  # TODO(thtrieu)
    else:
      autos = self.add_one_seg(deps.items(), result)
    self.add_autos(autos)

  def add_one_seg(self, t_and_s, r):
    autos = {}

    l2ts = defaultdict(lambda: [])

    for t, s in t_and_s:
      x, y = t
      l = self.line_of(x, y)
      l2ts[l].append((t, s))
    
    l_ts = l2ts.items()
    l_ts = sorted(l_ts, key=lambda lts: len(lts[1]))

    for l, ts in l_ts[:-1]:
      for t, s in ts:
        if t not in l.segments:
          autos.update(l.add_free_seg(*t))
        if s == 0:
          continue

        if s > 0:
          f = 1 / l.segments[t]
        else:
          s = -s
          f = l.segments[t]
        
        for _ in range(s):
          r *= f
    
    l, ts = l_ts[-1]
    pos, neg = [], []
    for t, s in ts:
      if s == 0:
        continue
      if s > 0:
        for _ in range(s):
          pos += [t]
      else:
        for _ in range(-s):
          neg += [t]

    if len(neg) > len(pos):
      pos, neg = neg, pos
      r = 1./r
    # pos / neg = r

    # 1: (x-y) = r
    if len(pos) == 1 and len(neg) == 0:
      x, y = pos[0]
      autos.update(l.add_seg(x, y, r))
    
    # 2: (x-y)(a-b) = r
    elif len(pos) == 2 and len(neg) == 0:
      (x, y), (a, b) = pos
      autos.update(l.add_abxy_m(a, b, x, y, r))
    
    # 2: (x-y) = (a-b) r
    elif len(pos) == 1 and len(neg) == 1:
      (x, y) = pos[0]
      (a, b) = neg[0]
      autos.update(l.add_eq_seg(x, y, a, b, r))
    
    # 3: (x-y)(a-b) = r(t-v)
    elif len(pos) == 2 and len(neg) == 1:
      (a, b), (x, y) = pos
      (p, q) = neg[0]
      autos.update(l.add_abxy_mpq(a, b, x, y, p, q, r))
      
    # 4: (x-y)(a-b) = (t-v)(r-s)
    elif len(pos) == 2 and len(neg) == 2:
      (a, b), (x, y) = pos
      (p, q), (r, s) = neg
      autos.update(l.add_abxy_pqrs(a, b, x, y, p, q, r, s))
    
    else:
      print('todo {}pos {}neg'.format(len(pos), len(neg)))
      return {}

    return autos
  
  def add_autos(self, autos):
    new_eqs = []

    for seg, v in autos.items():
      if v in self.v2seg:
        if seg in self.v2seg[v]:
          continue
        for x in self.v2seg[v]:
          new_eqs.append((v, x, seg))
        self.v2seg[v].add(seg)
      else:
        self.v2seg[v] = {seg}

    # print new_eqs:
    for v, s1, s2 in new_eqs:
      s1 = s1[0] + s1[1]
      s2 = s2[0] + s2[1]
      print('{} = {} = {}'.format(s1, s2, v))
    # print(new_eqs)
    

def test_eq_ratio():

  print('Test eq ratios')
  e = DistanceEngine()

  e['ab'].add_free('A')
  e['ab'].add_free('B')

  e['bc'].add_free('B')
  e['bc'].add_free('C')

  e['ca'].add_free('A')
  e['ca'].add('C')

  # import pdb; pdb.set_trace()
  e.add_eq_seg('C', 'A', 'A', 'B')
  e['bc'].add_free('M')

  e['ab'].add('N')
  # s = time.time()
  e.add_eq_ratio(
      'N', 'A', 'N', 'B', 
      'M', 'B', 'M', 'C')
  # print(time.time()-s)
  
  e['ca'].add('P')
  # s = time.time()
  e.add_eq_ratio(
      'P', 'C', 'M', 'B', 
      'P', 'A', 'M', 'C')
  # print(time.time()-s)

  na = e[('N', 'A')]
  pc = e[('P', 'C')]

  assert na == pc

  # for l in e.lines.values():
  #   for s in l.segments:
  #     print(s, l.segments[s])


def test_mid_point():
  print('Test mid point')
  e = DistanceEngine()
  e['ab'].add_free('A')
  e['ab'].add_free('B')
  e['ab'].add('M')
  e.add_eq_seg('A', 'M', 'M', 'B')

  e['ab'].add('N')
  e.add_eq_seg('B', 'N', 'N', 'A')
  assert e['ab']['M'] == e['ab']['N']