import time
from collections import defaultdict as ddict



_ENABLE_PROFILING = False
_ALL_RECORDERS = {}


def enable_profiling():
  global _ENABLE_PROFILING
  _ENABLE_PROFILING = True


def disable_profiling():
  global _ENABLE_PROFILING
  _ENABLE_PROFILING = False


class Recorder(object):

  def __init__(self, name):
    self.count = 0
    self.sum = 0.

  def add(self, t):
    self.sum += t
    self.count += 1

  def avg(self):
    return self.sum / self.count


def print_records():
  global _ALL_RECORDERS
  print('Profiling averages * hit times:')

  max_name_len = max(len(name) for name in _ALL_RECORDERS.keys())
  name_format = '{' + ':<{}'.format(max_name_len) + '}'

  def default_dict():
    return dict(sum=0.0,
                time=0.0, 
                children={})

  d = default_dict()

  for name, rec in _ALL_RECORDERS.items():
    x = d
    for scope in name.split('/'):
      if scope not in x['children']:
        x['children'][scope] = default_dict()
      x = x['children'][scope]
    x['time'] += rec.sum

  def siblings_sum(d):
    for children in d['children'].values():
      d['sum'] += children['time']
      siblings_sum(children)

  siblings_sum(d)

  for name, rec in sorted(_ALL_RECORDERS.items()):
    # remove scoping
    scopes = name.split('/')
    x = d
    for scope in scopes[:-1]:
      x = x['children'][scope]

    scopes, name = scopes[:-1], scopes[-1]
    name = '  ' * len(scopes) + name

    print((name_format + ': {:>8}e-5 * {:<10} ({:2.0%})').format(
        name, int(rec.avg()*10e5), rec.count, rec.sum/x['sum']))


class _Timer(object):

  def __init__(self, name, start=False):
    global _ALL_RECORDERS
    if name not in _ALL_RECORDERS:
      _ALL_RECORDERS[name] = Recorder(name)
    self.recorder = _ALL_RECORDERS[name]
    if start:
      self.start()

  def start(self):
    self.start_time = time.time()

  def stop(self):
    self.recorder.add(time.time() - self.start_time)

  def __enter__(self):
    self.start()

  def __exit__(self, type, value, traceback):
    self.stop()


class _NullTimer(object):

  def __init__(self, name, start=False):
    pass

  def start(self):
    pass

  def stop(self):
    pass

  def __enter__(self):
    pass

  def __exit__(self, type, value, traceback):
    pass


def Timer(name, start=False):
  global _ENABLE_PROFILING
  if _ENABLE_PROFILING:
    return _Timer(name, start=start)
  else:
    return _NullTimer(name, start=start)
