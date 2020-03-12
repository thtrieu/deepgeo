import time


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
  global __ALL_RECORDERS
  print('Profiling averages * hit times:')
  for name, rec in sorted(_ALL_RECORDERS.items()):
    print('{:>20}: {:>9}e-8 * {}'.format(
        name, int(rec.avg()*10e8), rec.count))


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
