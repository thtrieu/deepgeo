import multiprocess as multiprocessing
import time
import random
import theorems_utils

# def pp(s):
#   if isinstance(s, (list, tuple)):
#     s = [pp(x) for x in s]
#     return '[{}]'.format(','.join(s))
#   elif isinstance(s, dict):
#     s = {pp(k): pp(v) for k, v in s.items()}
#     return '{' + ','.join(['{}:{}'.format(k, v)
#                            for k, v in s.items()]) + '}'
#   else:
#     return type(s).__name__


def target(idx, fn, arg, return_dict):
  result = fn(*arg)
  if result is not None:
    return_dict[idx] = result


def create_jobs(fns, args):
  manager = multiprocessing.Manager()
  return_dict = manager.dict()

  fn_arg = zip(fns, args)
  jobs = []
  for i, (f, a) in enumerate(fn_arg):
    arg = (i, f, a, return_dict)
    job = multiprocessing.Process(target=target, args=arg)
    jobs.append(job)

  return jobs, return_dict


def parallelize_return_on_first_finish(fns, args):
  jobs, return_dict = create_jobs(fns, args)

  [p.start() for p in jobs]
  while not return_dict:
    pass
  [p.terminate() for p in jobs]

  return return_dict.values()[0]


def parallelize(fns, args):
  jobs, return_dict = create_jobs(fns, args)

  [p.start() for p in jobs]
  [p.join() for p in jobs]

  result = []
  for k, v in return_dict.items():
    result += v
  return result


# def run_sequentially(fns, args):
#   t = args[1][0]
#   print(t, t.name)
#   jobs, return_dict = create_jobs(fns, args)

#   for i, p in enumerate(jobs):
#     p.start()
#     while i not in return_dict:
#       pass
#     p.join()
#     if i == 1:
#       r = fns[i](*args[i])[0]
#       assert t in r
#       for x in return_dict[i][0]:
#         if x.name == t.name:
#           print(x, x.name)

#   result = []
#   for k, v in return_dict.items():
#     result += v
#   return result


# def run_sequentially(fns, args):
#   result = []
#   for f, a in zip(fns, args):
#     r = f(*a)
#     if r is not None:
#       result += r
#   return result


def worker(n, f):
  t = random.randint(3, n+3)
  time.sleep(t * 0.1)
  return t, f


if __name__ == '__main__':
  n = 10
  fns = [worker] * n

  f = lambda: 1
  args = [[n, f]] * n

  print(f, f())

  _, f = parallelize_return_on_first_finish(fns, args)
  print(f, f())
