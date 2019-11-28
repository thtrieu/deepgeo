import tensorflow as tf
import numpy as np
import time

import problem
import model

n_bit = 32

a_dec = tf.placeholder(tf.int32, [None, 512])
a_bin = model.dec_to_att_mask(a_dec)

# def bin2dec_v1_(arr):
#   arr = arr[::-1]
#   assert len(arr) == n_bit
#   if arr[0] == 1 and sum(arr) > 1 :
#     return - 1 - int(''.join(str(1-y) for y in arr), 2)
#   else:
#     return int(''.join(str(y) for y in arr), 2)

# def bin2dec_v2_(arr):
#   arr = arr[::-1]
#   assert len(arr) == n_bit
#   num = int(''.join(str(y) for y in arr), 2)
#   if arr[0] == 1 and sum(arr) > 1 :
#     return num - 2**n_bit
#   return num


# def bin2dec_v1(arr):
#   return [bin2dec_v1_(x) for x in arr]


# def bin2dec_v2(arr):
#   return [bin2dec_v2_(x) for x in arr]


def bin2dec_v3(a):
  # import pdb; pdb.set_trace()
  bool_array = a.astype(bool)
  return problem.bin2dec_v3(bool_array)


if n_bit == 4:
  ar = np.array([[1, 0, 0, 1], # -7
                 [0, 1, 0, 1], # -6
                 [1, 1, 0, 1], # -5
                 [0, 0, 1, 1], # -4
                 [1, 0, 1, 1], # -3
                 [0, 1, 1, 1], # -2
                 [1, 1, 1, 1], # -1
                 [0, 0, 0, 0], # 0
                 [1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [1, 1, 0, 0],
                 [0, 0, 1, 0],
                 [1, 0, 1, 0],
                 [0, 1, 1, 0],
                 [1, 1, 1, 0],
                 [0, 0, 0, 1]])

  # print(bin2dec_v1(ar))
  # print(bin2dec_v2(ar))
  print(bin2dec_v3(ar))
  ar_ = s.run(a_bin, {a_dec: bin2dec_v3(ar)})
  assert np.array_equal(ar_, ar)


# batch_size = 16 * 128 * 2
# start = time.time()
# bin2dec_v1(np.random.randint(2, size=(batch_size, n_bit)))
# print(time.time() - start)

# start = time.time()
# bin2dec_v2(np.random.randint(2, size=(batch_size, n_bit)))
# print(time.time() - start)

batch_size = 4096

start = time.time()
binary = np.random.randint(2, size=(batch_size, 128, 128))
ar = bin2dec_v3(binary.reshape((-1, 32))).reshape((batch_size, -1))
print(time.time() - start)

assert ar.shape == (batch_size, 512)

with tf.Session() as sess:
  start = time.time()
  binary_ = sess.run(a_bin, {a_dec: ar})
  print(time.time() - start)

print(binary[0], binary_[0])

assert np.array_equal(binary, binary_)
print('OK')




