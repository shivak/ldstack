import numpy as np
import tensorflow as tf
import sys
sys.path.append('./')
from linear_recurrent_net.tensorflow_binding import sparse_linear_recurrence, sparse_linear_recurrence_cpu, sparse_linear_recurrence_naive

T = 128
b = 4
k = 3
n = 6

decay = (np.random.uniform(size=(k, n)) + 1j*np.random.uniform(size=(k, n))).astype(np.complex64) / 2.0
scales = np.random.normal(size=(T, b, k)).astype(np.complex64)
impulse = (np.random.normal(size=(k, n)) + 1j*np.random.normal(size=(k, n))).astype(np.complex64)
init_state = np.random.normal(size=(k,n)).astype(np.complex64)

def relnorm(sym, num):
  return tf.linalg.norm(sym - num) / tf.linalg.norm(num)

def err(vals):
  return [relnorm(sym, num).numpy() for sym, num in zip(*vals)]

def run_test(slin_rec):
  response = lambda inp: slin_rec(tf.complex(inp, decay.imag), scales, impulse, init_state)
  print('Decay.real grad err:', err(tf.test.compute_gradient(response, [decay.real])))

  response = lambda inp: slin_rec(tf.complex(decay.real, inp), scales, impulse, init_state)
  print('Decay.imag grad err:', err(tf.test.compute_gradient(response, [decay.imag])))

  response = lambda inp: slin_rec(inp, scales, impulse, init_state)
  print('Decay grad err:', err(tf.test.compute_gradient(response, [decay])))

  response = lambda inp: slin_rec(decay, inp, impulse, init_state)
  print('Scales grad err:', err(tf.test.compute_gradient(response, [scales])))
  
  response = lambda inp: slin_rec(decay, scales, inp, init_state)
  print('Impulse grad err:', err(tf.test.compute_gradient(response, [impulse])))

  response = lambda inp: slin_rec(decay, scales, impulse, inp)
  print('Init state grad err:', err(tf.test.compute_gradient(response, [init_state])))

if __name__ == "__main__":
  print("GPU vs CPU forward pass")
  print(relnorm(sparse_linear_recurrence(decay, scales, impulse, init_state),
                sparse_linear_recurrence_cpu(decay, scales, impulse, init_state)))
  print("GPU")
  run_test(sparse_linear_recurrence)
  print("CPU")
  run_test(sparse_linear_recurrence_cpu)