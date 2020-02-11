from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
sys.path.append('./')
from linear_recurrent_net.tensorflow_binding import linear_recurrence, linear_recurrence_cpu

n_dims = 20
n_steps = 30

np.random.seed(2018)
#decays = np.random.uniform(size=(n_steps, n_dims)).astype(np.complex64) 
decays = (np.random.uniform(size=(n_steps, n_dims)) + 1.0j*np.random.uniform(size=(n_steps, n_dims))).astype(np.complex64) / 2.0
impulses = np.random.randn(n_steps, n_dims).astype(np.complex64)
initial_state = np.random.randn(n_dims).astype(np.complex64)

def run_test(linear_recurrence, sess):
    inp = tf.constant(decays.real)
    response = linear_recurrence(tf.complex(inp, tf.constant(decays.imag)), impulses, initial_state)
    print('Decays.real grad err:',
          tf.test.compute_gradient_error(inp, decays.shape,
                                        response, impulses.shape)
    )
    inp = tf.constant(decays.imag)
    response = linear_recurrence(tf.complex(tf.constant(decays.real), inp), impulses, initial_state)
    print('Decays.imag grad err:',
          tf.test.compute_gradient_error(inp, decays.shape,
                                        response, impulses.shape)
    )

    inp = tf.constant(decays, dtype=tf.complex64)
    response = linear_recurrence(inp, impulses, initial_state)
    print('Decays grad err:',
          tf.test.compute_gradient_error(inp, decays.shape,
                                        response, impulses.shape)
    )

    inp = tf.constant(impulses, dtype=tf.complex64)
    response = linear_recurrence(decays, inp, initial_state)
    print('Impulses grad err:',
          tf.test.compute_gradient_error(inp, impulses.shape,
                                        response, impulses.shape)
    )

    inp = tf.constant(initial_state, dtype=tf.complex64)
    response = linear_recurrence(decays, impulses, inp)
    print('Initial state grad err:',
          tf.test.compute_gradient_error(inp, initial_state.shape,
                                        response, impulses.shape)
    )

if __name__ == "__main__":
  with tf.Session() as sess:
    print("GPU")
    run_test(linear_recurrence, sess)
    print("CPU")
    run_test(linear_recurrence_cpu, sess)