from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
sys.path.append('./')
from linear_recurrent_net.tensorflow_binding import linear_recurrence

n_dims = 20
n_steps = 30

np.random.seed(2016)
decays = np.random.uniform(size=(n_steps, n_dims)).astype(np.complex64) 
impulses = np.random.randn(n_steps, n_dims).astype(np.complex64)
initial_state = np.random.randn(n_dims).astype(np.complex64)

with tf.Session() as sess:
    inp = tf.constant(decays, dtype=tf.complex64)
    print(inp.dtype)
    print("goat")
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
