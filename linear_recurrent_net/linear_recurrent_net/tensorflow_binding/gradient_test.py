n_dims = 20
n_steps = 30

np.random.seed(2018)
decays = (np.random.uniform(size=(n_steps, n_dims)) + 1j*np.random.uniform(size=(n_steps, n_dims))).astype(np.complex64) / 2.0
impulses = np.random.randn(n_steps, n_dims).astype(np.complex64)
initial_state = np.random.randn(n_dims).astype(np.complex64)

def err(vals):
  for sym, num in zip(*vals):
    return tf.linalg.norm(sym - num) / tf.linalg.norm(num)

def run_test(lin_rec):
  response = lambda inp: lin_rec(tf.complex(inp, decays.imag), impulses, initial_state)
  print('Decays.real grad err:', err(tf.test.compute_gradient(response, [decays.real])))

  response = lambda inp: lin_rec(tf.complex(decays.real, inp), impulses, initial_state)
  print('Decays.imag grad err:', err(tf.test.compute_gradient(response, [decays.imag])))

  response = lambda inp: lin_rec(inp, impulses, initial_state)
  print('Decays grad err:', err(tf.test.compute_gradient(response, [decays])))

  response = lambda inp: lin_rec(decays, inp, initial_state)
  print('Impulses grad err:', err(tf.test.compute_gradient(response, [impulses])))

  response = lambda inp: lin_rec(decays, impulses, inp)
  print('Initial state grad err:', err(tf.test.compute_gradient(response, [initial_state])))

if __name__ == "__main__":
  print("GPU")
  run_test(linear_recurrence)
  print("CPU")
  run_test(linear_recurrence_cpu)