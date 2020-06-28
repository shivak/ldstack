import tensorflow as tf
from LDStack import LDStack

'''
A MIMO LDS in (perturbed) Luenberger canonical form
n/d SIMO LDS, each of size d, with inputs coupled by a matrix V
'''
class LuenbergerLDS(tf.keras.layers.Layer):
  def __init__(self, n, m, eig_param=randroot_eig_param(), out_param=relaxed_out_param(), last_only=False):
    super(LuenbergerLDS, self).__init__()
    self.total_n = n
    self.m = m
    self.eig_param = eig_param
    self.out_param = out_param
    self.last_only = last_only

  def build(self, input_shape):
    _,T,d = input_shape
    print(input_shape)
    if self.total_n % d != 0:
      raise "n (LDS state size) must be divisible by d (input dimension)"
    n = round(self.total_n / d)
    k = d
    self.V = tf.Variable(tf.random.normal((d,d), stddev=1/float(d)), name="V", dtype=tf.float32)
    self.lds = LDStack(n, d, self.m, k, Δ=1, eig_param=self.eig_param, out_param=self.out_param, standard=True, last_only=self.last_only)

  def call(self, x):
    V·x = tf.tensordot(x, self.V, [[-1], [0]])
    return self.lds(V·x)