import numpy as np
import tensorflow as tf
from linear_recurrent_net.linear_recurrent_net.tensorflow_binding import linear_recurrence, sparse_linear_recurrence

# Takes [batch size, T, d] real
# Returns [batch_size, T, m] real (in averaging mode)
#      or [batch_size, T, d, m]   (with k=None)
# Note: requires a fixed batch size. This means you must:
# 1. Use an InputLayer in the Keras model specification. 
# 2. Pass a TF dataset which has been batched with drop_remainder=True to model.fit(), 
#    not a NumPy array. (Otherwise the last batch will be the wrong size.)
class LDStack(tf.keras.layers.Layer):
  # n: state size
  # d: input size
  # m: output size
  # k: number of random projections (averaging mode)
  # Δ: depth of stack
  # standard: whether to compute 
  # last_only: return just the last element of the output sequence
  def __init__(self, n, d, m, k=None, Δ=1, init='randroot', standard=True, last_only=False, num_stacks=1):
    super(LDStack, self).__init__()
    self.n = n
    self.m = m
    self.k = k
    self.Δ = Δ
    self.init = init
    self.standard = standard
    self.average = k is not None
    self.last_only = last_only
    self.num_stacks = num_stacks

  def build(self, input_shape):
    print("input shape is: ", input_shape)
    self.b, self.T, d = input_shape
    n, m, k, Δ, init,standard, average, last_only, num_stacks = (self.n, self.m, self.k, self.Δ, self.init, self.standard, self.average, self.last_only, self.num_stacks) 

    # Only perform random projection if number of projections is specified
    # Otherwise, run SIMO LDS on each coordinate of original input
    if average:
      self.R = tf.Variable(np.random.normal(size=(d,k)), name="R", dtype=tf.float32, trainable=False)
    else:
      k = d

    self.mid_layers = []
    # Performance optimization for sparse, purely-linear case 
    if Δ == 1 and num_stacks == 1 and last_only:
      self.last_layer = SparseLDS(n, m, k, init, average, standard)
    else:
      for i in np.arange(num_stacks-1):
        self.mid_layers.append( LDStackInner(n, k, k, Δ, init, average, standard) )
      self.last_layer = LDStackInner(n, m, k, Δ, init, average, standard, last_only=last_only)

  def call(self, x):
    if self.average:
      x = tf.tensordot(x, self.R, [[-1], [0]])

    x = tf.complex(tf.transpose(x, (1,0,2)), 0.0)
    for layer in self.mid_layers:
      x = layer(x)
      x = tf.reshape(x, (self.T, self.b, -1))
    y = tf.math.real(self.last_layer(x))

    # Return to batch-major shape 
    if self.last_only:
      # time axis already eliminated when last state was taken
      return y
    else:
      degree = len(y.shape)
      return tf.transpose(y, (1,0,2,3)) if degree == 4 else tf.transpose(y, (1,0,2))

# This LDStack implementation is somewhat unusual in that it supports multiple different parameterizations
# of the optimization variables. (For example, we can optimize in logspace, or not; over unitary eigenvalues
# parameterized solely by angles; etc.) These must be initialized in different manners, and are they are
# consumed by the algorithm in slightly different manners. 
def get_init_and_const_funcs(init):
  if init == 'unitary':
    return unitary_initialization, unitary_constitute
  elif init == 'randroot':
    return randroot_initialization, fixed_constitute

# Average of k SIMO LDS, only returning last state. Uses much more memory-efficient SparseLinearRecurrence op
class SparseLDS(tf.keras.layers.Layer):
  def __init__(self, n, m, k, init, average, standard=True):
    super(SparseLDS, self).__init__()
    initialize, self.constitute = get_init_and_const_funcs(init)
    self.underlying = initialize(n, m, k)
    self.k = k
    self.n = n
    self.m = m 
    self.average = average
    self.standard = standard

  def call(self, x):
    T, b, k = x.shape
    n = self.n
    lnλ, λ, Cʹ, D, Dₒ = self.constitute(*self.underlying)
    Bʹ = computeBʹ(lnλ, λ)

    # linear_recurrence computes sʹ_t = λ·sʹ_{t-1} + Bx_t
    # for standard LDS, we need to shift x
    # sʹ_t = λ·sʹ_{t-1} + Bx_{t-1}
    if self.standard:
      x = tf.concat([tf.zeros((1,b,k), dtype=tf.complex64),  x[:-1]], axis=0)
    # [b,k,n]
    sₜʹ = sparse_linear_recurrence(λ, x, Bʹ) 
    Cʹ·sₜʹ = tf.reduce_sum(tf.expand_dims(sₜʹ, -1)*tf.expand_dims(Cʹ,0), axis=-2) # [b, k, m]
    D·xₜ = tf.expand_dims(x[-1], -1) * tf.complex(tf.expand_dims(D, 0), 0.0)
    yₜ = Cʹ·sₜʹ + D·xₜ + tf.complex(Dₒ, 0.0)
    if self.average:
      yₜ = tf.reduce_mean(yₜ, axis=1) 
    return yₜ

  # TODO: (A, B, C, D) by computing elementary symmetric polynomials 
  def canonical_lds():
    return None

# Full generality, but slower
class LDStackInner(tf.keras.layers.Layer):
  def __init__(self, n, m, k, Δ, init, average, standard=True, last_only=False):
    super(LDStackInner, self).__init__()
    initialize, self.constitute = get_init_and_const_funcs(init)
    self.underlying = initialize(n, m, k)
    self.n = n
    self.m = m 
    self.k = k
    self.Δ = Δ
    self.average = average
    self.standard = standard
    self.last_only = last_only

  # x : [T, b, k]
  # λ : [k, n]
  # C : [k, m, n]
  # D : [k, m]
  # α : [T, b, k, n] is:
  #      α_0, ..., α_{T-1} (in "standard" mode)
  #      α_1, ..., α_T     (otherwise)
  # Returns (for t in [1,...,T]):
  # sʹ: [T, b, k, n] is:
  #     sʹ_t = α_{t-1}·λ·sʹ_{t-1}  + Bx_{t-1}   (in "standard" mode)
  #     sʹ_t = α_t    ·λ·sʹ_{t-1}  + Bx_t       (otherwise)
  # currently with sʹ_0 = 0
  # y : [T, b, k, m] is:
  #     y_t  = Cʹsʹ_t + Dx_t
  # y returned only if C, D, Dₒ are provided

  def batch_simo_lds(self, x, lnλ, λ, Cʹ=None, D=None, Dₒ=None, α=None, standard=True):
    k, n, m = (self.k, self.n, self.m)
    T, b, _ = x.shape
    states_only = Cʹ is None or D is None or Dₒ is None
    Bʹ = computeBʹ(lnλ, λ)

    # Bʹ·x : [T, b, k*n]   
    Bʹ·x = tf.reshape(tf.expand_dims(x, -1)*Bʹ, (T, b, k*n))
    
    # α·λ : [T, b, k*n]
    # sʹ : [T, b, k, n]  
    if α is None:
      α·λ =  tf.tile(tf.reshape(λ, (1,1,-1)), (T, b, 1)) 
    else:
      α·λ = α*tf.reshape(λ, (1, 1, k, n))  
      α·λ = tf.reshape(α·λ, (T, b, k*n))
    # linear_recurrence computes sʹ_t = α_t·λ·sʹ_{t-1} + Bx_t
    # for standard LDS, we need to shift α and x
    # sʹ_t = α_{t-1}·λ·sʹ_{t-1} + Bx_{t-1}
    if standard:
      α·λ  = tf.concat([tf.zeros((1,b,k*n), dtype=tf.complex64),  α·λ[:-1]], axis=0)
      Bʹ·x = tf.concat([tf.zeros((1,b,k*n), dtype=tf.complex64),  Bʹ·x[:-1]], axis=0)
    sʹ = linear_recurrence(α·λ, Bʹ·x)
    sʹ = tf.reshape(sʹ, [T, b, k, n])
    if states_only:
      return sʹ
      
    Cʹ·sʹ = tf.reduce_sum(tf.expand_dims(sʹ, -1)*tf.reshape(Cʹ, (1,1,k,n,m)), axis=-2)
    D·x = tf.expand_dims(x, -1) * tf.complex(tf.reshape(D, (1,1,k,m)), 0.0)
    y = Cʹ·sʹ + D·x + tf.complex(Dₒ, 0.0)

    return sʹ, y

  # x: [T, batch size, k] is complex (this is unusual, but matches the format of linear_recurrence.
  #       And actually, is faster for GPU computation, and should be the standard for RNNs.)
  # Returns complex output y of shape [T, batch size, m]
  def call(self, x):
    T, b, k = x.shape
    n = self.n
    lnλ, λ, Cʹ, D, Dₒ = self.constitute(*self.underlying)  

    # λ : [k, n]
    # C : [k, m, n]
    # α : [T, b, k, n]
    # sʹ: [T, b, k, n]
    # y : [T, b, k, m]
    α=None
    for i in np.arange(self.Δ - 1):
      sʹ = self.batch_simo_lds(x, lnλ, λ, α=α, standard=self.standard)
      λ·sʹ = tf.reshape(λ, (1,1,k,n)) * sʹ
      α = recipsq(λ·sʹ)
    _, y = self.batch_simo_lds(x, lnλ, λ, Cʹ, D, Dₒ, α, self.standard)
    if self.average:
      y = tf.reduce_mean(y, axis=2)
    if self.last_only:
      y = y[-1]
    return y

# Initialize with uniformly random complex eigenvalues of magnitude 1.
# If λ has polar coordinates (r, θ) then ln(λ) = ln(r) + θi
# Also, ln(λ*) = ln(r) - θi
# Fix r=1, i.e. ln(r) = 0. (For numerical reasons, can fix r=1-𝛿 for some small 𝛿) 
# Note this only requires n/2 parameters rather than n
# Should constrain -π ≤ θ ≤ π
def unitary_initialization(n, m, k, trainλ=True, 𝛿=0.02):
  if n % 2 != 0:
    raise "n must be even"

  half_n = round(n/2)
  θ_init = np.random.uniform(low=-np.pi, high=np.pi, size=[k,half_n]).astype(np.float32)
  θ = tf.Variable(θ_init, name="eig_angle", dtype=tf.float32, trainable=trainλ)

  lnCʹ_r, lnCʹ_i = Cʹ_initialization(n, m, k)
  D_init = np.random.uniform(low=-0.001, high=0.001, size=[k,m]).astype(np.float32)
  D = tf.Variable(D_init, name="D", dtype=tf.float32)
  Dₒ_init = np.random.uniform(low=-0.0000001, high=0.0000001, size=[m]).astype(np.float32)
  Dₒ = tf.Variable(Dₒ_init, name="D0", dtype=tf.float32)

  return θ, lnCʹ_r, lnCʹ_i, D, Dₒ

def unitary_constitute(θ, lnCʹ_r, lnCʹ_i, D, Dₒ):
  k,half_n = θ.shape
  lnλ_r = tf.zeros((k,half_n*2), dtype=tf.float32)
  lnλ_i = tf.concat([θ, -θ], axis=1)
  lnλ = tf.complex(lnλ_r, lnλ_i)
  λ = tf.exp(lnλ)

  #Cʹ = Cʹ_constitute(lnλ, λ, C)
  Cʹ = Cʹ_constitute(lnCʹ_r, lnCʹ_i)

  return lnλ, λ, Cʹ, D, Dₒ


# Initialization as roots of monic polynomial with random coefficients
def randroot_initialization(n, m, k, stable=True):
  λ_init = np.zeros((k,n), dtype=np.complex64)
  for i in np.arange(k):
    Ainit = np.diagflat(np.ones(shape=n-1), 1).astype(np.float32)
    Ainit[-1,:] = np.random.normal(size=n) / n
    λ_init[i] = np.linalg.eigvals(Ainit)

  if stable:
    λ_init = λ_init / np.maximum(1.0, np.abs(λ_init))

  return fixed_initialization(n, m, k, λ_init)

def fixed_initialization(n, m, k, λ_init, C_init=None, D_init=None, Dₒ_init=None):
  λ_init = λ_init.flatten()
  
  # isolated (unpaired) log eigenvalue ends up real if imaginary part is either 0 or pi 
  # only optimize over real part, fix the imaginary part. this ensures eigenvalue is real, but fixes its sign
  # (which is OK, because it shouldn't approach zero anyway.)
  lnλ_real_init = np.log(λ_init[np.isreal(λ_init)] + 0j)
  lnλ_real_r = tf.Variable(lnλ_real_init.real, name="ln_eig_real_r", dtype=tf.float32)

  comp_pair = (λ_init[np.iscomplex(λ_init)])[::2] # only get one part of conjugate pair.
  ln_comp_pair = np.log(comp_pair)
  lnλ_comp_init_a = ln_comp_pair.real.astype(np.float32)
  lnλ_comp_init_b = ln_comp_pair.imag.astype(np.float32)
            
  lnλ_comp_a = tf.Variable(lnλ_comp_init_a, name="ln_eig_comp_a", dtype=tf.float32)
  lnλ_comp_b = tf.Variable(lnλ_comp_init_b, name="ln_eig_comp_b", dtype=tf.float32)
  
  where_λ_init_r = np.argwhere(np.isreal(λ_init))
  where_λ_init_i = np.argwhere(np.iscomplex(λ_init))
  

  lnCʹ_r, lnCʹ_i = Cʹ_initialization(n, m, k)
  if D_init is None:
    D_init = np.random.uniform(low=-0.001, high=0.001, size=[k,m]).astype(np.float32) / m
  D = tf.Variable(D_init, name="D", dtype=tf.float32)
  if Dₒ_init is None:
    Dₒ_init = 0.0
  Dₒ = tf.Variable(Dₒ_init, name="D0", dtype=tf.float32)  

  return lnλ_real_r, lnλ_real_init.imag, lnλ_comp_a, lnλ_comp_b, where_λ_init_r, where_λ_init_i, lnCʹ_r, lnCʹ_i, D, Dₒ

def fixed_constitute(lnλ_real_r, lnλ_real_i_init, lnλ_comp_a, lnλ_comp_b, where_λ_init_r, where_λ_init_i, lnCʹ_r, lnCʹ_i, D, Dₒ):
  k,_,n = lnCʹ_r.shape
  lnλ_real = tf.complex(lnλ_real_r, lnλ_real_i_init)

  # Keep conjugate pairs adjacent [-b_1, b_1, -b_2, b_2, ...]
  lnλ_comp_r = tf.repeat(lnλ_comp_a, 2, axis=0)
  lnλ_comp_i = tf.repeat(lnλ_comp_b, 2, axis=0) * np.tile([-1,1], lnλ_comp_b.shape[0])
  lnλ_comp = tf.complex(lnλ_comp_r, lnλ_comp_i)
  # restore original order of eigenvalues
  lnλ = tf.scatter_nd(
              np.concatenate((where_λ_init_r, where_λ_init_i)),
              tf.concat([lnλ_real, lnλ_comp], axis=0),
              [k*n])    
  lnλ = tf.reshape(lnλ, [k,n])
  λ = tf.exp(lnλ)

  #Cʹ = Cʹ_constitute(lnλ, λ, C)
  Cʹ = Cʹ_constitute(lnCʹ_r, lnCʹ_i)

  return lnλ, λ, Cʹ, D, Dₒ

# Expands optimization over real C to complex Cʹ.
# After optimization, CU ≈ Cʹ with equivalent loss can be found 
# Furthermore, optimize over lnCʹ for numerical reasons
# (typically, C is very close to 0)
# Uses lnλ, λ for initial value 
def Cʹ_initialization(n, m, k):
  lnCʹ_r = tf.Variable(tf.random.normal((k,m,n), stddev=0.0001), name="lnC'_r", dtype=tf.float32)
  lnCʹ_i = tf.Variable(tf.random.normal((k,m,n), stddev=0.0001), name="lnC'_i", dtype=tf.float32)
  return lnCʹ_r, lnCʹ_i

def Cʹ_constitute(lnCʹ_r, lnCʹ_i):
  lnCʹ = tf.complex(lnCʹ_r, lnCʹ_i)
  Cʹ = tf.exp(lnCʹ)
  return Cʹ

#def C_initialization(n, m, k):
#  C = tf.Variable(tf.random.normal((k,m,n), stddev=0.00001), name="C", dtype=tf.float32)
#  return C
#
#def Cʹ_constitute(lnλ, λ, C):
#  return computeCʹ(lnλ, λ, C)  

# Careful computation of numerically unstable quantities
def computeBʹ(lnλ, λ):
  k, n = lnλ.shape
  
  # ratios[k,i,j] = λi / λj for k'th λ
  ratios = tf.reshape(λ, (k, -1, 1)) / tf.reshape(λ, (k, 1, -1))
  #ratios = tf.exp(tf.reshape(lnλ, (k, -1, 1)) - tf.reshape(lnλ, (k, 1, -1)))
  ratios = tf.linalg.set_diag(ratios, tf.zeros(shape=[k,n], dtype=tf.complex64))

  # Bʹ_i = λi^{n-1} / ∏_{i≠j} λi-λj
  # log Bʹ_i 
  # = (n-1) logλi - ∑_{i≠j} log(λi-λj)
  # = (n-1) logλi - ∑_{i≠j} logλi + log(1 - λj/λi)
  # = -∑_{i≠j} log(1 - λj/λi)
  # = -∑_j log(1 - ratios[k,j,i]) 
  # because log(1 - ratios[k,i,i]) = 0
  lnBʹ = -1* tf.reduce_sum(tf.math.log(1.0 - ratios), axis=1)
  # Bʹ : [k, n]
  Bʹ = tf.exp(lnBʹ)
  return Bʹ


def computeCʹ(lnλ, λ, C, first_method=False): 
  n = λ.shape[-1]
  if first_method:
    # Avoids taking log(C), but does need to take log(C·λⁱ)
    λⁱ = tf.expand_dims(tf.math.pow(tf.expand_dims(λ, -1), tf.cast(tf.range(1,n+1), dtype=tf.complex64)), 0) # [1, k, n, n]  
    C = tf.expand_dims(tf.transpose(tf.cast(C, tf.complex64), (1,0,2)), 2) # [m, k, 1, n] 
    C·λⁱ = tf.reduce_sum(C*λⁱ, axis=-1) # [m, k, n]  
    Cʹ = tf.exp(tf.cast(-n, tf.complex64)*tf.expand_dims(lnλ, 0) + tf.math.log(C·λⁱ)) # [1,k,n]+[m,k,n]
    Cʹ = tf.transpose(Cʹ, (1,2,0)) # [k, n, m]
  else:
    # Takes log(C)
    lnC = tf.transpose(tf.math.log(tf.complex(C, 0.0)), (1,0,2)) # [m, k, n]
    p = tf.cast(n - tf.range(1,n+1), tf.complex64)
    lnU = tf.reshape(-1*p, (1,-1,1)) * tf.expand_dims(lnλ, 1) # [1,n,1]*[k,1,n] -> [k, n, n] 
    lnC_lnU = tf.expand_dims(lnC,-1) + tf.expand_dims(lnU, 0) # [m,k,n,1]+[1,k,n,n] -> [m, k, n, n]
    Cʹ = tf.reduce_sum(tf.exp(lnC_lnU), axis=-2) # [m, k, n]
    Cʹ = tf.transpose(Cʹ, (1,2,0)) # [k, n, m]
  tf.debugging.check_numerics(tf.math.real(Cʹ), message="C'")
  return Cʹ

# Reciprocal square root nonlinearity
def recipsq(a):
  return tf.complex(tf.math.rsqrt(1 + tf.math.square(tf.abs(a))), 0.0)
