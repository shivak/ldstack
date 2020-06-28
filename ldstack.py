import numpy as np
import tensorflow as tf
from linear_recurrent_net.linear_recurrent_net.tensorflow_binding import linear_recurrence, sparse_linear_recurrence

'''
LDStack is somewhat unusual in that it supports multiple different underlying parameterizations of the 
optimization variables. A simple example is that we can optimize over C as an n-dimensional real vector,
or over C' = CU, an n-dimensional complex vector, or over log C'. These different parameterizations must be 
initialized in different manners. They "reconstitute" the actual quantities used by the model in different
ways, as well. 

Each parameterization is defined by two functions. 
1. An initialization function, which takes options for the parameterization, and 
returns a function which is called once the dimensions of the variables are known. 
That function, in turn, returns the underlying optimization variables, as well as
the reconstitution function, defined next.
Example:
   def unitary_eig_params(trainλ=True, useD=True):
     def params(n,m,k):
       ...
       return underlying, θ
     return params

2. A reconstitution function, which takes the underlying variables and produces the quantities used by 
the model. Example:
   def unitary_eig_constitute(θ):
     ...
     return lnλ, λ
'''

# Initialize with uniformly random complex eigenvalues of magnitude 1.
# If λ has polar coordinates (r, θ) then ln(λ) = ln(r) + θi
# Also, ln(λ*) = ln(r) - θi
# Fix r=1, i.e. ln(r) = 0. (For numerical reasons, can fix r=1-𝛿 for some small 𝛿) 
# Note this only requires n/2 parameters rather than n
# Should constrain -π ≤ θ ≤ π
def unitary_eig_param(trainλ=True):
  def params(n, m, k):
    if n % 2 != 0:
      raise "n must be even"
    half_n = round(n/2)
    θ_init = np.random.uniform(low=-np.pi, high=np.pi, size=[k,half_n]).astype(np.float32)
    θ = tf.Variable(θ_init, name="eig_angle", dtype=tf.float32, trainable=trainλ)
    return (θ,), unitary_eig_constitute
  return params

def unitary_eig_constitute(θ):
  k, half_n = θ.shape
  lnλ_r = tf.zeros((k,half_n*2), dtype=tf.float32)
  lnλ_i = tf.concat([θ, -θ], axis=1)
  lnλ = tf.complex(lnλ_r, lnλ_i)
  λ = tf.exp(lnλ)
  return lnλ, λ

# Initialization as roots of monic polynomial with random coefficients
def randroot_eig_param(stable=True):
  def params(n,m,k):
    λ_init = np.zeros((k,n), dtype=np.complex64)
    for i in np.arange(k):
      Ainit = np.diagflat(np.ones(shape=n-1), 1).astype(np.float32)
      Ainit[-1,:] = np.random.normal(size=n) / n
      λ_init[i] = np.linalg.eigvals(Ainit)

    if stable:
      λ_init = λ_init / np.maximum(1.0, np.abs(λ_init))

    return log_eig_param(λ_init)
  return params


def log_eig_param(λ_init):
  def params(n,m,k):
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

    return (lnλ_real_r, lnλ_real_init.imag, lnλ_comp_a, lnλ_comp_b, where_λ_init_r, where_λ_init_i), log_eig_constitute
  return params

def log_eig_constitute(lnλ_real_r, lnλ_real_i_init, lnλ_comp_a, lnλ_comp_b, where_λ_init_r, where_λ_init_i):
  k,n,_ = lnCʹ_r.shape
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

  return lnλ, λ

def canonical_eig_param(n, m, k, useD=False):
  def params(n,m,k):
    a = tf.Variable(tf.random.normal((k,n)) / float(10000*n), name="a", dtype=tf.float32)
    return (a,), canonical_eig_constitute

def canonical_eig_constitute(a):
  k,n = a.shape
  A = np.diagflat(np.ones(shape=n-1), 1)[:-1]
  A = np.tile(A.reshape(1,n-1,n), (k,1,1))
  A = A.astype(np.float32)
  a_ = tf.expand_dims(a, 1)
  A = tf.concat([A, -a_], axis=1)
  λ = tf.linalg.eigvals(A) #FIXME: double work
  lnλ = tf.math.log(λ)
  return lnλ, λ

# Optimize over real C, potentially encountering numerical stability
def standard_out_param(C_init=None, D_init=None, Dₒ_init=None):
  def params(n,m,k):
    C = tf.Variable(tf.random.normal((k,n,m), stddev=0.00001), name="C", dtype=tf.float32)

    if D_init is None:
      D_init = np.random.uniform(low=-0.001, high=0.001, size=[k,m]).astype(np.float32) / m
    D = tf.Variable(D_init, name="D", dtype=tf.float32)
    if Dₒ_init is None:
      Dₒ_init = 0.0
    Dₒ = tf.Variable(Dₒ_init, name="D0", dtype=tf.float32)  

    return (C, D, Dₒ), standard_out_constitute
  return params

def standard_out_constitute(lnλ, λ, C, D, Dₒ):
  Cʹ = computeCʹ(lnλ, λ, C) 
  return Cʹ, D, Dₒ


# Expands optimization over real C to complex Cʹ.
# After optimization, CU ≈ Cʹ with equivalent loss can be found 
# Furthermore, optimize over lnCʹ for numerical reasons
# (typically, C is very close to 0)
# Uses lnλ, λ for initial value 
def relaxed_out_param(useD=True):
  def params(n,m,k):
    lnCʹ_init = tf.math.log(tf.cast(tf.random.normal((k,n,m), mean=1.0, stddev=0.01), tf.complex64))
    lnCʹ_r = tf.Variable(tf.math.real(lnCʹ_init), name="lnC'_r", dtype=tf.float32)
    lnCʹ_i = tf.Variable(tf.math.imag(lnCʹ_init), name="lnC'_i", dtype=tf.float32)
    
    if useD:
      D_init = np.random.uniform(low=-0.001, high=0.001, size=[k,m]).astype(np.float32)
      Dₒ_init = np.random.uniform(low=-0.0000001, high=0.0000001, size=[m]).astype(np.float32)
    else:
      D_init = np.zeros([k,m], dtype=np.float32)
      Dₒ_init = np.zeros([m], dtype=np.float32)
    D = tf.Variable(D_init, name="D", dtype=tf.float32, trainable=useD)
    Dₒ = tf.Variable(Dₒ_init, name="D0", dtype=tf.float32, trainable=useD) 

    return (lnCʹ_r, lnCʹ_i, D, Dₒ), relaxed_out_constitute
  return params

def relaxed_out_constitute(lnλ, λ, lnCʹ_r, lnCʹ_i, D, Dₒ):
  lnCʹ = tf.complex(lnCʹ_r, lnCʹ_i)
  Cʹ = tf.exp(lnCʹ)
  return Cʹ, D, D

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
  def __init__(self, n, d, m, k=None, Δ=1, eig_param=randroot_eig_param(), out_param=relaxed_out_param(), standard=True, last_only=False, num_stacks=1):
    super(LDStack, self).__init__()
    self.n = n
    self.m = m
    self.k = k
    self.Δ = Δ
    self.eig_param = eig_param
    self.out_param = out_param
    self.standard = standard
    self.average = k is not None
    self.last_only = last_only
    self.num_stacks = num_stacks

  def build(self, input_shape):
    self.b, self.T, d = input_shape
    print(input_shape, "input shape is")
    n, m, k, Δ, eig_param, out_param, standard, average, last_only, num_stacks = (self.n, self.m, self.k, self.Δ, self.eig_param, self.out_param, self.standard, self.average, self.last_only, self.num_stacks) 

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
        self.mid_layers.append( LDStackInner(n, k, k, Δ, eig_param, out_param, average, standard) )
      self.last_layer = LDStackInner(n, m, k, Δ, eig_param, out_param, average, standard, last_only=last_only)

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


def get_init_and_const_funcs(init):
  if init == 'unitary':
    return unitary_initialization, unitary_constitute
  elif init == 'randroot':
    return randroot_initialization, fixed_constitute
  elif init == 'canonical':
    return canonical_initialization, canonical_constitute

# Average of k SIMO LDS, only returning last state. Uses much more memory-efficient SparseLinearRecurrence op
class SparseLDS(tf.keras.layers.Layer):
  def __init__(self, n, m, k, eig_param, out_param, average, standard=True):
    super(SparseLDS, self).__init__()
    self.eig_underlying, self.constitute_lnλ_λ = eig_param(n, m, k)
    self.out_underlying, self.constitute_Cʹ_D_Dₒ = out_param(n, m, k)
    self.k = k
    self.n = n
    self.m = m 
    self.average = average
    self.standard = standard

  def call(self, x):
    T, b, k = x.shape
    n = self.n
    lnλ, λ = self.constitute_lnλ_λ(*self.eig_underlying)
    Cʹ, D, Dₒ = self.constitute_Cʹ_D_Dₒ(*((lnλ, λ) + self.out_underlying))
    Bʹ = computeBʹ(lnλ, λ)

    # linear_recurrence computes sʹ_t = λ·sʹ_{t-1} + Bx_t
    # for standard LDS, we need to shift x
    # sʹ_t = λ·sʹ_{t-1} + Bx_{t-1}
    if self.standard:
      x = tf.concat([tf.zeros((1,b,k), dtype=tf.complex64),  x[:-1]], axis=0)
    # [b,k,n]
    sₜʹ = sparse_linear_recurrence(λ, x, Bʹ) 
    # sₜʹ: [b, k, n]
    # Cʹ: [k, n, m]    
    # [b, k, n, 1] * [1, k, n, m] -> [b, k, m]
    Cʹ·sₜʹ = tf.reduce_sum(tf.expand_dims(sₜʹ, -1)*tf.expand_dims(Cʹ,0), axis=-2)
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
  def __init__(self, n, m, k, Δ, eig_param, out_param, average, standard=True, last_only=False):
    super(LDStackInner, self).__init__()
    self.eig_underlying, self.constitute_lnλ_λ = eig_param(n, m, k)
    self.out_underlying, self.constitute_Cʹ_D_Dₒ = out_param(n, m, k)
    self.n = n
    self.m = m 
    self.k = k
    self.Δ = Δ
    self.average = average
    self.standard = standard
    self.last_only = last_only

  def build(self, input_shape):
    self.T, self.b, _ = input_shape 

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
    k, n, m, T, b = (self.k, self.n, self.m, self.T, self.b)
    states_only = Cʹ is None or D is None or Dₒ is None
    Bʹ = computeBʹ(lnλ, λ)

    # Bʹ : [k, n]  
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
      
    # sʹ : [T,b,k,n]
    # Cʹ : [k,n,m] 
    # [T,b,k,n,1] * [1,1,k,n,m] -> [T,b,k,m]
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
    lnλ, λ = self.constitute_lnλ_λ(*self.eig_underlying)
    Cʹ, D, Dₒ = self.constitute_Cʹ_D_Dₒ(*((lnλ, λ) + self.out_underlying))

    # λ : [k, n]
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


# Careful computation of numerically unstable quantities
def computeBʹ(lnλ, λ):
  k, n = lnλ.shape
  
  # ratios[k,i,j] = λi / λj for k'th λ
  ratios = tf.reshape(λ, (k, -1, 1)) / tf.reshape(λ, (k, 1, -1))
  #ln_ratios = tf.reshape(lnλ, (k, -1, 1)) - tf.reshape(lnλ, (k, 1, -1))
  ratios = tf.linalg.set_diag(ratios, tf.zeros(shape=[k,n], dtype=tf.complex64))

  # Bʹ_i = λi^{n-1} / ∏_{i≠j} λi-λj
  # log Bʹ_i 
  # = (n-1) logλi - ∑_{i≠j} log(λi-λj)
  # = (n-1) logλi - ∑_{i≠j} logλi + log(1 - λj/λi)
  # = -∑_{i≠j} log(1 - λj/λi)
  # = -∑_j log(1 - ratios[k,j,i]) 
  # because log(1 - ratios[k,i,i]) = 0
  lnBʹ = -1* tf.reduce_sum(tf.math.log1p(-ratios), axis=1)
  # Bʹ : [k, n]
  Bʹ = tf.exp(lnBʹ)
  return Bʹ


def computeCʹ(lnλ, λ, C, first_method=False): 
  n = λ.shape[-1]

  if first_method:
    # Avoids taking log(C), but does need to take log(C·λⁱ)
    # BROKEN
    λⁱ = tf.expand_dims(tf.math.pow(tf.expand_dims(λ, -1), tf.cast(tf.range(1,n+1), dtype=tf.complex64)), 0) # [1, k, n, n]  
    C = tf.expand_dims(tf.transpose(tf.cast(C, tf.complex64), (1,0,2)), 2) # [m, k, 1, n] 
    C·λⁱ = tf.reduce_sum(C*λⁱ, axis=-1) # [m, k, n]  
    Cʹ = tf.exp(tf.cast(-n, tf.complex64)*tf.expand_dims(lnλ, 0) + tf.math.log(C·λⁱ)) # [1,k,n]+[m,k,n]
    Cʹ = tf.transpose(Cʹ, (1,2,0)) # [k, n, m]
  else:
    # Takes log(C)
    lnC = tf.transpose(tf.math.log(tf.complex(C, 0.0)), (2,0,1)) # [m, k, n]
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
