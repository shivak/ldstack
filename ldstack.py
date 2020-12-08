import numpy as np
import tensorflow as tf
from linear_recurrent_net.linear_recurrent_net.tensorflow_binding import linear_recurrence, sparse_linear_recurrence
#from harold import controllability_matrix

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
def log_polar_AB_param(train_ln_r=True, 
                       train_θ=True, 
                       ln_r_init=tf.keras.initializers.RandomUniform(minval=-10,maxval=0), 
                       θ_init=tf.keras.initializers.RandomUniform(-2*np.pi, 2*np.pi)):
  def params(n, m, k):
    if n % 2 != 0:
      raise "n must be even"
    half_n = round(n/2)

    ln_r_val = ln_r_init((k,half_n))
    θ_val = θ_init((k, half_n))
    ln_r = tf.Variable(ln_r_val,  name="eig_radius", dtype=tf.float32, trainable=train_ln_r)
    θ = tf.Variable(θ_val, name="eig_angle",  dtype=tf.float32, trainable=train_θ)

    return (ln_r,θ), log_polar_AB_constitute
  return params
 

def log_polar_AB_constitute(ln_r, θ):
  k, half_n = ln_r.shape
  lnλ_r = tf.concat([ln_r, ln_r], axis=1)
  lnλ_i = tf.concat([θ, -θ], axis=1)
  lnλ = tf.complex(lnλ_r, lnλ_i)
  λ = tf.exp(lnλ)
  return lnλ, λ

def unitary_AB_param():
  return log_polar_AB_param(train_ln_r=False, ln_r_init=tf.keras.initializers.Zeros())


# Let h be relu or similar "hinge" function. The two parameters (α, ω) represent 
# two eigenvalues (either both real, or complex conjugate pairs) as: 
#   α      + h(-ω) i
#   α+h(ω) - h(-ω) i
# ω>0 makes the eigenvalues α and α+ω real
# ω<0 makes the eigenvalues α±ωi conjugate pairs
#
# We can also log space.
# If λ = r exp(iθ) (i.e. has polar coordinates (r, θ)) then ln(λ) = ln(r) + θi
# Also, ln(λ*) = ln(r) - θi.
# Then we can repeat the hinging on ln(r) and θ rather than re(λ) and im(λ)

def hinged_AB_param(λ_init=None, h=tf.keras.activations.relu, log=True): #elu(a, 0.1))):
  def params(n, m, k):
    if n % 2 != 0:
      raise "n must be even"
    half_n = int(n/2)

    if λ_init is None:
      α_init = tf.keras.initializers.RandomUniform(-1, 1)((k,half_n))
      ω_init = tf.keras.initializers.RandomUniform(-0.8, 0.8)((k,half_n))
    else:
      λ_init_val = λ_init(n,k) if callable(λ_init) else λ_init
      if log:
        # hack
        λ_init_val = np.log(np.absolute(λ_init_val)) + 1j*np.angle(λ_init_val)

      α_init = np.zeros((k,half_n), dtype=np.float32)
      ω_init = np.zeros((k,half_n), dtype=np.float32)
      
      for i in np.arange(k):
        λ_init_f = λ_init_val[i]
        # couple of real eigs represented as α and α+ω where ω>0 
        real_λ = np.real(λ_init_f[np.isreal(λ_init_f)])
        real_α = np.minimum(real_λ[::2], real_λ[1::2])
        real_ω = np.maximum(real_λ[::2], real_λ[1::2]) - real_α

        # complex eig pair represented as α ± ωi
        comp_λ_pair = λ_init_f[np.iscomplex(λ_init_f)][::2] # assumes conj pairs are adjacent
        comp_α = np.real(comp_λ_pair)
        comp_ω = -np.abs(np.imag(comp_λ_pair))

        α_init[i] = np.concatenate([real_α, comp_α], axis=0)
        ω_init[i] = np.concatenate([real_ω, comp_ω], axis=0)
              
    α = tf.Variable(α_init, dtype=tf.float32, name="alpha")
    ω = tf.Variable(ω_init, dtype=tf.float32, name="omega")

    return (α,ω,h,log), hinged_AB_constitute
  return params

def hinged_AB_constitute(α, ω, h, log):
  r = tf.concat([α, α + h(ω)], axis=1)
  i = tf.concat([h(-ω), -h(-ω)], axis=1)
  if log:
    lnλ = tf.complex(r, i)
    λ = tf.math.exp(lnλ)
  else:
    λ = tf.complex(r, i)
    lnλ = tf.math.log(λ)
  return lnλ, λ

# Initialization as roots of monic polynomial with random coefficients
def randroot_λ_init(n, k, max_radius=1.0):
  λ_init = np.zeros((k,n), dtype=np.complex64)
  for i in np.arange(k):
    Ainit = np.diagflat(np.ones(shape=n-1), 1).astype(np.float32)
    Ainit[-1,:] = np.random.normal(size=n) / n
    λ_init[i] = np.linalg.eigvals(Ainit)

  if max_radius is not None:
    λ_init = λ_init / np.maximum(max_radius, np.abs(λ_init))

  return λ_init

def safe_randroot_λ_init(n, k, scale=4):  
  return randroot_λ_init(scale*n, k)[:,:n]

# Initialization on Chebyshev nodes
def chebyshev_λ_init(n, k, kind='first'):
  f = np.polynomial.chebyshev.chebpts1 if kind == 'first' else np.polynomial.chebyshev.chebpts2
  p = f(n)
  return np.tile(np.expand_dims(p, 0), (k, 1)).astype(np.complex64)


def log_AB_param(λ_init):
  λ_init_val = λ_init # Python local variable strangeness
  def params(n,m,k):
    λ_init = λ_init_val.flatten()
    
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

    return (lnλ_real_r, lnλ_real_init.imag, lnλ_comp_a, lnλ_comp_b, where_λ_init_r, where_λ_init_i, k, n), log_AB_constitute
  return params

def log_AB_constitute(lnλ_real_r, lnλ_real_i_init, lnλ_comp_a, lnλ_comp_b, where_λ_init_r, where_λ_init_i, k, n):
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

def standard_AB_param(λ_init):
  λ_init_val = λ_init # Python local variable strangeness
  def params(n,m,k):
    λ_init = λ_init_val.flatten()
    
    λ_real_init = λ_init[np.isreal(λ_init)]
    λ_real_r = tf.Variable(λ_real_init.real, name="eig_real_r", dtype=tf.float32)

    comp_pair = (λ_init[np.iscomplex(λ_init)])[::2] # only get one part of conjugate pair.
    λ_comp_init_a = comp_pair.real.astype(np.float32)
    λ_comp_init_b = comp_pair.imag.astype(np.float32)
              
    λ_comp_a = tf.Variable(λ_comp_init_a, name="eig_comp_a", dtype=tf.float32)
    λ_comp_b = tf.Variable(λ_comp_init_b, name="eig_comp_b", dtype=tf.float32)
    
    where_λ_init_r = np.argwhere(np.isreal(λ_init))
    where_λ_init_i = np.argwhere(np.iscomplex(λ_init))

    return (λ_real_r, λ_comp_a, λ_comp_b, where_λ_init_r, where_λ_init_i, k, n), standard_AB_constitute
  return params

def standard_AB_constitute(λ_real_r, λ_comp_a, λ_comp_b, where_λ_init_r, where_λ_init_i, k, n):
  λ_real = tf.complex(λ_real_r, 0.0)

  # Keep conjugate pairs adjacent [-b_1, b_1, -b_2, b_2, ...]
  λ_comp_r = tf.repeat(λ_comp_a, 2, axis=0)
  λ_comp_i = tf.repeat(λ_comp_b, 2, axis=0) * np.tile([-1,1], λ_comp_b.shape[0])
  λ_comp = tf.complex(λ_comp_r, λ_comp_i)
  # restore original order of eigenvalues
  λ = tf.scatter_nd(
              np.concatenate((where_λ_init_r, where_λ_init_i)),
              tf.concat([λ_real, λ_comp], axis=0),
              [k*n])    
  λ = tf.reshape(λ, [k,n])
  lnλ = tf.math.log(λ)

  return lnλ, λ

def canonical_AB_param(a_stddev = 0.1):
  def params(n,m,k):
    a = tf.Variable(tf.random.normal((k,n), stddev=a_stddev) / float(n), name="a", dtype=tf.float32)
    A = np.diagflat(np.ones(shape=n-1), 1)[:-1]
    A = np.tile(A.reshape(1,n-1,n), (k,1,1))
    A = A.astype(np.float32)
    return (a,A), canonical_AB_constitute
  return params

def canonical_AB_constitute(a, A):
  a_ = tf.expand_dims(a, 1)
  A = tf.concat([A, -a_], axis=1)

  λ, _ = tf.linalg.eig(A) #FIXME: double work?
  lnλ = tf.math.log(λ)
  return lnλ, λ

# Used for naive variant, which just has plain real eigenvalues
def naive_AB_param():
  def params(n,m,k):
    λ = tf.Variable(tf.keras.initializers.GlorotNormal()((k,n)), dtype=tf.float32)
    return (λ,), naive_AB_constitute
  return params

def naive_AB_constitute(λ):
  λ = tf.complex(λ, 0.0)
  lnλ = tf.math.log(λ)
  return lnλ, λ

# Optimize over real C, potentially encountering numerical stability
def standard_C_param(C_init=None, C_stddev=0.00001):
  def params(n,m,k):
    C = tf.Variable(tf.random.normal((k,n,m), stddev=C_stddev), name="C", dtype=tf.float32)
    return (C,), standard_C_constitute
  return params

def standard_C_constitute(lnλ, λ, C):
  return C, None


# Expands optimization over real C to complex Cʹ.
# After optimization, CU ≈ Cʹ with equivalent loss can be found 
# Furthermore, optimize over lnCʹ for numerical reasons
# (typically, C is very close to 0)
# Uses lnλ, λ for initial value

def relaxed_C_param(Cʹ_mean=0.0, Cʹ_stddev=1):
  def params(n,m,k):
    Cʹ_r = tf.Variable(tf.random.normal((k,n,m), mean=Cʹ_mean, stddev=Cʹ_stddev), name="C'_r", dtype=tf.float32)
    Cʹ_i = tf.Variable(tf.random.normal((k,n,m), mean=Cʹ_mean, stddev=Cʹ_stddev), name="C'_i", dtype=tf.float32)
    return (Cʹ_r, Cʹ_i), relaxed_C_constitute
  return params

def relaxed_C_constitute(lnλ, λ, Cʹ_r, Cʹ_i):
  Cʹ = tf.complex(Cʹ_r, Cʹ_i)
  return None, Cʹ

def relaxed_log_C_param(Cʹ_mean=1.0, Cʹ_stddev=0.01):
  def params(n,m,k):
    lnCʹ_init = tf.math.log(tf.cast(tf.random.normal((k,n,m), mean=Cʹ_mean, stddev=Cʹ_stddev), tf.complex64))
    lnCʹ_r = tf.Variable(tf.math.real(lnCʹ_init), name="lnC'_r", dtype=tf.float32)
    lnCʹ_i = tf.Variable(tf.math.imag(lnCʹ_init), name="lnC'_i", dtype=tf.float32)
    return (lnCʹ_r, lnCʹ_i), relaxed_log_C_constitute
  return params

def relaxed_log_C_constitute(lnλ, λ, lnCʹ_r, lnCʹ_i):
  lnCʹ = tf.complex(lnCʹ_r, lnCʹ_i)
  Cʹ = tf.exp(lnCʹ)
  return None, Cʹ

def standard_D_param(D_init=None, Dₒ_init=None, useD=True):
  def params(n,m,d):
    if useD:
      D = tf.Variable(
          np.random.uniform(low=-0.001, high=0.001, size=[d,m]).astype(np.float32) / m if D_init is None else D_init,
          name="D", dtype=tf.float32)
      Dₒ = tf.Variable(
          0.0 if Dₒ_init is None else Dₒ_init, 
          name="D0", dtype=tf.float32)  
    else:
      D = tf.zeros((d,m), dtype=tf.float32)
      Dₒ = 0.0
    return (D, Dₒ), lambda D, Dₒ: (D,Dₒ)
  return params

  
# Batch of d SIMO LDS, only returning last state. Uses much more memory-efficient SparseLinearRecurrence op
class SparseLDS(tf.keras.layers.Layer):
  def __init__(self, n, m, AB_param, C_param, D_param, standard=False):
    super(SparseLDS, self).__init__()
    self.lnλ_λ_underlying, self.constitute_lnλ_λ = AB_param(n, m, k)
    self.Cʹ_underlying, self.constitute_Cʹ = C_param(n, m, k)
    self.D_Dₒ_underlying, self.constitute_D_Dₒ = D_param(n, m, k)
    self.d = 1
    self.n = n
    self.m = m 
    self.standard = standard

  def call(self, x):
    b, T, d = x.shape
    n = self.n
    lnλ, λ = self.constitute_lnλ_λ(*self.lnλ_λ_underlying)
    _, Cʹ = self.constitute_Cʹ(*((lnλ, λ) + self.Cʹ_underlying))
    D, Dₒ = self.constitute_D_Dₒ(*self.D_Dₒ_underlying)
    Bʹ = tf.ones([1,n], dtype=tf.complex64)

    x = tf.complex(tf.transpose(x, (1,0,2)), 0.0)

    # linear_recurrence computes sʹ_t = λ·sʹ_{t-1} + Bx_t
    # for standard LDS, we need to shift x
    # sʹ_t = λ·sʹ_{t-1} + Bx_{t-1}
    if self.standard:
      x = tf.concat([tf.zeros((1,b,d), dtype=tf.complex64),  x[:-1]], axis=0)
    sₜʹ = sparse_linear_recurrence(λ, x, Bʹ) 
    # sₜʹ: [b, 1, n]
    # Cʹ: [1, n, m]    
    # D : [1, m]
    # [b, 1, n] * [1, n, m] -> [b, m]
    Cʹ·sₜʹ = tf.matmul(tf.squeeze(sₜʹ), tf.squeeze(Cʹ))
    D·xₜ = tf.matmul(x[-1], tf.complex(D, 0.0))
    yₜ = Cʹ·sₜʹ + D·xₜ + tf.complex(Dₒ, 0.0)
    
    return tf.math.real(yₜ)

  # TODO: (A, B, C, D) by computing elementary symmetric polynomials 
  def canonical_lds():
    return None

'''
A MIMO LDS in (perturbed) Luenberger canonical form
n/d SIMO LDS, each of size d, with inputs coupled by a matrix E
'''
class LuenbergerLDS(tf.keras.layers.Layer):
  def __init__(self, n, m, AB_param, C_param, D_param, E_init=None, last_only=False):
    super(LuenbergerLDS, self).__init__()
    self.n = n
    self.m = m

    # Optimization for sparse case
    if last_only:
      self.lds = SparseLDS(n, m, AB_param, C_param, D_param)
    else:
      self.lds = LuenbergerStack(n, self.m, 1, AB_param, C_param, D_param, last_only=last_only)
    self.E = tf.keras.layers.Dense(use_bias=False, kernel_initializer=E_init)

  def build(self, input_shape):
    _, T, d = input_shape
    if self.n % d != 0:
      raise "n (LDS state size) must be divisible by d (input dimension)"  

  def call(self, x):
    E·x = self.E(x)
    return self.lds(E·x)


# Takes [b, T, d] real
# Returns [b, T, m] real
# Note: requires a fixed batch size. This means you must:
# 1. Use an InputLayer in the Keras model specification. 
# 2. Pass a TF dataset which has been batched with drop_remainder=True to model.fit(), 
#    not a NumPy array. (Otherwise the last batch will be the wrong size.)
class LDStack(tf.keras.layers.Layer):
  # n: state size
  # m: output size
  # Δ: depth of stack
  # standard: whether to compute 
  # last_only: return just the last element of the output sequence
  def __init__(self, n, m, Δ, AB_param, C_param, D_param, 
               𝒯_init=tf.keras.initializers.GlorotNormal(), 
               W_init=tf.keras.initializers.GlorotNormal(),
               ρ=tf.keras.activations.elu, 
               mode='standard', 
               last_only=False, 
               relaxV=True,
               r=1):
    super(LDStack, self).__init__()
    self.AB_param = AB_param
    self.C_param = C_param
    self.D_param = D_param
    self.𝒯_init = 𝒯_init
    self.W_init = W_init
    self.n = n
    self.m = m 
    self.Δ = Δ
    if mode != 'standard' and mode != 'adjoint' and mode != 'naive':
      raise ValueError("mode must be standard, adjoint, or naive")
    self.mode = mode
    self.last_only = last_only
    self.relaxV = relaxV
    # TODO: fuse for common activations
    self.𝛿 = lambda a: ρ(a) - a
    self.r = r    
    self.lnλ_λ_underlying, self.constitute_lnλ_λ = self.AB_param(n, m, 1)

  def build(self, input_shape):
    _, _, self.d = input_shape

    # perhaps move to Luenberger? 
    self.C_Cʹ_underlying, self.constitute_C_Cʹ = self.C_param(int(self.n/self.d), self.m, self.d)
    self.D_Dₒ_underlying, self.constitute_D_Dₒ = self.D_param(self.n, self.m, self.d)
    if self.d > 1:
      self.E = tf.Variable(tf.random.normal((self.d,self.r), stddev=1/np.sqrt(self.n)), name="E", dtype=tf.float32, trainable=False)

    if self.mode != 'naive':
      if self.relaxV:
        if self.d == 1:
          self.W_r = tf.Variable(self.W_init((self.n, self.n)), dtype=tf.float32, name="W_r")
          self.W_i = tf.Variable(tf.keras.initializers.Zeros()((self.n, self.n)), dtype=tf.float32, name="W_i") 
        else:
          self.W_r = tf.Variable(self.W_init((self.n, self.n, self.d)), dtype=tf.float32, name="W_r")
          self.W_i = tf.Variable(tf.keras.initializers.Zeros()((self.n, self.n, self.d)), dtype=tf.float32, name="W_i")
      else:
        if self.d == 1:
          self.𝒯_r = tf.Variable(self.𝒯_init((self.n, self.n)), dtype=tf.float32, name="T_r")
          self.𝒯_i = tf.Variable(tf.keras.initializers.Zeros()((self.n, self.n)), dtype=tf.float32, name="T_i")        
          #self.𝒯__ = tf.Variable(self.𝒯_init((self.n, self.n)), dtype=tf.float32, name="T")
        # WEIRD: self.𝒯 and self.T are treated the same by Python
        else:
          self.𝒯_ = tf.Variable(self.𝒯_init((self.n, self.n, self.d)), dtype=tf.float32, name="T")

  # sʹ = Vs in standard mode
  # sʹ = V^{-H}s in adjoint mode
  # sʹ : [T,b,n]
  # λ  : [k,n]
  def s_from_sʹ(self, sʹ, λ):
    # FIXME: eliminate tiling
    # FIXME: k and b
    T,b,r,n = sʹ.shape
    sʹ = tf.reshape(sʹ, (T,b*r,n))    
    λ = tf.tile(λ, (sʹ.shape[1], 1))
    if self.mode == 'standard':
      s = sʹ
      #s = vander_solve(λ, sʹ)
      tf.debugging.check_numerics(tf.math.real(s), "after vander solve")
    elif self.mode == 'adjoint':
      s = vanderh_solve(λ, sʹ)
    elif self.mode == 'naive':
      s = sʹ
    s = tf.reshape(s, (T,b,r,n))  
    return tf.math.real(s)
  def sʹ_from_s(self, s, λ):
    s = tf.complex(s, 0.0)
    T,b,r,n = s.shape
    s = tf.reshape(s, (T,b*r,n))
    # FIXME: eliminate tiling
    λ = tf.tile(λ, (s.shape[1], 1))
    if self.mode == 'standard':
      sʹ = s
      #sʹ = vander_mul(λ, s)
    elif self.mode == 'adjoint':
      sʹ = vanderh_mul(λ, s)
    elif self.mode == 'naive':
      sʹ = s
    sʹ = tf.reshape(sʹ, (T,b,r,n))  
    return sʹ
  
  def 𝒯j(self):
    𝒯_ = tf.complex(self.𝒯_r, self.𝒯_i)[...,None] if self.d == 1 else self.𝒯_
    # [1,1,r,d].[n,n,d,1]. -> [r,n,n]
    𝒯E = tf.matmul(self.E[None,None,...], 𝒯_[...,None,:], transpose_a=True)[...,0] if self.d > 1 else 𝒯_
    return tf.transpose(𝒯E, (2,0,1))
  # h = s in naive mode 
  # s = 𝒯h otherwise
  def h_from_s(self, s, avg=False):
    if self.mode == 'naive':
      return s
    else:
      # FIXME: need linear system solving broadcasting
      𝒯jinv = tf.linalg.inv(self.𝒯j())
      # [1,1,r,n,n].[T,b,r,n,1] -> [T,b,r,n]
      H = tf.matmul(𝒯jinv[None,None,...], s[...,None])[...,0]
      return tf.reduce_mean(H, axis=-2) if avg else H
  def s_from_h(self, h):
    if self.mode == 'naive':
      return h
    else:
      #[1,1,r,n,n].[T,b,r,n,1] -> [T,b,r,n]
      return tf.matmul(self.𝒯j()[None,None,...],h[...,None])[...,0]

  def Wj(self):
    W = tf.complex(self.W_r, self.W_i)[...,None] if self.d == 1 else tf.complex(self.W_r, self.W_i)
    WE = tf.matmul(tf.complex(self.E[None,None,...], 0.0), W[...,None], transpose_a=True)[...,0] if self.d > 1 else W
    return tf.transpose(WE, (2,0,1))
  def direct_h_from_sʹ(self, sʹ, avg=False): 
    # FIXME: need linear system solving broadcasting
    Wjinv = tf.linalg.inv(self.Wj())
    # [1,1,r,n,n].[T,b,r,n,1] -> [T,b,r,n]
    H = tf.matmul(Wjinv[None,None,...], sʹ[...,None])[...,0]
    h = tf.reduce_mean(H, axis=-2) if avg else H    
    return tf.math.real(h)
  def direct_sʹ_from_h(self, h):
    return tf.matmul(self.Wj()[None,None,...],tf.complex(h[...,None], 0.0))[...,0]

  # Bʹ = 1s in standard mode
  # Bʹ_i = λi^{n-1} / ∏_{i≠j} λi-λj in adjoint mode
  # Bʹ : [k, n]
  def Bʹ(self, lnλ, λ):
    if self.mode == 'standard':
      return tf.ones_like(λ)
    else:
      k, n = lnλ.shape 
      # ratios[k,i,j] = λi / λj for k'th λ
      ratios = tf.reshape(λ, (k, -1, 1)) / tf.reshape(λ, (k, 1, -1))
      ratios = tf.linalg.set_diag(ratios, tf.zeros(shape=[k,n], dtype=tf.complex64))

      # log Bʹ_i 
      # = (n-1) logλi - ∑_{i≠j} log(λi-λj)
      # = (n-1) logλi - ∑_{i≠j} logλi + log(1 - λj/λi)
      # = -∑_{i≠j} log(1 - λj/λi)
      # = -∑_j log(1 - ratios[k,j,i]) 
      # because log(1 - ratios[k,i,i]) = 0
      lnBʹ = -1* tf.reduce_sum(tf.math.log1p(-ratios), axis=1)
      Bʹ = tf.exp(lnBʹ)
      return Bʹ

  def corrections(self, sʹ, λ, x):
    λ·sʹ_1x = sʹ * λ[0] + x[...,None]
    if self.relaxV:
      A̅h_B̅x = self.direct_h_from_sʹ(λ·sʹ_1x, avg=True)
    else:
      Ah_Bx = self.s_from_sʹ(λ·sʹ_1x, λ)
      A̅h_B̅x = self.h_from_s(Ah_Bx, avg=True)

    # cʹ_t = 𝛿(A̅h_t + B̅x_t)
    # linear_recurrence computes sʹ_t = λ·sʹ_{t-1} + 1x_t + cʹ_t
    # Suppose cʹ_t for t  = {0,...,T-1} in code is actually cʹ_t for t  = {1,...,T} in math
    # sʹ_t for t = {0,...,T-1} in code is actually sʹ_{t+1} for t = {1, ... ,T} in math
    # In math, for t = {1,...,T}
    #   h_{t+1} = Ah_t + Bx_t + c_t
    #   Ah_t + Bx_t = h_{t+1} - c_t
    # Corresponding to code, for t = {1,...,T}:
    #   h_t - c_t
    #A̅h_B̅x = h - (c if c is not None else 0.0)
    k = self.𝛿(A̅h_B̅x)
    #print(
    #  tf.reduce_mean(tf.linalg.norm(λ·sʹ_1x, axis=-1)),
    #  tf.reduce_mean(tf.linalg.norm(A̅h_B̅x, axis=-1)), 
    #  tf.reduce_mean(tf.linalg.norm(k, axis=-1)), "corrections")    
    #tf.debugging.check_numerics(k, "corrections")
    # [T,b,n] -> [T,b,r,n]
    if self.relaxV:
      kʹ = self.direct_sʹ_from_h(k[...,None,:])
    else:
      kʹ = self.sʹ_from_s(self.s_from_h(k[...,None,:]), λ)
    return kʹ

  # x: [b, T, d] if not time_major, else [T, b, d] 
  # linear_recurrence computes sʹ_t = λ·sʹ_{t-1} + Bʹx_t + cʹ_t
  # currently with Sʹ_0 = 0  
  # Suppose cʹ_t for t  = {0,...,T-1} in code is actually cʹ_t for t  = {1,...,T} in math
  # sʹ_t for t = {0,...,T-1} in code is actually sʹ_{t+1} for t = {1, ... ,T} in math
  def call(self, x):
    b,T,_ = x.shape
    # TODO: eliminate wasteful transpose
    x = tf.transpose(x, (1,0,2))

    if self.mode != 'naive' and self.d > 1:
      # [T,b,1,d].[1,1,d,r] -> [T,b,r]
      E·x = tf.complex(tf.matmul(x[...,None,:], self.E[None,None,...], ), 0.0)[...,0,:]
    else:
      E·x = tf.complex(x, 0.0)

    d, n, m = (self.d, self.n, self.m)
    lnλ, λ = self.constitute_lnλ_λ(*self.lnλ_λ_underlying)
    C, Cʹ = self.constitute_C_Cʹ(*((lnλ, λ) + self.C_Cʹ_underlying))
    D, Dₒ = self.constitute_D_Dₒ(*self.D_Dₒ_underlying)
    Bʹ = self.Bʹ(lnλ, λ)

    # λ  : [1, n]
    # kʹ : [T, b, n] // [T, b, r, n]
    # sʹ : [T, b, n] // [T, b, r, n]
    # y  : [T, b, m] // [T, b, m]
    kʹ=None
    for i in np.arange(self.Δ):
      sʹ = self.run_simo_lds(E·x, lnλ, λ, Bʹ, kʹ) # [T, b, r, n]
      tf.debugging.check_numerics(tf.math.real(sʹ), "SIMO LDS")
      if i < self.Δ - 1:
        kʹ = self.corrections(sʹ, λ, E·x)
    
    # h : [T, b, n]
    # Cʹ : [d, n/d, m] ?? [n, m]
    # D  : [d, m] 
    # [T, b, n, 1] * [1, 1, n, m] -> [T, b, m]
    if self.last_only:
      sʹ = sʹ[None,-1]
      x = x[None,-1]

    if C is not None:
      # [T, b, n]
      if self.relaxV:
        h = self.direct_h_from_sʹ(sʹ, avg=True)
      else: 
        h = self.h_from_s(self.s_from_sʹ(sʹ, λ), avg=True)
      C = tf.reshape(C, (n, m))
      # [T,b,1,n].[1,1,n,m] -> [T,b,m]
      C·h = tf.matmul(h[...,None,:], C[None,None,...])[...,0,:]  
    else:
      sʹ = tf.reduce_mean(sʹ, axis=-2)
      Cʹ = tf.reshape(Cʹ, (n, m))
      # [T,b,1,n].[1,1,n,m] -> [T,b,m]
      C·h = tf.math.real(tf.matmul(sʹ[...,None,:], Cʹ[None,None,...])[...,0,:]) 

    D·x = tf.tensordot(x, D, [[-1], [0]])
    y = C·h + D·x + Dₒ

    # Return to batch-major shape
    return y[0] if self.last_only else tf.transpose(y, (1,0,2))

  # x : [T, b, r]
  # λ : [1, n]
  # kʹ : [T, b, r, n] 
  def run_simo_lds(self, x, lnλ, λ, Bʹ, kʹ=None):
    r, n, m = (self.r, self.n, self.m)
    T,b,_ = x.shape

    # x_kʹ : [T, b, r*n] 
    x_kʹ = tf.repeat(x, n, -1)
    if kʹ is not None:
      x_kʹ += tf.reshape(kʹ, (T,b,r*n))   

    # TODO: replace tiling with more efficient op
    # λ : [T, b, r*n]
    λ = tf.tile(tf.reshape(λ, (1,1,-1)), (T, b, r)) 
    
    # sʹ : [T, b, r*n] to [T, b, r, n]
    sʹ = linear_recurrence(λ, x_kʹ)
    sʹ = tf.reshape(sʹ, (T,b,r,n))
    return sʹ

class LuenbergerStack(LDStack):
  def __init__(self, n, m, Δ, AB_param, C_param, D_param, 
               E_init=tf.keras.initializers.GlorotNormal(), 
               𝒯_init=tf.keras.initializers.Identity(), 
               mode='standard', 
               ρ=tf.keras.activations.elu, 
               last_only=False):
    super(LuenbergerStack, self).__init__(n, m, Δ, AB_param, C_param, D_param, 𝒯_init, ρ, mode, last_only)
    self.E_init = E_init
    
  def build(self, input_shape):
    super().build(input_shape)
    n_d = int(self.n/self.d)
    if self.n % self.d != 0:
      raise "n (LDS state size) must be divisible by d (input dimension)"
    self.lnλ_λ_underlying, self.constitute_lnλ_λ = self.AB_param(n_d, self.m, self.d)
    if self.d > 1:
      self.E = tf.Variable(self.E_init(shape=(self.d,self.d)), name="E", dtype=tf.float32)
    else:
      self.E = tf.ones((1,1), dtype=tf.float32)  

# Reciprocal square root nonlinearity
def recipsq(a):
#  return tf.complex(tf.math.rsqrt(1 + tf.math.square(tf.abs(a))), 0.0)
  return tf.math.rsqrt(1 + a*tf.math.conj(a))

# For 𝒯_init
def random_controllability_matrix(shape):
  n = shape[0]
  A = np.random.normal(size=(n,n))
  B = np.random.normal(size=(n,1))
  return controllability_matrix((A,B))[0]
