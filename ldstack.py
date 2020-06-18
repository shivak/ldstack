import numpy as np
import tensorflow as tf
from linear_recurrent_net.linear_recurrent_net.tensorflow_binding import linear_recurrence

class LDStack(tf.keras.layers.Layer):
  # n: state size
  # d: input size
  # m: output size
  # k: number of random projections
  # Δ: depth of stack
  # standard: whether to compute 
  def __init__(self, n, d, m, k, Δ, init='unitary', standard=True):
    super(LDStack, self).__init__()
    
    if d > 1 or k > 1:
      self.R = tf.Variable(np.random.normal(size=(d,k)), dtype=tf.float32, trainable=False)
    
    if init == 'unitary':
      (lnλ, C, D, Dₒ), (lnλ_init, C_init, D_init, Dₒ_init) = self.unitary_initialization(m, n, k)
    elif init == 'randroots':
      (lnλ, C, D, Dₒ), (lnλ_init, C_init, D_init, Dₒ_init) = self.randroot_or_fixed_initialization(m, n, k)
    elif isinstance(init, tuple):
      (λ_init, C_init, D_init, Dₒ_init) = init
      lnλ, C, D, Dₒ = self.randroot_or_fixed_initialization(m, n, k, λ_init, C_init, D_init, Dₒ_init)

    self.lnλ = lnλ
    self.C = C
    self.D = D
    self.Dₒ = Dₒ
    self.k = k
    self.n = n
    self.Δ = Δ
    self.standard = standard

  # Initialize with uniformly random complex eigenvalues of magnitude 1.
  # If λ has polar coordinates (r, θ) then ln(λ) = ln(r) + θi
  # Also, ln(λ*) = ln(r) - θi
  # Fix r=1, i.e. ln(r) = 0. (For numerical reasons, can fix r=1-𝛿 for some small 𝛿) 
  # Note this only requires n/2 parameters rather than n
  # Should constrain -π ≤ θ ≤ π
  def unitary_initialization(self, m, n, k):
    if n % 2 != 0:
      raise "n must be even"

    half_n = round(n/2)
    θ_init = np.random.uniform(low=-np.pi, high=np.pi, size=[k,half_n]).astype(np.float32)
    θ = tf.Variable(θ_init, dtype=tf.float32)
    lnλ_r = tf.zeros((k,n), dtype=tf.float32)
    #lnλ_r = np.log(1-𝛿)*tf.ones((k,n), dtype=tf.float32)
      
    lnλ_i = tf.concat([θ, -θ], axis=1)
    lnλ = tf.complex(lnλ_r, lnλ_i)
    lnλ_init = 0 #FIXME

    C_init = np.random.uniform(low=-0.0001, high=0.0001, size=[k,m,n]).astype(np.float32)
    C = tf.Variable(C_init, dtype=tf.float32)
    D_init = np.random.uniform(low=-0.001, high=0.001, size=[k,m]).astype(np.float32)
    D = tf.Variable(D_init, dtype=tf.float32)
    Dₒ_init = np.random.uniform(low=-0.0000001, high=0.0000001, size=[m]).astype(np.float32)
    Dₒ = tf.Variable(Dₒ_init, dtype=tf.float32)    
      
    return (lnλ, C, D, Dₒ), (lnλ_init, C_init, D_init, Dₒ_init)

  # Initialization as roots of monic polynomial with random coefficients
  def randroot_or_fixed_initialization(self, m, n, k, λ_init=None, C_init=None, D_init=None, Dₒ_init=None):
    if λ_init is None:
      λ_init = np.zeros((k,n), dtype=np.complex64)
      for i in np.arange(k):
        Ainit = np.diagflat(np.ones(shape=n-1), 1).astype(np.float32)
        Ainit[-1,:] = np.random.normal(size=n) / n
        λ_init[i] = np.linalg.eigvals(Ainit)
    λ_init = λ_init.flatten()
    
    # isolated (non paired) log eigenvalue ends up real if imaginary part is either 0 or pi 
    # only optimize over real part, fix the imaginary part. this ensures eigenvalue is real, but fixes its sign
    # (which is OK, because it shouldn't approach zero anyway.)
    lnλ_real_init = np.log(λ_init[np.isreal(λ_init)] + 0j)
    lnλ_real_r = tf.Variable(lnλ_real_init.real, dtype=tf.float32)
    lnλ_real = tf.complex(lnλ_real_r, lnλ_real_init.imag)

    comp_pair = (λ_init[np.iscomplex(λ_init)])[::2] # only get one part of conjugate pair.
    ln_comp_pair = np.log(comp_pair)
    lnλ_comp_init_a = ln_comp_pair.real.astype(np.float32)
    lnλ_comp_init_b = ln_comp_pair.imag.astype(np.float32)
              
    lnλ_comp_a = tf.Variable(lnλ_comp_init_a, dtype=tf.float32)
    lnλ_comp_b = tf.Variable(lnλ_comp_init_b, dtype=tf.float32)
    # Keep conjugate pairs adjacent [-b_1, b_1, -b_2, b_2, ...]
    # tf.repeat() not in 1.13
    lnλ_comp_r = tf.keras.backend.repeat_elements(lnλ_comp_a, 2, axis=0)
    lnλ_comp_i = tf.keras.backend.repeat_elements(lnλ_comp_b, 2, axis=0) * np.tile([-1,1], lnλ_comp_init_b.shape[0])
    lnλ_comp = tf.complex(lnλ_comp_r, lnλ_comp_i)
    # restore original order of eigenvalues
    # TF BUG in scatter_nd() seems to mangle complex values 
    if False:
      lnλ = tf.scatter_nd(
                np.concatenate((np.argwhere(np.isreal(λ_init)), np.argwhere(np.iscomplex(λ_init)))),
                tf.concat([lnλ_real, lnλ_comp], axis=0),
                [k*n])    
    else:
      lnλ_r = tf.scatter_nd(
                  np.concatenate((np.argwhere(np.isreal(λ_init)), np.argwhere(np.iscomplex(λ_init)))),
                  tf.real(tf.concat([lnλ_real, lnλ_comp], axis=0)),
                  [k*n]) 
      lnλ_i = tf.scatter_nd(
                  np.concatenate((np.argwhere(np.isreal(λ_init)), np.argwhere(np.iscomplex(λ_init)))),
                  tf.imag(tf.concat([lnλ_real, lnλ_comp], axis=0)),
                  [k*n]) 
      lnλ = tf.complex(lnλ_r, lnλ_i)
    lnλ = tf.reshape(lnλ, [k,n])

    if C_init is None:
      C_init = np.random.uniform(low=-0.0000001, high=0.0000001, size=[k,m,n]).astype(np.float32)
    C = tf.Variable(C_init, dtype=tf.float32)
    if D_init is None:
      D_init = np.random.uniform(low=-0.0000001, high=0.0000001, size=[k,m]).astype(np.float32)
    D = tf.Variable(D_init, dtype=tf.float32)
    if Dₒ_init is None:
      Dₒ_init = np.random.uniform(low=-0.0000001, high=0.0000001, size=[m]).astype(np.float32)
    Dₒ = tf.Variable(Dₒ_init, dtype=tf.float32)  

    return (lnλ, C, D, Dₒ), (np.log(λ_init), C_init, D_init, Dₒ_init) 

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
  def batch_simo_lds(self, x, lnλ, C, D, Dₒ, α=None, standard=False):
    k, n = (self.k, self.n)
    T, b, d = x.shape
    m = C.shape[1]

    # ratios[k,i,j] = λi / λj for k'th λ
    λ = tf.exp(lnλ)
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
    lnBʹ = -1* tf.reduce_sum(tf.log(1.0 - ratios), axis=1)
    # Bʹ : [k, n]
    # Bʹ·x : [T, b, k*n]
    Bʹ = tf.exp(lnBʹ)
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
      
    # U_{i,j} = 1 / λj^{n-i}
    # log U_{i,j} = -(n-i) logλj
    # let C be a row of 
    # log C'_{i,j}
    # = log <C_{i,:}, U_{:,j}>
    # = log sum_k exp(log C_{i,k} + log U_{k,j})
    lnC = tf.transpose(tf.log(tf.complex(C, 0.0)), (1,0,2)) # [m, k, n]
    powers = tf.cast(n - tf.range(1,n+1), tf.complex64)
    lnU = tf.reshape(-1*powers, (1,-1,1)) * tf.expand_dims(lnλ, 1) # [1,n,1]*[k,1,n] -> [k, n, n] 
    sum_logs = tf.expand_dims(lnC,-1) + tf.expand_dims(lnU, 0) # [m,k,n,1]+[1,k,n,n] -> [m, k, n, n]
    Cʹ = tf.reduce_sum(tf.exp(sum_logs), axis=-2) # [m, k, n]
    Cʹ = tf.transpose(Cʹ, (1,2,0)) # [k, n, m]
    
    Cʹ·sʹ = tf.real(tf.reduce_sum(tf.expand_dims(sʹ, -1)*tf.reshape(Cʹ, (1,1,k,n,m)), axis=-2))
    D·x = tf.real(tf.expand_dims(x, -1)) * tf.reshape(D, (1,1,k,m))
    y = Cʹ·sʹ + D·x + Dₒ

    return sʹ, y

  def recipsq(self, a):
    return tf.complex(tf.math.rsqrt(1 + tf.math.square(tf.abs(a))), 0.0)

  # x: [T, batch size, d] is complex (this is unusual, but matches the format of linear_recurrence.
  #       And actually, is faster for GPU computation, and should be the standard for RNNs.)
  # Returns output y of shape [T, batch size, m]
  def call(self, x):
    T, b, d = x.shape
    k, n = (self.k, self.n)

    if d > 1 or k > 1:
      R = tf.cast(self.R, tf.complex64)
      x = tf.tensordot(x, R, [[-1], [0]])
      
    λ = tf.exp(self.lnλ)
    # λ : [k, n]
    # C : [k, m, n]
    # α : [T, b, k, n]
    # sʹ: [T, b, k, n]
    # y : [T, b, k, m]
    α = None
    for i in np.arange(self.Δ):
      sʹ, y = self.batch_simo_lds(x, self.lnλ, self.C, self.D, self.Dₒ, α, self.standard)
      λ·sʹ = tf.reshape(λ, (1,1,k,n)) * sʹ
      α = self.recipsq(λ·sʹ)

    y = tf.reduce_mean(y, axis=2)
    return y

  # TODO: (A, B, C, D) by computing elementary symmetric polynomials 
  def canonical_lds():
    if self.Δ > 1:
      raise "LDStack with Δ > 1 does not represent an LDS"
    return None
