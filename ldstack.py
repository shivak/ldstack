import numpy as np
import tensorflow as tf
from linear_recurrent_net.linear_recurrent_net.tensorflow_binding import linear_recurrence

def ldstack_vars(m, n, k, scope, λ_init=None, C_init=None):
  # Initialization as roots of random coefficient polynomial
  if λ_init is None:
    λ_init = np.zeros((k,n), dtype=np.complex64)
    for i in np.arange(k):
      Ainit = np.diagflat(np.ones(shape=n-1), 1).astype(np.float32)
      Ainit[-1,:] = np.random.normal(size=n) / n
      λ_init[i] = np.linalg.eigvals(Ainit)
  λ_init = λ_init.flatten()
  
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    # isolated (non paired) log eigenvalue ends up real if imaginary part is either 0 or pi 
    # only optimize over real part, fix the imaginary part. this ensures eigenvalue is real, but fixes its sign
    # (which is OK, because it shouldn't approach zero anyway.)
    lnλ_real_init = np.log(λ_init[np.isreal(λ_init)] + 0j)
    lnλ_real_r = tf.get_variable("ln_eig_real_r", dtype=tf.float32, initializer=tf.constant(lnλ_real_init.real), trainable=True)
    lnλ_real = tf.complex(lnλ_real_r, lnλ_real_init.imag)

    comp_pair = (λ_init[np.iscomplex(λ_init)])[::2] # only get one part of conjugate pair.
    ln_comp_pair = np.log(comp_pair)
    lnλ_comp_init_a = ln_comp_pair.real.astype(np.float32)
    lnλ_comp_init_b = ln_comp_pair.imag.astype(np.float32)
            
    lnλ_comp_a = tf.get_variable("ln_eig_comp_a", dtype=tf.float32, initializer=tf.constant(lnλ_comp_init_a), trainable=True)
    lnλ_comp_b = tf.get_variable("ln_eig_comp_b", dtype=tf.float32, initializer=tf.constant(lnλ_comp_init_b), trainable=True)
    lnλ_comp_r = tf.concat([lnλ_comp_a, lnλ_comp_a], axis=0)
    lnλ_comp_i = tf.concat([lnλ_comp_b, -1*lnλ_comp_b], axis=0)
    lnλ_comp = tf.complex(lnλ_comp_r, lnλ_comp_i)
    lnλ = tf.concat([lnλ_real, lnλ_comp], axis=0)
    lnλ = tf.reshape(lnλ, [k,n], name="ln_eig")
    

    if C_init is None:
      C_init = np.random.uniform(low=-0.0000001, high=0.0000001, size=[k,m,n]).astype(np.float32)
    C = tf.get_variable("C", dtype=tf.float32, initializer=tf.constant(C_init), trainable=True)
    return lnλ, C, λ_init, C_init 

# x : [T, b, k]
# λ : [k, n]
# C : [k, m, n]
# α : [T, b, k, n]
# Returns:
# sʹ: [T, b, k, n]
# y : [T, b, k, m]
def batch_simo_lds(x, lnλ, C, α=None):
  k = x.shape[2]
  n = tf.shape(lnλ)[1]
  b = x.shape[1]
  T = x.shape[0]
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
  log_Bʹ = -1* tf.reduce_sum(tf.log(1.0 - ratios), axis=1)
  # Bʹ : [k, n]
  # Bʹ·x : [T, b, k*n]
  Bʹ = tf.exp(log_Bʹ)
  Bʹ·x = tf.reshape(tf.expand_dims(x, -1)*Bʹ, (T, b, k*n))

  # α·λ : [T, b, k*n]
  # sʹ : [T, b, k, n]  
  if α is None:
    α·λ =  tf.tile(tf.reshape(λ, (1,1,-1)), (T, b, 1)) 
  else:
    α·λ = α*tf.reshape(λ, (1, 1, k, n))  
    α·λ = tf.reshape(α·λ, (T, b, k*n))
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
  
  y = tf.real(tf.reduce_sum(tf.expand_dims(sʹ, -1)*tf.reshape(Cʹ, (1,1,k,n,m)), axis=-2))
  #y = tf.real(tf.tensordot(sʹ, Cʹ, [[-1], [1]]))
  return sʹ, y

def recipsq(a):
  return tf.math.rsqrt(1 + tf.math.square(a))

# x: [T, batch_size, d] is complex (this is unusual, but matches the format of linear_recurrence.
#       And actually, is faster for GPU computation, and should be the standard for RNNs.)
# returns [T, batch_size, m] and params
# NOTE: does *not* return output corresponding to initial state, so make sure to shift target to y[1:] and output to y[:-1] in loss
def ldstack(x, n, m, k, D, scope, λ_init=None, C_init=None):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    T, b, d = x.shape

    if d > 1 or k > 1:
      R = tf.get_variable("R", dtype=tf.float32, initializer=tf.random_normal_initializer, shape=[d,k], trainable=False)
      R = tf.cast(R, tf.complex64)
      x = tf.tensordot(x, R, [[-1], [0]])
    
    lnλ, C, λ_init, C_init = ldstack_vars(m, n, k, "lds", λ_init=λ_init, C_init=C_init)
    λ = tf.exp(lnλ)
    # λ : [k, n]
    # C : [k, m, n]
    # α : [T, b, k, n]
    # sʹ: [T, b, k, n]
    # y : [T, b, k, m]
    α = None
    for i in np.arange(D):
      sʹ, y = batch_simo_lds(x, lnλ, C, α)
      λ·sʹ = tf.reshape(λ, (1,1,k,n)) * sʹ
      α = recipsq(λ·sʹ)

    y = tf.reduce_mean(y, axis=2)
    return y, (lnλ, C), (λ_init, C_init)