import os
import tensorflow as tf
from tensorflow.python.framework import ops

dir = os.path.dirname(os.path.abspath(__file__))
_lr_module = tf.load_op_library('%s/../../lib/tf_linear_recurrence.so' % dir)

def linear_recurrence(decays, impulses, initial_state=None):
    '''
    Compute r[i] = decays[i] * r[i - 1] + impulses[i] with r[0] = initial_state.

    decays and impulses must have the same shape and are [n_steps, ...].
    initial_state must be None (to zero initialize) or [...]
    '''
    
    if initial_state is None:
        initial_state = tf.zeros_like(impulses[0, :])

    shape = decays.shape
    rank = len(shape)
    if rank > 2:
        tail = tf.reduce_prod(shape[1:])
        decays = tf.reshape(decays, [shape[0], tail])
        impulses = tf.reshape(impulses, [shape[0], tail])
        initial_state = tf.reshape(initial_state, [tail])

    resp = _lr_module.linear_recurrence(decays, impulses, initial_state)

    if rank > 2:
        resp = tf.reshape(resp, shape)
    return resp

def sparse_linear_recurrence(decay, scales, impulse, init_state=None, serial=False):
    '''
    Computes h[T] where
             h[t,a,j,i] = decay[j,i] * h[t-1,a,j,i] + scales[t,a,j]*impulse[j,i] with h[0,a,j,i] = init_state[j,i].
    decay         : [k, n]
    scales        : [T, b, k]
    impulse       : [k, n]
    init_state    : [k, n] or None
    
    Returns h[T]  : [b, k, n] 
    '''
    
    if init_state is None:
        init_state = tf.zeros_like(decay)
    
    T,b,k = scales.shape
    n = decay.shape[1]
    decay = tf.tile(decay, (b, 1))
    scales = tf.reshape(scales, (T, b*k))
    impulse = tf.tile(impulse, (b, 1))
    init_state = tf.tile(init_state, (b, 1))

    out = _lr_module.sparse_linear_recurrence(decay, scales, impulse, init_state)
    return tf.reshape(out, (b,k,n))

@ops.RegisterGradient("LinearRecurrence")
def _linear_recurrence_grad(op, dl_dresp):
    decays = op.inputs[0]
    impulses = op.inputs[1]
    initial_state = op.inputs[2]

    n_steps = tf.shape(impulses)[0]

    # forwards goes from h_0 to h_{T-1}
    forwards_tail = linear_recurrence(tf.math.conj(decays), tf.math.conj(impulses), tf.math.conj(initial_state))[:-1, :]
    forwards = tf.concat([tf.expand_dims(tf.math.conj(initial_state), 0), forwards_tail],
                         axis=0)

    reverse = lambda x: tf.reverse(x, axis=[0])

    # recur on
    # decays from T, T-1, ..., 2
    # output gradients from T-1, T-2, ..., 1
    dl_dh_head = reverse(
        linear_recurrence(
            tf.math.conj(reverse(decays)[:-1, :]),
            reverse(dl_dresp)[1:, :],
            dl_dresp[-1, :],
        )
    )

    dl_dh = tf.concat([dl_dh_head, dl_dresp[-1:, :]], axis=0)

    dl_dinit = tf.math.conj(decays[0, :]) * dl_dh[0, :]
    dl_dimpulses = dl_dh
    dl_ddecays = dl_dh * forwards

    return [dl_ddecays, dl_dimpulses, dl_dinit]

# conj(df/dz + dconj(f)/dz)    
@ops.RegisterGradient("SparseLinearRecurrence")
def _sparse_linear_recurrence_grad(op, dL_dhT):
    decay, scales, impulse, init_state = op.inputs
    T = scales.shape[0]
    reverse = lambda x: tf.reverse(x, axis=[0])

    dL_ddecay = dL_dhT * tf.math.conj((T-1)*tf.math.pow(decay, T-1)*init_state + _lr_module.sparse_linear_recurrence( 
        decay, 
        tf.expand_dims(tf.cast(T - tf.range(1, T), tf.complex64), -1)*scales[:-1],
        impulse,
        init_state
    ))
    
    # Warning: TxN 
    ln_decay = tf.math.log(decay)
    pows = tf.cast(T - tf.range(1, T+1), tf.complex64)
    decayⁱ = tf.math.exp(tf.reshape(pows, (T,1,1)) * tf.expand_dims(ln_decay, 0)) 
    dL_dscales = tf.reduce_sum(tf.math.conj(decayⁱ) * tf.math.conj(impulse) * dL_dhT, axis=-1) 
      
    dL_dimpulse = dL_dhT * tf.math.conj(_lr_module.sparse_linear_recurrence(
        decay,
        scales,
        tf.ones(impulse.shape, dtype=tf.complex64),
        init_state
    ))
    dL_dinit = dL_dhT * tf.math.conj(tf.math.pow(decay, T))
    
    return [dL_ddecay, dL_dscales, dL_dimpulse, dL_dinit]
    
    
def linear_recurrence_cpu(f, b, h0=0.0):
    """Compute the linear recurrence using native tf operations
    so that we evaluate without a GPU. We evaluate the recurrence
    which is stepwise h_t = f * h_{t-1} + b, returning all h."""
    fs = tf.unstack(f, axis=0)
    bs = tf.unstack(b, axis=0)
    h = tf.identity(b)

    hs = []
    for index in range(len(bs)):
        to_append = tf.add(tf.multiply(fs[index], (hs[index-1] if index > 0 else h0)), bs[index])
        hs.append(to_append)
    return tf.stack(hs)
    
def sparse_linear_recurrence_cpu(decay, scales, impulse, init_state):
    if init_state is None:
        init_state = tf.zeros_like(decay)
        
    T,b,k = scales.shape
    decay_ = tf.expand_dims(decay, -1)
    # Warning: T*b*k*n
    λⁱ = tf.expand_dims(tf.transpose(tf.math.pow(decay_, tf.cast(T - tf.range(T+1), tf.complex64)), (2,0,1)), 1) 
    λⁱ_scales = tf.expand_dims(impulse, 0) * tf.reduce_sum(λⁱ[1:]*tf.expand_dims(scales, -1), axis=0) 
    return λⁱ[0]*tf.expand_dims(init_state,0) + λⁱ_scales
    
def sparse_linear_recurrence_naive(decay, scales, impulse, init_state):
    T,b,k = scales.shape
    n = decay.shape[1]
    decay = tf.tile(tf.expand_dims(decay, 0), (b, 1, 1))
    impulse = tf.tile(tf.expand_dims(impulse, 0), (b, 1, 1))
    init_state = tf.tile(tf.expand_dims(init_state, 0), (b, 1, 1))
    
    h = init_state
    for t in range(T):
        h = decay*h + impulse*tf.expand_dims(scales[t], -1)
    return h
print("...")