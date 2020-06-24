#include <cuComplex.h>
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"

extern "C" {
void compute_linear_recurrence(
  const cuComplex *decays,          /* n_steps x n_dims row major matrix */
  const cuComplex *impulses,        /* n_steps x n_dims row major matrix */
  const cuComplex *initial_state,   /* n_dims vector */
  cuComplex *out,                   /* n_steps x n_dims row major matrix */
  int n_dims,                       /* dimensionality of recurrent vector */
  int n_steps,                      /* length of input and output sequences */
  const Eigen::GpuDevice& d
);
void compute_serial_linear_recurrence(
  const cuComplex *decays,          /* n_steps x n_dims row major matrix */
  const cuComplex *impulses,        /* n_steps x n_dims row major matrix */
  const cuComplex *initial_state,   /* n_dims vector */
  cuComplex *out,                   /* n_steps x n_dims row major matrix */
  int n_dims,                       /* dimensionality of recurrent vector */
  int n_steps,                      /* length of input and output sequences */
  const Eigen::GpuDevice& d
);

void compute_sparse_linear_recurrence(
    cuComplex *decay, 
    cuComplex *scales,
    cuComplex *impulse,
    cuComplex *init_state,
    cuComplex *out,
    int n,
    int K, 
    int N, 
    int T,
    const Eigen::GpuDevice& d);
void compute_serial_sparse_linear_recurrence(
    cuComplex *decay, 
    cuComplex *scales,
    cuComplex *impulse,
    cuComplex *init_state,
    cuComplex *out,
    int n,
    int K, 
    int N, 
    int T,
    const Eigen::GpuDevice& d);    
}
