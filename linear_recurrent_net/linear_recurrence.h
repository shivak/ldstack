#include <cuComplex.h>

extern "C" {
void compute_linear_recurrence(
  const cuComplex *decays,          /* n_steps x n_dims row major matrix */
  const cuComplex *impulses,        /* n_steps x n_dims row major matrix */
  const cuComplex *initial_state,   /* n_dims vector */
  cuComplex *out,                   /* n_steps x n_dims row major matrix */
  int n_dims,                                 /* dimensionality of recurrent vector */
  int n_steps                                 /* length of input and output sequences */
);
void compute_serial_linear_recurrence(
  const cuComplex *decays,          /* n_steps x n_dims row major matrix */
  const cuComplex *impulses,        /* n_steps x n_dims row major matrix */
  const cuComplex *initial_state,   /* n_dims vector */
  cuComplex *out,                   /* n_steps x n_dims row major matrix */
  int n_dims,                                 /* dimensionality of recurrent vector */
  int n_steps                                /* length of input and output sequences */
);
}
