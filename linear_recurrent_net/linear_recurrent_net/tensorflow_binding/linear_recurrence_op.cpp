#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "../../linear_recurrence.h"
#include <cuComplex.h>

using namespace tensorflow;

REGISTER_OP("LinearRecurrence")
    .Input("decays: complex64")
    .Input("impulses: complex64")
    .Input("initial_state: complex64")
    .Output("response: complex64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });

class GpuLinearRecurrenceOp : public OpKernel {
public:
  explicit GpuLinearRecurrenceOp(OpKernelConstruction *ctx): OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor& decays_tensor = ctx->input(0);
    const Tensor& impulses_tensor = ctx->input(1);
    const Tensor& initial_state_tensor = ctx->input(2);

    int n_steps = impulses_tensor.dim_size(0);
    int n_dims = impulses_tensor.dim_size(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(impulses_tensor.shape()),
		errors::InvalidArgument("Impulses must be a matrix"));


    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(decays_tensor.shape()),
		errors::InvalidArgument("Decays must be a matrix"));

    OP_REQUIRES(ctx,
		decays_tensor.dim_size(0) == n_steps &&
		decays_tensor.dim_size(1) == n_dims,
		errors::InvalidArgument("Decay shape must match impulse shape"));

    OP_REQUIRES(ctx,
		TensorShapeUtils::IsVector(initial_state_tensor.shape()) &&
		initial_state_tensor.dim_size(0) == n_dims,
		errors::InvalidArgument("Initial state must be a vector of length n_dims"));

    Tensor *response_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, impulses_tensor.shape(), &response_tensor));

    auto decays = decays_tensor.flat<std::complex<float>>().data();
    auto impulses = impulses_tensor.flat<std::complex<float>>().data();
    auto initial_state = initial_state_tensor.flat<std::complex<float>>().data();
    auto response = response_tensor->flat<std::complex<float>>().data();

    /* Layout purposefully matches, so this isn't just wishful thinking */
    cuComplex *cu_decays = (cuComplex*) decays;
    cuComplex *cu_impulses = (cuComplex*) impulses;
    cuComplex *cu_initial_state = (cuComplex*) initial_state;
    cuComplex *cu_response = (cuComplex*) response;

    compute_linear_recurrence(cu_decays, cu_impulses,
			      cu_initial_state, cu_response,
			      n_dims, n_steps, ctx->eigen_device<Eigen::GpuDevice>());
  }
};
REGISTER_KERNEL_BUILDER(Name("LinearRecurrence").Device(DEVICE_GPU), GpuLinearRecurrenceOp);

/*
 * Sparse Linear Recurrence
 *
 */
 
REGISTER_OP("SparseLinearRecurrence")
    .Input("decay: complex64")
    .Input("scales: complex64")
    .Input("impulse: complex64")
    .Input("init_state: complex64")
    .Output("response: complex64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

class GpuSparseLinearRecurrenceOp : public OpKernel {
public:
  explicit GpuSparseLinearRecurrenceOp(OpKernelConstruction *ctx): OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor& decay_tensor = ctx->input(0);
    const Tensor& scales_tensor = ctx->input(1);
    const Tensor& impulse_tensor = ctx->input(2);
    const Tensor& init_state_tensor = ctx->input(3);

    int T = scales_tensor.dim_size(0);
    int K = scales_tensor.dim_size(1);
    int n = decay_tensor.dim_size(1);
    int N = K*n; 
    
    OP_REQUIRES(ctx,
        decay_tensor.dim_size(0) == K &&
		impulse_tensor.dim_size(0) == K &&
		init_state_tensor.dim_size(0) == K &&
		impulse_tensor.dim_size(1) == n &&
		init_state_tensor.dim_size(1) == n,		
		errors::InvalidArgument("Decay, impulse, and init_state vectors must have same shape"));

    Tensor *response_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, decay_tensor.shape(), &response_tensor));

    auto decay = decay_tensor.flat<std::complex<float>>().data();
    auto scales = scales_tensor.flat<std::complex<float>>().data();
    auto impulse = impulse_tensor.flat<std::complex<float>>().data();
    auto init_state = init_state_tensor.flat<std::complex<float>>().data();
    auto response = response_tensor->flat<std::complex<float>>().data();

    /* Layout purposefully matches, so this isn't just wishful thinking */
    cuComplex *cu_decay = (cuComplex*) decay;
    cuComplex *cu_scales = (cuComplex*) scales;
    cuComplex *cu_impulse = (cuComplex*) impulse;
    cuComplex *cu_init_state = (cuComplex*) init_state;
    cuComplex *cu_response = (cuComplex*) response;

    compute_sparse_linear_recurrence(
        cu_decay, 
        cu_scales,
        cu_impulse,
		cu_init_state,
		cu_response,
		n, K, N, T, 
		ctx->eigen_device<Eigen::GpuDevice>());
  }
};
REGISTER_KERNEL_BUILDER(Name("SparseLinearRecurrence").Device(DEVICE_GPU), GpuSparseLinearRecurrenceOp);