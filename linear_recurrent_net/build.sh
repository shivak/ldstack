#! /bin/bash
rm -rf lib/

mkdir lib
nvcc -c linear_recurrence.cu -o lib/linear_recurrence.o -O3 --compiler-options '-fPIC'
nvcc lib/linear_recurrence.o -shared -o lib/liblinear_recurrence.so --compiler-options '-fPIC'

# building tensorflow op
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
CUDA_LIB=/usr/local/cuda/lib64
CUDA_INC=/usr/local/cuda/include

g++ -std=c++11 -shared -o lib/tf_linear_recurrence.so linear_recurrent_net/tensorflow_binding/linear_recurrence_op.cpp lib/linear_recurrence.o -O3 ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I $CUDA_INC -L $CUDA_LIB -fPIC -lcudart
