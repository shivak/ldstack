#! /bin/bash
rm -rf lib/

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
CUDA_LIB=/usr/local/cuda/lib64
CUDA_INC=/usr/local/cuda/include 


mkdir lib
nvcc -std=c++11 -c -DNDEBUG --expt-relaxed-constexpr -I /usr/local linear_recurrence.cu.cc -o lib/linear_recurrence.cu.o ${TF_CFLAGS[@]}  -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC # --compiler-options '-fPIC'
#nvcc lib/linear_recurrence.cu.o -shared -o lib/liblinear_recurrence.so ${TF_CFLAGS[@]} --compiler-options '-fPIC'
g++ -std=c++11 -shared -o lib/tf_linear_recurrence.so linear_recurrent_net/tensorflow_binding/linear_recurrence_op.cpp lib/linear_recurrence.cu.o ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I $CUDA_INC -L $CUDA_LIB -fPIC -lcudart
