import numpy as np
import tensorflow as tf
from ldstack import ldstack
import sys
sys.path.append('.')
from cocob_optimizer import COCOB

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/fashion/', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)

learning_rate = 0.1
training_steps = 60000
batch_size = 32
display_every = 500
total_batch = int(mnist.train.num_examples / batch_size)
print("Total number of batches:", total_batch)

# Network Parameters
num_input = 1 # MNIST data input (img shape: 28*28)
timesteps = 28 * 28 # timesteps
num_hidden = 100 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)
k = 1
D = 4

# Doesn't work on CUDNN RNNs because they use an opaque buffer for weights
# Use count_params() instead
def num_parameters(scope):
  total_parameters = 0
  for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope):
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      print(variable, shape)
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.value
      total_parameters += variable_parameters
  return total_parameters

def lds(X, Y, scope):
  with tf.variable_scope(scope):
    X_cplx = tf.complex(X, 0.0) #[T, batch_size, 1]
    y, _, _ = ldstack(X_cplx, num_hidden, num_classes, k, D, scope, standard=True)
    y = y[-1]
    print(scope + "parameters: ", num_parameters(scope))
    return trainer(y, Y, scope, COCOB())
def gru(X, Y, scope):
  with tf.variable_scope(scope):
    g = tf.contrib.cudnn_rnn.CudnnGRU(1, num_hidden, kernel_initializer=tf.orthogonal_initializer())
    outs, _ = g(X)
    outs = outs[-1]
    outs = tf.layers.Dense(num_classes)(outs)
    return trainer(outs, Y, scope, tf.train.AdamOptimizer(0.001)), g
def lstm(X, Y, scope):
  with tf.variable_scope(scope):
    g = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_hidden, kernel_initializer=tf.orthogonal_initializer())
    outs, _ = g(X)
    outs = outs[-1]
    outs = tf.layers.Dense(num_classes)(outs)
    return trainer(outs, Y, scope, tf.train.AdamOptimizer(0.001)), g    

def trainer(logits, Y, scope, optimizer):
  prediction = tf.nn.softmax(logits)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  train_op = optimizer.minimize(loss, var_list=var_list)
        
  correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  return train_op, loss, accuracy

tf.reset_default_graph()
if True:
    # tf Graph input
    X = tf.placeholder("float32", [timesteps, batch_size, 1])
    Y = tf.placeholder("float32", [batch_size, num_classes])
    
    (gru_train_op, gru_loss, gru_accuracy), gru_layer = gru(X, Y, "GRU")
    (lstm_train_op, lstm_loss, lstm_accuracy), lstm_layer = lstm(X, Y, "LSTM")
    lds_train_op, lds_loss, lds_accuracy = lds(X, Y, "LDStack")

    init = tf.global_variables_initializer()

    #print("GRU parameters: ", gru_layer.count_params())
    #print("LSTM parameters: ", lstm_layer.count_params())
    print("LDStack parameters: ", num_parameters("LDStack"))

def batched_accuracies(accuracies, data):
  num_batches = 12
  total_batches = int(np.floor(data.num_examples / batch_size))
  rand_batches = np.random.choice(total_batches, num_batches, replace=False)
  accs = np.zeros(shape=[num_batches, len(accuracies)])

  for s,i in enumerate(rand_batches):
    test_data = data.images[i*batch_size:(i+1)*batch_size].reshape((batch_size, timesteps, num_input))
    test_data = np.transpose(test_data, (1,0,2))
    test_label = data.labels[i*batch_size:(i+1)*batch_size]

    accs[s] = sess.run(accuracies, feed_dict={X: test_data, Y: test_label})
  return np.mean(accs, axis=0)
              
# Start training
best_val_acc = 0.8
sess = tf.Session()

display_steps = round(training_steps / display_every) + 1
all_train_accs = np.zeros((display_steps, 3))
all_test_accs = np.zeros((display_steps, 3))
display_step = 0
if True:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        batch_x = np.transpose(batch_x, (1,0,2))
        # Run optimization op (backprop)
        sess.run(lds_train_op, feed_dict={X: batch_x, Y: batch_y})
        sess.run(gru_train_op, feed_dict={X: batch_x, Y: batch_y})
        sess.run(lstm_train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_every == 0 or step == 1:
          accuracies = [lds_accuracy, gru_accuracy, lstm_accuracy]
          all_train_accs[display_step] = batched_accuracies(accuracies, mnist.train)
          all_test_accs[display_step]  = batched_accuracies(accuracies, mnist.test)
          print(step, end=': ')
          for i in range(3):
            print(all_train_accs[display_step,i], all_test_accs[display_step,i], sep=' ', end= ' ')
          print("")
          display_step = display_step + 1
