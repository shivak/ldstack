import numpy as np
import tensorflow as tf
from ldstack import ldstack
import sys
sys.path.append('.')
from cocob_optimizer import COCOB

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file
import os

def load_data():
  """Loads CIFAR10 dataset.
  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  dirname = 'cifar-10-batches-py'
  origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  path = get_file(
      dirname,
      origin=origin,
      untar=True,
      file_hash=
      '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce')

  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if True: #K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

  x_test = x_test.astype(x_train.dtype)
  y_test = y_test.astype(y_train.dtype)

  return (x_train, y_train), (x_test, y_test)

learning_rate = 1
training_steps = 60000
batch_size = 32
display_every = 500

# Network Parameters
num_input = 3 # CIFAR channels 
timesteps = 32 * 32 # timesteps
num_hidden = 192 # hidden layer num of features
num_classes = 10 # CIFAR total classes
k = 4
Delta = 3
num_layers = 1

def lds(X, Y, scope):
  with tf.variable_scope(scope):
    outs = X
    for i in np.arange(num_layers):
      outs = tf.complex(outs, 0.0)
      outs, _, _ = ldstack(outs, num_hidden, num_classes, k, Delta, scope + str(i))
    outs = outs[-1]
    return trainer(outs, Y, scope, tf.train.AdamOptimizer(0.00005))
def gru(X, Y, scope):
  with tf.variable_scope(scope):
    g = tf.contrib.cudnn_rnn.CudnnGRU(num_layers, num_hidden, kernel_initializer=tf.orthogonal_initializer())
    outs, _ = g(X)
    outs = outs[-1]
    outs = tf.layers.Dense(num_classes)(outs)
    return trainer(outs, Y, scope, tf.train.AdamOptimizer(0.001))
def lstm(X, Y, scope):
  with tf.variable_scope(scope):
    g = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers, num_hidden, kernel_initializer=tf.orthogonal_initializer())
    outs, _ = g(X)
    outs = outs[-1]
    outs = tf.layers.Dense(num_classes)(outs)
    return trainer(outs, Y, scope, tf.train.AdamOptimizer(0.001))    

def trainer(logits, Y, scope, optimizer):
  prediction = tf.nn.softmax(logits)

  Yhot = tf.one_hot(Y, depth=num_classes)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Yhot))
  var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  train_op = optimizer.minimize(loss, var_list=var_list)
        
  correct_pred = tf.equal(tf.argmax(prediction, 1), Y)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  return train_op, loss, accuracy

tf.reset_default_graph()
if True:
    # tf Graph input
    X = tf.placeholder("float32", [timesteps, batch_size, num_input])
    Y = tf.placeholder("int64", [batch_size, 1])
    
    gru_train_op, gru_loss, gru_accuracy = gru(X, Y, "GRU")
    lstm_train_op, lstm_loss, lstm_accuracy = lstm(X, Y, "LSTM")
    lds_train_op, lds_loss, lds_accuracy = lds(X, Y, "LDS")

    init = tf.global_variables_initializer()


def random_batches(images, labels, num_batches):
  num_examples = images.shape[0]
  total_batches = int(np.floor(num_examples / batch_size))
  rand_batches = np.random.choice(total_batches, num_batches, replace=True)
  for s,i in enumerate(rand_batches):
    image_batch = images[i*batch_size:(i+1)*batch_size].reshape((batch_size, timesteps, num_input))
    image_batch = np.transpose(image_batch, (1,0,2))
    label_batch = labels[i*batch_size:(i+1)*batch_size]
    yield image_batch, label_batch

def batched_accuracies(accuracies, images, labels):
  num_batches = 16
  accs = np.zeros(shape=[num_batches, len(accuracies)])
  for i, (test_data, test_label) in enumerate(random_batches(images, labels, num_batches)):
    accs[i] = sess.run(accuracies, feed_dict={X: test_data, Y: test_label})
  return np.mean(accs, axis=0)
              
# Start training
sess = tf.Session()

display_steps = round(training_steps / display_every) + 1
all_train_accs = np.zeros((display_steps, 3))
all_test_accs = np.zeros((display_steps, 3))
display_step = 0

(x_train, y_train), (x_test, y_test) = load_data()
if True:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps+1):
      for batch_x, batch_y in random_batches(x_train, y_train, 1):
        # Run optimization op (backprop)
        sess.run(lds_train_op, feed_dict={X: batch_x, Y: batch_y})
        sess.run(gru_train_op, feed_dict={X: batch_x, Y: batch_y})
        sess.run(lstm_train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_every == 0 or step == 1:
          accuracies = [lds_accuracy, gru_accuracy, lstm_accuracy]
          all_train_accs[display_step] = batched_accuracies(accuracies, x_train, y_train)
          all_test_accs[display_step]  = batched_accuracies(accuracies, x_test, y_test)
          print(step, end=': ')
          for i in range(3):
            print(all_train_accs[display_step,i], all_test_accs[display_step,i], sep=' ', end= ' ')
          print("")
          display_step = display_step + 1
