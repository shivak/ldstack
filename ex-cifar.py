import numpy as np
import tensorflow as tf
from ldstack import LDStack

learning_rate = 0.1
training_steps = 60000
batch_size = 128
display_every = 500
num_epochs = 30

# Network Parameters
num_input = 3 # CIFAR channels 
timesteps = 32 * 32 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # CIFAR total classes
k = 1
Delta = 1

tf.enable_eager_execution()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(x_train.reshape(-1, 32*32, 3)/255, tf.float32),
   tf.cast(y_train,tf.int64)))
train_dataset = train_dataset.shuffle(1000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(x_test.reshape(-1, 32*32, 3)/255, tf.float32),
   tf.cast(y_test,tf.int64)))
test_dataset = test_dataset.shuffle(1000).batch(batch_size)


#lds = LDStack(num_hidden, num_input, num_classes, k, Delta)

lds = tf.keras.Sequential([
  tf.keras.layers.CuDNNLSTM(num_hidden),
  tf.keras.layers.Dense(num_classes)                           
])

def loss_accuracy(logits, Y):
  prediction = tf.nn.softmax(logits)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y[:,0]))
  correct_pred = tf.equal(tf.argmax(prediction, 1), Y[:,0])
  #print(tf.argmax(prediction, 1))
  #print("true: ", Y[:,0])
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return loss, accuracy

def accuracy(dataset):
  num_batches = 128
  accs = np.zeros(shape=[num_batches, 1])

  for i, (batch_x, batch_y) in enumerate(dataset.take(num_batches)):
    #lds_logits = lds(tf.complex(tf.transpose(batch_x, (1,0,2)), 0.0))[-1]
    lds_logits = lds(batch_x)
    _, accuracy = loss_accuracy(lds_logits, batch_y)
    accs[i, 0] = accuracy
  return np.mean(accs, axis=0)
              
# Start training
all_train_accs = np.zeros((num_epochs, 3))
all_test_accs = np.zeros((num_epochs, 3))

optimizer = tf.train.AdamOptimizer(learning_rate)
for epoch in np.arange(num_epochs):
  all_test_accs[epoch]  = accuracy(test_dataset)
  print(epoch, end=': ')
  print(all_test_accs[epoch])
      
  for step, (batch_x, batch_y) in enumerate(train_dataset):
    with tf.GradientTape() as tape:
      #lds_logits = lds(tf.complex(tf.transpose(batch_x, (1,0,2)), 0.0))[-1]
      lds_logits = lds(batch_x)
      lds_loss, lds_accuracy = loss_accuracy(lds_logits, batch_y)
    #print(lds_loss.numpy(), lds_accuracy.numpy())
    grads = tape.gradient(lds_loss, lds.trainable_variables)
    #print([v.name + ": " + str(tf.linalg.norm(g).numpy()) for g,v in zip(grads, lds.trainable_variables)])
    optimizer.apply_gradients(zip(grads, lds.trainable_variables))
