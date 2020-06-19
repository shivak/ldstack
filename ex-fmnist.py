import numpy as np
import tensorflow as tf
from ldstack import LDStack

learning_rate = 0.001
batch_size = 64
num_epochs = 30

# Network Parameters
num_input = 1 
timesteps = 28 * 28 
num_hidden = 128 
num_classes = 10 
k = 1
Delta = 2

tf.enable_eager_execution()
tf.enable_v2_behavior()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(x_train.reshape(-1, timesteps, 1)/255, tf.float32),
   tf.cast(y_train,tf.int64)))
train_dataset = train_dataset.shuffle(3000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(x_test.reshape(-1, timesteps, 1)/255, tf.float32),
   tf.cast(y_test,tf.int64)))
test_dataset = test_dataset.batch(batch_size)


#lds = LDStack(num_hidden, num_input, num_classes, k, Delta)

lds = tf.keras.Sequential([
  tf.keras.layers.CuDNNGRU(num_hidden, kernel_initializer=tf.keras.initializers.Orthogonal()),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.Orthogonal())                           
])

def loss_accuracy(logits, Y):
  prediction = tf.nn.softmax(logits)
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(Y, logits)
  correct_pred = tf.equal(tf.argmax(prediction, 1), Y)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return loss, accuracy

def accuracy(dataset):
  accs = []

  for i, (batch_x, batch_y) in enumerate(dataset):
    #lds_logits = lds(tf.complex(tf.transpose(batch_x, (1,0,2)), 0.0))[-1]
    lds_logits = lds(batch_x)
    _, accuracy = loss_accuracy(lds_logits, batch_y)
    accs.append(accuracy)
  return np.mean(accs, axis=0)
              

# Start training
all_train_accs = np.zeros((num_epochs, 3))
all_test_accs = np.zeros((num_epochs, 3))

optimizer = tf.train.AdamOptimizer(learning_rate)

#lds.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(),
#  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#lds.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test))

def train_step(model, batch_x, batch_y, opt):
  with tf.GradientTape() as tape:
    #lds_logits = lds(tf.complex(tf.transpose(batch_x, (1,0,2)), 0.0))[-1]
    logits = model(batch_x)
    loss, accuracy = loss_accuracy(logits, batch_y)
  #print(lds_loss.numpy(), lds_accuracy.numpy())
  grads = tape.gradient(loss, model.trainable_variables)
  grads, _ = tf.clip_by_global_norm(grads, 1.)
  opt.apply_gradients(zip(grads, model.trainable_variables))

for epoch in np.arange(num_epochs):
  for step, (batch_x, batch_y) in enumerate(train_dataset):
    train_step(lds, batch_x, batch_y, optimizer)
  all_test_accs[epoch]  = accuracy(test_dataset)
  print(epoch, end=': ')
  print(all_test_accs[epoch])
      
