import numpy as np
import tensorflow as tf
from ldstack import LDStack

lds_learning_rate = 1
gru_learning_rate = 0.001
batch_size = 32
num_epochs = 30

# Network Parameters
num_channels = 1 
timesteps = 28 * 28 
n_lds = 784
n_gru = 256
num_classes = 10 
k = 1
Delta = 1
num_stacks = 1

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, timesteps, num_channels).astype(np.float32)/255
x_test  = x_test.reshape(-1, timesteps, num_channels).astype(np.float32)/255
y_train = y_train.astype(np.int64)
y_test  = y_test.astype(np.int64)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(3000).batch(batch_size, drop_remainder=True)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size, drop_remainder=True)


lds = tf.keras.Sequential([
  tf.keras.layers.InputLayer((timesteps, num_channels), batch_size=batch_size),
  #LDStack(32, 1, 10, 1, 4, init=randroot_or_fixed_initialization, num_stacks=1, last_only=True)#,
  LDStack(n_lds, num_channels, num_classes, k, Delta, init=randroot_or_fixed_initialization, last_only=True, num_stacks=num_stacks)
])
lds_optimizer = tf.keras.optimizers.Adadelta(lds_learning_rate)
print(lds.summary())

gru = tf.keras.Sequential([
  tf.compat.v1.keras.layers.CuDNNGRU(n_gru, kernel_initializer=tf.keras.initializers.Orthogonal()),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.Orthogonal())                           
])
gru_optimizer = tf.keras.optimizers.Adam(gru_learning_rate)
gru.build(input_shape=(batch_size, timesteps, num_channels))
print(gru.summary())

print("Learning LDStack")
lds.compile(optimizer=lds_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
lds.fit(train_dataset, batch_size=batch_size, epochs=num_epochs, validation_data=test_dataset)

print("Learning GRU")
gru.compile(optimizer=gru_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
gru.fit(train_dataset, batch_size=batch_size, epochs=num_epochs, validation_data=test_dataset)