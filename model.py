import tensorflow as tf
# uncomment two lines for tensorflow version 2
# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()

def model():

  x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
  y_true = tf.placeholder(tf.float32, shape=[None, 10])
  train_mode = tf.placeholder(tf.bool)

  conv_1 = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3], padding="same", kernel_initializer='he_uniform', name='conv_1')
  batchnorm_1 = tf.layers.batch_normalization(conv_1, momentum=0.9, training=train_mode)
  activation_1 = tf.nn.relu(batchnorm_1)
  pooling_1 = tf.layers.max_pooling2d(activation_1, pool_size=[2, 2], strides=2)
  print("conv_1: ", conv_1.shape)
  print("batchnorm_1: ", batchnorm_1.shape)
  print("activation_1: ", activation_1.shape)
  print("pooling_1: ", pooling_1.shape)

  conv_2 = tf.layers.conv2d(pooling_1, filters=64, kernel_size=[3, 3], padding="same", kernel_initializer='he_uniform', name='conv_2')
  batchnorm_2 = tf.layers.batch_normalization(conv_2, momentum=0.9, training=train_mode)
  activation_2 = tf.nn.relu(batchnorm_2)
  pooling_2 = tf.layers.max_pooling2d(activation_2, pool_size=[2, 2], strides=2)
  print("conv_2: ", conv_2.shape)
  print("batchnorm_2: ", batchnorm_2.shape)
  print("activation_2: ", activation_2.shape)
  print("pooling_2: ", pooling_2.shape)

  conv_3 = tf.layers.conv2d(pooling_2, filters=128, kernel_size=[3, 3], padding="same", kernel_initializer='he_uniform', name='conv_3')
  batchnorm_3 = tf.layers.batch_normalization(conv_3, momentum=0.9, training=train_mode)
  activation_3 = tf.nn.relu(batchnorm_3)
  pooling_3 = tf.layers.max_pooling2d(activation_3, pool_size=[2, 2], strides=2)
  print("conv_3: ", conv_3.shape)
  print("batchnorm_3: ", batchnorm_3.shape)
  print("activation_3: ", activation_3.shape)
  print("pooling_3: ", pooling_3.shape)

  flatten = tf.reshape(pooling_3, [-1, 4 * 4 * 128])
  print("flatten: ", flatten.shape)

  full_connected_1 = tf.layers.dense(flatten, units=512, kernel_initializer='he_uniform', name='full_connected_1')
  activation_4 = tf.nn.relu(full_connected_1)
  dropout_4 = tf.layers.dropout(activation_4, rate=0.5, training=train_mode)
  print("full_connected_1: ", full_connected_1.shape)
  print("activation_4: ", activation_3.shape)
  print("dropout_4: ", dropout_4.shape)

  y_pred = tf.layers.dense(dropout_4, units=10)
  print("y_pred: ", y_pred.shape)

  return x, y_true, y_pred, train_mode, dropout_4, flatten