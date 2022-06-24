import pickle
import numpy as np

from model import model

import tensorflow as tf
# uncomment this line for tensorflow version 2
# from tensorflow.compat import v1 as tf

def load_test_data():
    fd = open("cifar10_data/test_batch", 'rb')
    dict = pickle.load(fd, encoding='latin1')
    fd.close()

    x_test, y_test = dict['data'], dict['labels']
    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    x_test = np.rollaxis(x_test, 1, 4).astype('float32')
    y_test = np.eye(10)[np.array(y_test)]

    return x_test, y_test


x_test, y_test = load_test_data()
x_test /= 255 # normalize data
print("-> Data is loaded and normalized.")

x, y_true, y_pred, train_mode, _, _ = model()

# accuracy calculation
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))

saver = tf.train.Saver()
sess = tf.Session()

saver.restore(sess, 'trained_model')
print("-> Checkpoints are restored from trained_model")

test_accuracy = sess.run(accuracy, {x: x_test, y_true: y_test, train_mode: False})

print("-> Test set accuracy {:.2f}%".format(test_accuracy*100))

sess.close()