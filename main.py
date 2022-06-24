from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import pickle
import seaborn
from model import model

import tensorflow as tf
# uncomment this line for tensorflow version 2
# from tensorflow.compat import v1 as tf

# initialization of parameters
max_number_of_epochs = 150
size_of_batch = 32
learning_rate = 0.001
plot_tsne = False
stop_training = False

# load train and validation data
def load_data():
    y = []
    for i in range(1, 6):
        fd_train = open("cifar10_data/data_batch_{}".format(i), 'rb')
        train_dict = pickle.load(fd_train, encoding='latin1')
        fd_train.close()

        if i == 1: x = train_dict['data']
        else: x = np.vstack((x, train_dict['data']))

        y = y + train_dict['labels']

    x = x.reshape((len(x), 3, 32, 32))
    x = np.rollaxis(x, 1, 4).astype('float32')
    y = np.array(y)

    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=5000, random_state=42)
    y_train, y_validation = np.eye(10)[y_train], np.eye(10)[y_validation]

    return x_train, y_train, x_validation, y_validation

# load test data (for tsne plots)
def load_test_data():
    fd = open("cifar10_data/test_batch", 'rb')
    dict = pickle.load(fd, encoding='latin1')
    fd.close()

    x_test, y_test = dict['data'], dict['labels']
    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    x_test = np.rollaxis(x_test, 1, 4).astype('float32')
    y_test = np.eye(10)[np.array(y_test)]

    return x_test, y_test

# created by using:
# https://medium.com/@pslinge144/representation-learning-cifar-10-23b0d9833c40
def plot_tsne(latent, labels, file_name):
    tsne_data = TSNE(n_components=2).fit_transform(latent[0])  # (45000, 2)
    print("-> 2d tsne_data is created.")

    tsne_data = np.vstack((tsne_data.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
    seaborn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, "Dim_1", "Dim_2").add_legend()

    plt.savefig(file_name + '.png')
    print("-> tsne.png saved.")
    plt.show()


def plot_convergence_curves(accuracies_train, losses_train, accuracies_validation, losses_validation):
    accuracies_train, losses_train = np.array(accuracies_train), np.array(losses_train)
    accuracies_validation, losses_validation = np.array(accuracies_validation), np.array(losses_validation)

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlabel('epochs')
    ax1.tick_params(axis='x', which='minor', direction='out', bottom=True, length=10)
    ax2 = ax1.twinx()

    ax1.plot(accuracies_train, label='train accuracy')
    ax1.plot(accuracies_validation, label='validation accuracy')

    ax2.plot(losses_train, color='r', label='train loss')
    ax2.plot(losses_validation, color='g', label='validation loss')

    ax1.set_ylabel("accuracy (%)")
    ax2.set_ylabel("loss")

    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    file_name = 'accuracy_loss_curves.png'
    plt.savefig(file_name)
    plt.show()

# we load training, validation and test data
x_train, y_train, x_validation, y_validation = load_data()
x_test, y_test = load_test_data()

# we normalize training, validation and test data (0-1 range)
x_train /= 255
x_validation /= 255
x_test /= 255

# we augment training data
augumentation = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=True)
augumentation.fit(x_train)

# model
x, y_true, y_pred, train_mode, latent_space_fc1, latent_space_flatten = model()

# loss function
loss = tf.losses.softmax_cross_entropy(y_true, y_pred)

# accuracy calculation
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # added for batchnorm

# optimization functions
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

train_optimizer = tf.group([train_optimizer, update_ops]) # added for batchnorm

print("-> Max number of epochs = ", max_number_of_epochs)
print("-> Size of batch = ", size_of_batch)
print("-> Learning_rate = ", learning_rate)
print("-" * 100)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("-> Session is started.")

train_accuracies_plot, train_losses_plot = [], []
validation_accuracies_plot, validation_losses_plot = [], []

print("-> Training is started.")
for epoch in range(max_number_of_epochs):

    if stop_training == True: break

    number_of_batch = 0
    epoch_accuracy, epoch_loss = 0, 0
    epoch_validation_accuracy, epoch_validation_loss = 0, 0

    for mini_batch_x, mini_batch_y in augumentation.flow(x_train, y_train, batch_size=size_of_batch):
        _, batch_loss = sess.run([train_optimizer, loss],
                                 feed_dict={x: mini_batch_x, y_true: mini_batch_y, train_mode: True})
        batch_accuracy = sess.run(accuracy, feed_dict={x: mini_batch_x, y_true: mini_batch_y, train_mode: False})
        epoch_loss += batch_loss
        epoch_accuracy += batch_accuracy
        number_of_batch += 1

        if number_of_batch >= len(x_train)/size_of_batch:
            break

    epoch_loss /= number_of_batch
    epoch_accuracy /= number_of_batch

    epoch_validation_loss, epoch_validation_accuracy = sess.run([loss, accuracy],
                                                                {x: x_validation, y_true: y_validation,
                                                                 train_mode: False})
    epoch_accuracy *= 100
    epoch_validation_accuracy *= 100

    validation_losses_plot.append(epoch_validation_loss)
    validation_accuracies_plot.append(epoch_validation_accuracy)
    train_losses_plot.append(epoch_loss)
    train_accuracies_plot.append(epoch_accuracy)

    print(
        "Epoch: {} Training loss: {:.2f} - Validation loss: {:.2f} | Training accuracy: {:.2f}% - Validation accuracy: {:.2f}% ".format(
            epoch + 1, epoch_loss, epoch_validation_loss, epoch_accuracy, epoch_validation_accuracy))

    if plot_tsne == True and (epoch == 0 or epoch == 40 or epoch == 80):
        latent_fc1, latent_flatten = sess.run([latent_space_fc1, latent_space_flatten],
                                              {x: x_test, y_true: y_test, train_mode: False})
        plot_tsne(np.array([latent_fc1]), np.array([np.where(r == 1)[0][0] for r in np.array(y_test)]),
                  "fc1_test_tsne_" + str(epoch))
        plot_tsne(np.array([latent_flatten]), np.array([np.where(r == 1)[0][0] for r in np.array(y_test)]),
                  "flatten_test_tsne_" + str(epoch))

    # Early stopping:
    # We stop, if validation accuracy of x^th epoch is not higher than
    #  any validation accuracy in x-20^th, x-19^th, x-18^th,... x-1^th epochs.
    if (len(validation_losses_plot) >= 20 and
        all(i > validation_accuracies_plot[-1] for i in validation_accuracies_plot[-20:-1])):
        print("-> Early stopping is active.")
        print('-> Validation accuracy do not increasing. Stop training.')
        stop_training = True
        break

# since we loaded test data for tsne, we can also get directly
# performance results for test set by uncommenting two lines
# test_accuracy = sess.run(accuracy, {x: x_test, y_true: y_test, train_mode: False})
# print("Test accuracy: {:.3f}%".format(test_accuracy * 100))

# save model
saver = tf.train.Saver()
save_path = saver.save(sess, 'trained_model')
print("-> Model trained_model is saved.")

sess.close()

print("-> Session is ended.")

# plot accuracy and loss values with respect to the epoch numbers
plot_convergence_curves(train_accuracies_plot, train_losses_plot, validation_accuracies_plot, validation_losses_plot)