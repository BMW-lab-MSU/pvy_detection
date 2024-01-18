import glob

import numpy as np

from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2

if __name__ == "__main__":
    train_raw_dir = glob.glob(os.path.join(info()['save_dir'], 'train_raw', '*.npy'))
    train_raw_dir.sort()
    train_labels_dir = glob.glob(os.path.join(info()['save_dir'], 'train_labels', '*.npy'))
    train_labels_dir.sort()

    train_raw_dir = train_raw_dir[0: 6]

    train_data = []
    train_label = []

    for count, file in enumerate(train_raw_dir):
        print(count)
        train_data = np.append(train_data, np.load(file).flatten(), axis=0)
        train_label = np.append(train_label, np.load(train_labels_dir[count]).flatten(), axis=0)

    labels_to_keep = [0, 3, 4, 6]
    train_to_keep_idx = np.where(np.isin(train_label, labels_to_keep))[0]
    # test_to_keep_idx = np.where(np.isin(test_label, labels_to_keep))[0]

    y_train = to_categorical(train_label[train_to_keep_idx])
    # y_test = to_categorical(test_label[test_to_keep_idx])

    y_train = y_train[:, labels_to_keep]
    # y_test = y_test[:, labels_to_keep]

    train_data = train_data.reshape((2000 * 900 * len(train_raw_dir), 224))
    # test_data = test_data.reshape((1800000, 224))

    x_train = train_data[train_to_keep_idx, :]
    # x_test = test_data[test_to_keep_idx, :]

    # Check if GPU is available and set TensorFlow to use GPU
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found, using CPU")

    num_labels = 4

    input_size = 224
    # network parameters
    batch_size = 128
    hidden_units = 256
    # dropout = 0.45
    dropout = 0.5  # Slightly increased dropout for regularization

    # model is a 3-layer MLP with ReLU and dropout after each layer
    model = Sequential()

    # model.add(Dense(hidden_units, input_dim=input_size))
    model.add(Dense(hidden_units, input_dim=input_size, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  # Add Batch Normalization
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.1))  # Use Leaky ReLU instead of ReLU
    model.add(Dropout(dropout))

    # model.add(Dense(hidden_units))
    model.add(Dense(2 * hidden_units, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  # Add Batch Normalization
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout))

    model.add(Dense(hidden_units, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.summary()

    # plot_model(model, to_file='mlp-mnist.png', show_shapes=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, batch_size=batch_size)

    del train_data, train_label

    test_raw_dir = glob.glob(os.path.join(info()['save_dir'], 'test_raw', '*.npy'))
    test_raw_dir.sort()
    test_labels_dir = glob.glob(os.path.join(info()['save_dir'], 'test_labels', '*.npy'))
    test_labels_dir.sort()

    test_data = []
    test_label = []

    for count, file in enumerate(test_raw_dir):
        print(count)
        test_data = np.append(test_data, np.load(file).flatten(), axis=0)
        test_label = np.append(test_label, np.load(train_labels_dir[count]).flatten(), axis=0)

    test_to_keep_idx = np.where(np.isin(test_label, labels_to_keep))[0]
    y_test = to_categorical(test_label[test_to_keep_idx])
    y_test = y_test[:, labels_to_keep]
    test_data = test_data.reshape((2000 * 900 * len(test_raw_dir), 224))
    x_test = test_data[test_to_keep_idx, :]

    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))

    # model.add(Dense(hidden_units,
    #                 kernel_regularizer=l2(0.001),
    #                 input_dim=input_size))
