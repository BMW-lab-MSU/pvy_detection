import xml.etree.ElementTree as ET

import numpy as np
import random
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2

if __name__ == "__main__":
    # load dataset
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Loading training data")
    train_label = np.load(os.path.join(info()['save_dir'], 'labels', '2023_06_29_Potatoes_Pika_L_35-batch-RGB.png.npy'))
    train_data = np.load(os.path.join(info()['save_dir'], 'train_raw', '2023_06_29_Potatoes_Pika_L_35-Bad Band '
                                                                       'Removal.bip.hdr.npy'))

    print("Loading test data")
    test_label = np.load(os.path.join(info()['save_dir'], 'labels', '2023_06_29_Potatoes_Pika_L_34-batch-RGB.png.npy'))
    test_data = np.load(os.path.join(info()['save_dir'], 'test_raw', '2023_06_29_Potatoes_Pika_L_34-Bad Band '
                                                                     'Removal.bip.hdr.npy'))

    train_label = train_label.flatten()
    test_label = test_label.flatten()

    labels_to_keep = [0, 3, 4, 6]
    train_to_keep_idx = np.where(np.isin(train_label, labels_to_keep))[0]
    test_to_keep_idx = np.where(np.isin(test_label, labels_to_keep))[0]

    y_train = to_categorical(train_label[train_to_keep_idx])
    y_test = to_categorical(test_label[test_to_keep_idx])

    y_train = y_train[:, labels_to_keep]
    y_test = y_test[:, labels_to_keep]

    train_data = train_data.reshape((1800000, 224))
    test_data = test_data.reshape((1800000, 224))

    x_train = train_data[train_to_keep_idx, :]
    x_test = test_data[test_to_keep_idx, :]

    # keep same amount of 0, 3, and 6 (should be 0, 1, 3) from training data (4 is infected)
    idx_train_0 = np.where(y_train[:, 0] == 1)[0]
    idx_train_1 = np.where(y_train[:, 1] == 1)[0]
    idx_train_2 = np.where(y_train[:, 2] == 1)[0]
    idx_train_3 = np.where(y_train[:, 3] == 1)[0]

    random.seed(10)
    idx_0 = random.sample(list(idx_train_0), len(idx_train_2))
    idx_1 = random.sample(list(idx_train_1), len(idx_train_2))
    idx_3 = random.sample(list(idx_train_3), len(idx_train_2))
    y_train_idx = np.concatenate([idx_0, idx_1, idx_train_2, idx_3])
    y_train = y_train[y_train_idx]
    x_train = x_train[y_train_idx, :]

    # # count the number of unique train labels
    # unique, counts = np.unique(y_train, return_counts=True)
    # print("Train labels: ", dict(zip(unique, counts)))
    #
    # # count the number of unique test labels
    # unique, counts = np.unique(y_test, return_counts=True)
    # print("\nTest labels: ", dict(zip(unique, counts)))
    #
    # # sample 25 mnist digits from train dataset
    # indexes = np.random.randint(0, x_train.shape[0], size=25)
    # images = x_train[indexes]
    # labels = y_train[indexes]
    #
    # # plot the 25 mnist digits
    # plt.figure(figsize=(5, 5))
    # for i in range(len(indexes)):
    #     plt.subplot(5, 5, i + 1)
    #     image = images[i]
    #     plt.imshow(image, cmap='gray')
    #     plt.axis('off')
    #
    # plt.show()
    # # plt.savefig("mnist-samples.png")
    # # plt.close('all')
    #
    # # compute the number of labels
    # num_labels = len(np.unique(y_train))

    # Check if GPU is available and set TensorFlow to use GPU
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found, using CPU")

    # num_labels = 7
    num_labels = 4

    # # convert to one-hot vector
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    #
    # # image dimensions (assumed square)
    # image_size = x_train.shape[1]
    # input_size = image_size * image_size
    # print(input_size)
    #
    # # resize and normalize
    # x_train = np.reshape(x_train, [-1, input_size])
    # x_train = x_train.astype('float32') / 255
    # x_test = np.reshape(x_test, [-1, input_size])
    # x_test = x_test.astype('float32') / 255

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

    model.fit(x_train, y_train, epochs=50, batch_size=batch_size)

    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))

    # Make predictions using your trained model
    y_pred = model.predict(x_test)

    # Convert predictions and true labels to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    print(conf_matrix)

    # Labeling the axes
    classes = y_true_classes  # Replace with your actual class labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Displaying the values in the cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center')

    # Labeling the plot
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Save the figure
    plt.savefig('confusion_matrix.png')

    # Display the plot (optional)
    plt.show()

    # # Plot the confusion matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
    #             xticklabels=y_pred_classes, yticklabels=y_true_classes)
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    #
    # # Save the figure
    # plt.savefig('confusion_matrix.png')
    #
    # # Display the plot (optional)
    # plt.show()

    # model.add(Dense(hidden_units,
    #                 kernel_regularizer=l2(0.001),
    #                 input_dim=input_size))
