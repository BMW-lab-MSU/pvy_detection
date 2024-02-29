import xml.etree.ElementTree as ET
import glob
import numpy as np
import keras
import random
import cv2
from utils import *
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    ndvi_files = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized_ndvi', '*.hdr'))
    ndvi_files.sort()
    all_labels = glob.glob(os.path.join(info()['save_dir'], 'labels', '*.npy'))
    all_labels.sort()
    raw_files = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized', '*.hdr'))
    raw_files.sort()

    threshold_to_clear_shadow = 0.8
    kernel = np.ones((3, 3), np.uint8)

    train_data = np.empty((0, 223))
    train_labels = np.empty(0)     # there will be 2 labels, 0 and 1

    val_data = train_data
    val_labels = train_labels

    test_data = train_data
    test_labels = train_labels

    random.seed(10)
    idx = list(range(len(all_labels)))
    random.shuffle(idx)
    idx_test = idx[15:]
    idx_val = idx[11:15]

    for count in range(len(ndvi_files)):
        data = np.squeeze(read_hyper(ndvi_files[count])[0])
        data[data <= threshold_to_clear_shadow] = 0
        data_eroded = cv2.erode(data, kernel, iterations=1).flatten()     # this contains weeds and potato foliage

        label = np.load(all_labels[count]).flatten()
        foliage_with_weeds = np.logical_or(data_eroded, label)
        true_indices = np.where(foliage_with_weeds)[0]
        # find the number of weeds per image
        not_potato_indices = np.where(label[true_indices] == 0)[0]
        # print(f"Number of weeds in Image ", count, " is ", num_weeds)
        if np.sum(not_potato_indices) >= 50000:
            not_potato_indices = not_potato_indices[:50000]

        potato_indices = np.where(label != 0)[0]
        # take same amount of potato labels randomly
        random.seed(10)
        potato_indices = random.sample(list(potato_indices), len(not_potato_indices))

        img = read_hyper(raw_files[count])[0].reshape((1800000, 223))
        combined_indices = np.hstack((potato_indices, not_potato_indices))
        random.seed(10)
        np.random.shuffle(combined_indices)
        img = img[combined_indices, :]
        label = label[combined_indices]
        label[label !=0] = 1    # change the label to 1 for potatoes

        if count in idx_val:
            val_data = np.concatenate((val_data, img), axis=0)
            val_labels = np.concatenate((val_labels, label), axis=0)
        elif count in idx_test:
            test_data = np.concatenate((test_data, img), axis=0)
            test_labels = np.concatenate((test_labels, label), axis=0)
        else:
            train_data = np.concatenate((train_data, img), axis=0)
            train_labels = np.concatenate((train_labels, label), axis=0)

    # model params
    input_size = 223
    batch_size = 128
    hidden_units = 256
    dropout = 0.5  # Slightly increased dropout for regularization

    # model is a 3-layer MLP with ReLU and dropout after each layer
    model = Sequential()

    model.add(Dense(hidden_units, input_dim=input_size, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  # Add Batch Normalization
    model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.1))  # Use Leaky ReLU instead of ReLU
    model.add(Dropout(dropout))

    # model.add(Dense(hidden_units))
    model.add(Dense(2 * hidden_units, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  # Add Batch Normalization
    model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout))

    model.add(Dense(hidden_units, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation='softmax'))

    model.summary()

    model.compile(loss='binary_crossentropy',  # categorical_crossentropy, binary_crossentropy
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_data,
              train_labels,
              epochs=30,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(val_data, val_labels))

    model.save(os.path.join(info()['save_dir'], 'model_potato_not_potato_.keras'))
    # model = keras.models.load_model(os.path.join(info()['save_dir'], 'model_potato_not_potato_.keras'))

    del train_data, train_labels, val_data, val_labels

    history = model.history.history

    # Get loss and accuracy
    loss = history['loss']
    accuracy = history['accuracy']
    #
    # Get validation loss and accuracy if available
    val_loss = history.get('val_loss', None)
    val_accuracy = history.get('val_accuracy', None)

    # Plotting loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Training Accuracy')
    if val_accuracy is not None:
        plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(info()['save_dir'], 'train_val_loss_acc_potato_not_potato.png'))
    # plt.show()
    plt.close()

    predicted = model.predict(test_data)

    # Convert predictions and true labels to class labels
    predicted_classes = np.argmax(predicted, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # Create a confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print(conf_matrix)

    classes = ['not_potatoes', 'potatoes']

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(info()['save_dir'], 'test_conf_mat_potato_not_potato.png'))
    # plt.show()
    plt.close()





    # Number of weeds in Image  0  is  570
        # Number of weeds in Image  1  is  1698
        # Number of weeds in Image  2  is  11027
        # Number of weeds in Image  3  is  22821
        # Number of weeds in Image  4  is  37527
        # Number of weeds in Image  5  is  154965
        # Number of weeds in Image  6  is  43372
        # Number of weeds in Image  7  is  104490
        # Number of weeds in Image  8  is  380958
        # Number of weeds in Image  9  is  24743
        # Number of weeds in Image  10  is  7774
        # Number of weeds in Image  11  is  26749
        # Number of weeds in Image  12  is  885
        # Number of weeds in Image  13  is  163477
        # Number of weeds in Image  14  is  157396
        # Number of weeds in Image  15  is  466
        # Number of weeds in Image  16  is  29944
        # Number of weeds in Image  17  is  51040
        # Number of weeds in Image  18  is  100062


