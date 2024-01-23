import os
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import read_hyper, info
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    idx = list(range(10, 175)) + list(range(185, 244))

    print('Loading First Derivative of Train Raw Data')
    raw_ddx_1 = os.path.join(info()['general_dir'], 'first_derivative', '2023_06_29_Potatoes_Pika_L_35-Reflectance '
                                                                        'from Radiance Data and Downwelling '
                                                                        'Irradiance Spectrum-First Derivative.bip.hdr')
    img = read_hyper(raw_ddx_1)[0]

    # remove the unwanted bands
    img = img[:, :, idx]

    print('Loading Second Derivative of Train Raw Data')
    raw_ddx_1 = os.path.join(info()['general_dir'], 'second_derivative', '2023_06_29_Potatoes_Pika_L_35-Reflectance '
                                                                         'from Radiance Data and Downwelling '
                                                                         'Irradiance Spectrum-Second Derivative.bip.hdr')
    img2 = read_hyper(raw_ddx_1)[0]

    # remove the unwanted bands
    img2 = img2[:, :, idx]

    img = np.concatenate((img, img2), axis=2)

    print("Loading training data")
    train_label = np.load(os.path.join(info()['save_dir'], 'labels', '2023_06_29_Potatoes_Pika_L_35-batch-RGB.png.npy'))
    train_data = np.load(os.path.join(info()['save_dir'], 'train_raw', '2023_06_29_Potatoes_Pika_L_35-Bad Band '
                                                                       'Removal.bip.hdr.npy'))

    train_data = np.concatenate((train_data, img), axis=2)

    # TEST DATA
    print('Loading First Derivative of Test Raw Data')
    raw_ddx_1 = os.path.join(info()['general_dir'], 'first_derivative', '2023_06_29_Potatoes_Pika_L_34-Reflectance '
                                                                        'from Radiance Data and Downwelling '
                                                                        'Irradiance Spectrum-First Derivative.bip.hdr')
    img = read_hyper(raw_ddx_1)[0]

    # remove the unwanted bands
    img = img[:, :, idx]

    print('Loading Second Derivative of Test Raw Data')
    raw_ddx_1 = os.path.join(info()['general_dir'], 'second_derivative', '2023_06_29_Potatoes_Pika_L_34-Reflectance '
                                                                         'from Radiance Data and Downwelling '
                                                                         'Irradiance Spectrum-Second Derivative.bip.hdr')
    img2 = read_hyper(raw_ddx_1)[0]

    # remove the unwanted bands
    img2 = img2[:, :, idx]

    img = np.concatenate((img, img2), axis=2)

    print("Loading test data")
    test_label = np.load(os.path.join(info()['save_dir'], 'labels', '2023_06_29_Potatoes_Pika_L_34-batch-RGB.png.npy'))
    test_data = np.load(os.path.join(info()['save_dir'], 'test_raw', '2023_06_29_Potatoes_Pika_L_34-Bad Band '
                                                                     'Removal.bip.hdr.npy'))

    test_data = np.concatenate((test_data, img), axis=2)

    train_label = train_label.flatten()
    test_label = test_label.flatten()

    # CHANGE TO WANTED LABELS
    # labels_to_keep = [0, 3, 4, 6]
    # labels_to_keep = [3, 4, 6]
    labels_to_keep = [3, 4]

    train_to_keep_idx = np.where(np.isin(train_label, labels_to_keep))[0]
    test_to_keep_idx = np.where(np.isin(test_label, labels_to_keep))[0]

    y_train = to_categorical(train_label[train_to_keep_idx])
    y_test = to_categorical(test_label[test_to_keep_idx])

    y_train = y_train[:, labels_to_keep]
    y_test = y_test[:, labels_to_keep]

    train_data = train_data.reshape((1800000, 224 * 3))
    test_data = test_data.reshape((1800000, 224 * 3))

    x_train = train_data[train_to_keep_idx, :]
    x_test = test_data[test_to_keep_idx, :]

    # keep same amount of 0, 3, and 6 (should be 0, 1, 3) from training data (4 is infected)
    idx_train_0 = np.where(y_train[:, 0] == 1)[0]
    idx_train_1 = np.where(y_train[:, 1] == 1)[0]
    # idx_train_2 = np.where(y_train[:, 2] == 1)[0]
    # idx_train_3 = np.where(y_train[:, 3] == 1)[0]

    random.seed(10)
    idx_0 = random.sample(list(idx_train_0), round(len(idx_train_0) * 0.3))
    # idx_1 = random.sample(list(idx_train_1), round(len(idx_train_1) * 0.3))
    # idx_2 = random.sample(list(idx_train_2), round(len(idx_train_2) * 0.3))
    # idx_3 = random.sample(list(idx_train_3), round(len(idx_train_3) * 0.3))

    # idx_0 = random.sample(list(idx_train_0), len(idx_train_2) * 2)
    # idx_1 = random.sample(list(idx_train_1), len(idx_train_2) * 2)
    # idx_3 = random.sample(list(idx_train_3), len(idx_train_2) * 2)

    # y_train_idx = np.concatenate([idx_0, idx_1, idx_train_2, idx_3])
    # y_train_idx = np.concatenate([idx_0, idx_train_1, idx_2])
    y_train_idx = np.concatenate([idx_0, idx_train_1])
    y_train = y_train[y_train_idx]
    x_train = x_train[y_train_idx, :]

    y_train_flat = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)

    # num_labels = 4
    # num_labels = 3
    num_labels = 2

    input_size = 224 * 3
    batch_size = 128
    hidden_units = 256
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

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])

    model.fit(x_train,
              y_train,
              epochs=20,
              batch_size=batch_size,
              shuffle=True,
              class_weight=dict(enumerate(class_weights)))

    # # plot training validation loss and accuracy
    # Access the training history
    history = model.history.history

    # Get loss and accuracy
    loss = history['loss']
    accuracy = history['accuracy']
    #
    # # Get validation loss and accuracy if available
    # val_loss = history.get('val_loss', None)
    # val_accuracy = history.get('val_accuracy', None)
    #
    # # Plotting loss
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(loss, label='Training Loss')
    # if val_loss is not None:
    #     plt.plot(val_loss, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # # Plotting accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot(accuracy, label='Training Accuracy')
    # if val_accuracy is not None:
    #     plt.plot(val_accuracy, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()


    y_pred = model.predict(x_test)

    # Convert predictions and true labels to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    print(conf_matrix)

    # classes = ['background', 'negative', 'positive', 'resistant negative']
    # classes = ['negative', 'positive', 'resistant negative']
    classes = ['negative', 'positive']

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
