import os
import glob
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import read_hyper, info
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU, Conv1D, MaxPooling1D
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    all_files = glob.glob(os.path.join(info()['raw_data'], '*.hdr'))
    all_files.sort()
    all_first = glob.glob(os.path.join(info()['general_dir'], 'first_derivative', '*.hdr'))
    all_first.sort()
    all_second = glob.glob(os.path.join(info()['general_dir'], 'second_derivative', '*.hdr'))
    all_second.sort()
    all_labels = glob.glob(os.path.join(info()['save_dir'], 'labels', '*.npy'))
    all_labels.sort()

    idx_train = [3, 11, 12, 16, 21, 25, 28, 29]
    idx_val = [6, 20]
    idx_test = [7, 15]
    idx = list(range(10, 175)) + list(range(185, 244))
    labels_to_keep = [0, 3, 4, 6]

    # TRAINING DATA ACCUMULATION
    train_labels = np.empty((0, len(labels_to_keep)))
    train_data = np.empty((0, 224 * 3))
    for i in idx_train:
        print(f"Processing Train Data, Index number: {i}, filename: {all_files[i]}")
        img0 = read_hyper(all_files[i])[0]
        img1 = read_hyper((all_first[i]))[0]
        img1 = img1[:, :, idx]
        img2 = read_hyper(all_second[i])[0]
        img2 = img2[:, :, idx]
        img = np.concatenate((img0, img1, img2), axis=2)
        del img0, img1, img2
        label = np.load(all_labels[i]).flatten()
        to_keep_idx = np.where(np.isin(label, labels_to_keep))[0]
        y_val = to_categorical(label[to_keep_idx])
        y_val = y_val[:, labels_to_keep]
        img = img.reshape((1800000, 224 * 3))
        img = img[to_keep_idx, :]

        # keep same amount of 0, 3, and 6 (should be 0, 1, 3) from training data (4 is infected)
        idx_train_0 = np.where(y_val[:, 0] == 1)[0]
        idx_train_1 = np.where(y_val[:, 1] == 1)[0]
        idx_train_2 = np.where(y_val[:, 2] == 1)[0]
        idx_train_3 = np.where(y_val[:, 3] == 1)[0]

        min_value = np.min([len(idx_train_0), len(idx_train_1), len(idx_train_2), len(idx_train_3)])

        if min_value > 0:
            random.seed(10)
            idx_0 = random.sample(list(idx_train_0), min_value)
            idx_1 = random.sample(list(idx_train_1), min_value)
            idx_3 = random.sample(list(idx_train_3), min_value)

            y_train_idx = np.concatenate([idx_0, idx_1, idx_train_2, idx_3])
        else:
            y_train_idx = idx_train_2

        y_train = y_val[y_train_idx]
        x_train = img[y_train_idx, :]
        del img

        train_labels = np.concatenate((train_labels, y_train), axis=0)
        train_data = np.concatenate((train_data, x_train), axis=0)

    # VALIDATION DATA ACCUMULATION
    val_labels = np.empty((0, len(labels_to_keep)))
    val_data = np.empty((0, 224 * 3))
    for i in idx_val:
        print(f"Processing Validation Data, Index number: {i}, filename: {all_files[i]}")
        img0 = read_hyper(all_files[i])[0]
        img1 = read_hyper((all_first[i]))[0]
        img1 = img1[:, :, idx]
        img2 = read_hyper(all_second[i])[0]
        img2 = img2[:, :, idx]
        img = np.concatenate((img0, img1, img2), axis=2)
        del img0, img1, img2
        label = np.load(all_labels[i]).flatten()
        to_keep_idx = np.where(np.isin(label, labels_to_keep))[0]
        y_val = to_categorical(label[to_keep_idx])
        y_val = y_val[:, labels_to_keep]
        img = img.reshape((1800000, 224 * 3))
        img = img[to_keep_idx, :]

        # keep same amount of 0, 3, and 6 (should be 0, 1, 3) from training data (4 is infected)
        idx_train_0 = np.where(y_val[:, 0] == 1)[0]
        idx_train_1 = np.where(y_val[:, 1] == 1)[0]
        idx_train_2 = np.where(y_val[:, 2] == 1)[0]
        idx_train_3 = np.where(y_val[:, 3] == 1)[0]

        min_value = np.min([len(idx_train_0), len(idx_train_1), len(idx_train_2), len(idx_train_3)])

        if min_value > 0:
            random.seed(10)
            idx_0 = random.sample(list(idx_train_0), min_value)
            idx_1 = random.sample(list(idx_train_1), min_value)
            idx_3 = random.sample(list(idx_train_3), min_value)

            y_train_idx = np.concatenate([idx_0, idx_1, idx_train_2, idx_3])
        else:
            y_train_idx = idx_train_2

        y_train = y_val[y_train_idx]
        x_train = img[y_train_idx, :]
        del img

        val_labels = np.concatenate((val_labels, y_train), axis=0)
        val_data = np.concatenate((val_data, x_train), axis=0)

    y_train_flat = np.argmax(train_labels, axis=1)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)

    num_labels = 4

    input_size = 224 * 3
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

    model.add(Dense(hidden_units))
    model.add(Dense(2 * hidden_units, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())  # Add Batch Normalization
    model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(dropout))

    model.add(Dense(hidden_units, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(num_labels, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_data,
              train_labels,
              epochs=30,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(val_data, val_labels),
              class_weight=dict(enumerate(class_weights)))

    model.save(os.path.join(info()['save_dir'], 'model.keras'))

    # # plot training validation loss and accuracy
    # Access the training history
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
    plt.savefig(os.path.join(info()['save_dir'], 'train_val_loss_acc.png'))
    # plt.show()

    # TEST DATA ACCUMULATION
    test_labels = np.empty((0, len(labels_to_keep)))
    test_data = np.empty((0, 224 * 3))
    for i in idx_test:
        print(f"Processing Test Data, Index number: {i}, filename: {all_files[i]}")
        img0 = read_hyper(all_files[i])[0]
        img1 = read_hyper((all_first[i]))[0]
        img1 = img1[:, :, idx]
        img2 = read_hyper(all_second[i])[0]
        img2 = img2[:, :, idx]
        img = np.concatenate((img0, img1, img2), axis=2)
        del img0, img1, img2
        label = np.load(all_labels[i]).flatten()
        to_keep_idx = np.where(np.isin(label, labels_to_keep))[0]
        y_val = to_categorical(label[to_keep_idx])
        y_val = y_val[:, labels_to_keep]
        img = img.reshape((1800000, 224 * 3))
        img = img[to_keep_idx, :]

        test_labels = np.concatenate((test_labels, y_val), axis=0)
        test_data = np.concatenate((test_data, img), axis=0)

    predicted = model.predict(test_data)

    # Convert predictions and true labels to class labels
    predicted_classes = np.argmax(predicted, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    # Create a confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print(conf_matrix)

    classes = ['background', 'negative', 'positive', 'resistant negative']

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(info()['save_dir'], 'test_conf_mat.png'))
    # plt.show()
