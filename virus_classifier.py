import xml.etree.ElementTree as ET
import glob
import numpy as np
import keras
import random
import cv2
from utils import *
import seaborn as sns
import tensorflow as tf
from scipy.signal import convolve
from scipy.io import loadmat, savemat
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix


def get_potato_not_potato(ndvi_files_sorted, label_files_sorted):
    threshold_to_clear_shadow = 0.8
    kernel = np.ones((3, 3), np.uint8)

    masks = np.zeros((2000 * 900, len(label_files_sorted)))

    for count in range(len(ndvi_files_sorted)):
        data = np.squeeze(read_hyper(ndvi_files_sorted[count])[0])
        data[data <= threshold_to_clear_shadow] = 0
        data_eroded = cv2.erode(data, kernel, iterations=1).flatten()  # this contains weeds and potato foliage

        label = np.load(label_files_sorted[count]).flatten()
        foliage_with_weeds = np.logical_or(data_eroded, label)
        true_indices = np.where(foliage_with_weeds)[0]

        # find the number of weeds per image
        not_potato_indices = np.where(label[true_indices] == 0)[0]
        potato_indices = np.where(label != 0)[0]
        combined_indices = np.hstack((potato_indices, not_potato_indices))

        label = label[combined_indices]
        label[label != 0] = 1  # change the label to 1 for potatoes

        masks[combined_indices, count] = label

    return masks


# def process_and_generate_data(all_labels, raw_files, all_first, all_second, save_name):
def process_and_generate_data(all_labels, raw_files, save_name):
    random.seed(10)
    idx = list(range(len(all_labels)))
    random.shuffle(idx)
    idx_test = idx[15:]
    idx_val = idx[11:15]

    train_data = np.empty((0, 223 * 3))
    train_labels = np.empty(0)  # there will be 2 labels, 0 and 1

    val_data = train_data
    val_labels = train_labels

    test_data = train_data
    test_labels = train_labels

    kernel_size = 10
    kernel1 = np.ones((kernel_size, kernel_size))
    kernel2 = np.ones((kernel_size, kernel_size, 1))
    labels_to_keep = [299, 300, 399, 400, 599, 600]  # 3 (healthy), 4 (infected), 6 (resistant) * 10

    for num in range(len(all_labels)):
        print(f"Working on Image ", num)
        # img0 = read_hyper(raw_files[num])[0]
        img = read_hyper(raw_files[num])[0]
        # img1 = read_hyper(all_first[num])[0]
        # img2 = read_hyper(all_second[num])[0]
        # img = np.concatenate((img0, img1, img2), axis=2)
        # del img0, img1, img2
        label_whole = np.load(all_labels[num])
        down_sampled_label = convolve(label_whole, kernel1, mode='valid')[::kernel_size,
                             ::kernel_size].flatten().astype(int)
        # down_sampled_img = convolve(img, kernel2, mode='valid')[::kernel_size, ::kernel_size].reshape(
        #     (200 * 90, 223 * 3))
        down_sampled_img = convolve(img, kernel2, mode='valid')[::kernel_size, ::kernel_size].reshape(
            (200 * 90, 223))
        del img
        mask = masked_potatoes[:, num].reshape(2000, 900)
        down_sampled_mask = convolve(mask, kernel1, mode='valid')[::kernel_size, ::kernel_size].flatten().astype(int)

        to_keep_idx = np.where(np.isin(down_sampled_label, labels_to_keep))[0]  # indices with 3, 4, and 6
        mask_indices = np.where((down_sampled_mask == 99) | (down_sampled_mask == 100))[0]

        overlapped_indices = to_keep_idx[np.where(np.isin(to_keep_idx, mask_indices))[0]]
        idx_infected = overlapped_indices[
            np.where((down_sampled_label[overlapped_indices] == 399) | (down_sampled_label[overlapped_indices] == 400))[
                0]]

        idx_healthy = overlapped_indices[np.where(
            (down_sampled_label[overlapped_indices] == 299) | (down_sampled_label[overlapped_indices] == 300) | (
                    down_sampled_label[overlapped_indices] == 599) | (
                    down_sampled_label[overlapped_indices] == 600))[0]]

        if save_data:
            combined_indices = np.hstack((idx_infected, idx_healthy))
            label_selected = down_sampled_label[combined_indices]
            label_selected[(label_selected == 399) | (label_selected == 400)] = 1
            label_selected[label_selected != 1] = 0
            data_selected = down_sampled_img[combined_indices, :]
            del down_sampled_img
            img_num = all_labels[num].split('_')[-1].split('-')[0]
            # file_name_py = os.path.join(info()['save_dir'],
            #                             'compressed_virus_yes_no_img_' + str(img_num) + '_count_' + str(num) + '.npz')
            # file_name_mat = os.path.join(info()['save_dir'],
            #                              'compressed_virus_yes_no_img_' + str(img_num) + '_count_' + str(num) + '.mat')
            file_name_py = os.path.join(info()['save_dir'],
                                        'no_norm_compressed_virus_yes_no_img_' + str(img_num) + '_count_' + str(num) + '.npz')
            file_name_mat = os.path.join(info()['save_dir'],
                                         'no_norm_compressed_virus_yes_no_img_' + str(img_num) + '_count_' + str(num) + '.mat')
            np.savez(file_name_py, combined_indices=combined_indices, label_selected=label_selected,
                     data_selected=data_selected)
            savemat(file_name_mat, {'combined_indices': combined_indices, 'label_selected': label_selected,
                                    'data_selected': data_selected})
            continue

        random.seed(10)
        idx_healthy = random.sample(list(idx_healthy), 1000)  # randomly takng 1000 pixels
        combined_indices = np.hstack((idx_infected, idx_healthy))
        random.seed(10)
        np.random.shuffle(combined_indices)
        label_selected = down_sampled_label[combined_indices]
        label_selected[(label_selected == 399) | (label_selected == 400)] = 1
        label_selected[label_selected != 1] = 0
        data_selected = down_sampled_img[combined_indices, :]
        del down_sampled_img

        if num in idx_val:
            val_data = np.concatenate((val_data, data_selected), axis=0)
            val_labels = np.concatenate((val_labels, label_selected), axis=0)
        elif num in idx_test:
            test_data = np.concatenate((test_data, data_selected), axis=0)
            test_labels = np.concatenate((test_labels, label_selected), axis=0)
        else:
            train_data = np.concatenate((train_data, data_selected), axis=0)
            train_labels = np.concatenate((train_labels, label_selected), axis=0)

    np.savez(save_name, train_data=train_data, train_labels=train_labels, val_data=val_data,
             val_labels=val_labels, test_data=test_data, test_labels=test_labels)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


if __name__ == "__main__":
    ndvi_files = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized_ndvi', '*.hdr'))
    ndvi_files.sort()
    labels = glob.glob(os.path.join(info()['save_dir'], 'labels', '*.npy'))
    labels.sort()
    # raw = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized', '*.hdr'))
    raw = glob.glob(os.path.join(info()['general_dir'], 'raw_rad_ref_smooth_clipped', '*.hdr'))
    raw.sort()
    # first_der = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized_first_derivative', '*.hdr'))
    # first_der.sort()
    # second_der = glob.glob(
    #     os.path.join(info()['general_dir'], 'smoothed_clipped_normalized_second_derivative', '*.hdr'))
    # second_der.sort()

    mask_save_name = os.path.join(info()['save_dir'], 'masked_potato.npy')
    if os.path.exists(mask_save_name):
        masked_potatoes = np.load(mask_save_name)
    else:
        masked_potatoes = get_potato_not_potato(ndvi_files, labels)
        np.save(mask_save_name, masked_potatoes)

    save_data = 1

    # data_save_name = os.path.join(info()['save_dir'], 'virus_classifier_data.npz')
    data_save_name = os.path.join(info()['save_dir'], 'virus_classifier_data_without_normalization.npz')
    if os.path.exists(data_save_name) and not save_data:
        data = np.load(data_save_name)
        data_train = data['train_data']
        labels_train = data['train_labels']
        data_val = data['val_data']
        labels_val = data['val_labels']
        data_test = data['test_data']
        labels_test = data['test_labels']
    else:
        # [data_train, labels_train, data_val, labels_val, data_test, labels_test] = (
        #     process_and_generate_data(labels,
        #                               raw,
        #                               first_der,
        #                               second_der,
        #                               data_save_name))
        [data_train, labels_train, data_val, labels_val, data_test, labels_test] = (
            process_and_generate_data(labels,
                                      raw,
                                      data_save_name))

"""
    input_size = data_train.shape[1]
    # Calculate class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}  # converting to dictionary


    # create the model to tune
    def create_model(num_units, num_layers, learning_rate, batch_size):
        num_units = round(num_units)
        num_layers = round(num_layers)
        batch_size = 2 ** round(batch_size)

        model = Sequential()
        model.add(Dense(num_units, input_dim=input_size, activation='relu'))
        for _ in range(num_layers - 1):
            model.add(Dropout(0.2))
            model.add(Dense(num_units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(data_train, labels_train, epochs=20, batch_size=batch_size, verbose=2,
                  validation_data=(data_val, labels_val), shuffle=True, class_weight=class_weights_dict)
        _, accuracy = model.evaluate(data_test, labels_test, verbose=2)
        return accuracy  # minimize negative accuracy


    # # bounded region of the parameter space
    # pbounds = {
    #     'num_units': (16, 512),
    #     'num_layers': (1, 6),
    #     'learning_rate': (0.0001, 0.1),
    #     'batch_size': (4, 8),
    # }
    # optimizer = BayesianOptimization(
    #     f=create_model,
    #     pbounds=pbounds,
    #     random_state=1,
    #     verbose=2,
    # )
    # logger = JSONLogger(path=os.path.join(info()['save_dir'], 'virus_classifier_logs.json'))
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    # optimizer.maximize(
    #     init_points=5,
    #     n_iter=20,
    # )
    # print(optimizer.max)
    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))

    def optimized_model():
        num_units = 170
        num_layers = 2
        batch_size = 2 ** 6
        learning_rate = 0.072

        model = Sequential()
        model.add(Dense(num_units, input_dim=input_size, activation='relu'))
        for _ in range(num_layers - 1):
            model.add(Dropout(0.2))
            model.add(Dense(num_units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(1, activation='softmax'))
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(data_train, labels_train, epochs=20, batch_size=batch_size, verbose=2,
                  validation_data=(data_val, labels_val), shuffle=True, class_weight=class_weights_dict)
        return model


    # {"target": 0.7714561223983765, "params": {"batch_size": 5.6680880188102964, "learning_rate":
    # 0.0720604168948716, "num_layers": 1.0005718740867244, "num_units": 165.95695602539251}, "datetime": {
    # "datetime": "2024-03-05 16:14:47", "elapsed": 0.0, "delta": 0.0}}

    model_save_name = os.path.join(info()['save_dir'], 'model_opt_virus_classifier.keras')
    if os.path.exists(model_save_name):
        print('Loading Saved Model')
        model = keras.models.load_model(model_save_name)
    else:
        model = optimized_model()
        model.save(model_save_name)
        del data_train, labels_train, data_val, labels_val, data_test, labels_test

    test_data = np.empty((0, 223 * 3))
    test_labels = np.empty(0)  # there will be 2 labels, 0 and 1
    idx_test = [15, 13, 1, 18]

    kernel_size = 10
    kernel1 = np.ones((kernel_size, kernel_size))
    kernel2 = np.ones((kernel_size, kernel_size, 1))
    labels_to_keep = [299, 300, 399, 400, 599, 600]  # 3 (healthy), 4 (infected), 6 (resistant) * 10

    for num in idx_test:
        print(f"Working on Image ", num)
        img0 = read_hyper(raw[num])[0]
        img1 = read_hyper(first_der[num])[0]
        img2 = read_hyper(second_der[num])[0]
        img = np.concatenate((img0, img1, img2), axis=2)
        del img0, img1, img2
        label_whole = np.load(labels[num])
        down_sampled_label = convolve(label_whole, kernel1, mode='valid')[::kernel_size,
                             ::kernel_size].flatten().astype(int)
        down_sampled_img = convolve(img, kernel2, mode='valid')[::kernel_size, ::kernel_size].reshape(
            (200 * 90, 223 * 3))
        del img
        mask = masked_potatoes[:, num].reshape(2000, 900)
        down_sampled_mask = convolve(mask, kernel1, mode='valid')[::kernel_size, ::kernel_size].flatten().astype(int)

        to_keep_idx = np.where(np.isin(down_sampled_label, labels_to_keep))[0]  # indices with 3, 4, and 6
        mask_indices = np.where((down_sampled_mask == 99) | (down_sampled_mask == 100))[0]

        overlapped_indices = to_keep_idx[np.where(np.isin(to_keep_idx, mask_indices))[0]]
        idx_infected = overlapped_indices[
            np.where((down_sampled_label[overlapped_indices] == 399) | (down_sampled_label[overlapped_indices] == 400))[
                0]]

        idx_healthy = overlapped_indices[np.where(
            (down_sampled_label[overlapped_indices] == 299) | (down_sampled_label[overlapped_indices] == 300) | (
                    down_sampled_label[overlapped_indices] == 599) | (
                    down_sampled_label[overlapped_indices] == 600))[0]]

        # random.seed(10)
        # idx_healthy = random.sample(list(idx_healthy), 1000)  # randomly takng 1000 pixels
        combined_indices = np.hstack((idx_infected, idx_healthy))
        # random.seed(10)
        # np.random.shuffle(combined_indices)
        label_selected = down_sampled_label[combined_indices].astype(int)
        label_selected[(label_selected == 399) | (label_selected == 400)] = 1
        label_selected[label_selected != 1] = 0
        data_selected = down_sampled_img[combined_indices, :]
        del down_sampled_img

        pred = model.predict(data_selected)
        threshold = 0.8
        pred = np.squeeze((pred >= threshold).astype(int))

        img_pred = np.ones(200 * 90) * 2
        img_pred[combined_indices] = pred

        img_true = np.ones(200 * 90) * 2
        img_true[combined_indices] = label_selected

        acc = np.sum(np.equal(label_selected, pred)) / len(label_selected)

        plt.subplot(1, 2, 1)
        plt.imshow(img_true.reshape((200, 90)))
        plt.subplot(1, 2, 2)
        plt.imshow(img_pred.reshape((200, 90)))
        img_num = raw[num].split('_')[-1].split('-')[0]
        plt.suptitle('Virus True and Pred for Image ' + str(img_num) + '; Accuracy: ' + str(acc))
        plt_save_name = os.path.join(info()['save_dir'], 'Virus_True_and_Pred_for_Image_' + str(img_num) + '.png')
        plt.savefig(plt_save_name)
        plt.close()
"""