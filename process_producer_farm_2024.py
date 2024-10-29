import os.path
# import re
# import xml.etree.ElementTree as ET
import glob
from logging import exception
from uu import Error

# from calendar import firstweekday
#
# import numpy as np
import keras
import numpy as np
from scipy.io import savemat

# import random
# import cv2
from concurrent.futures import ProcessPoolExecutor
# from test_pre_processing import kernel
from utils import *
# import seaborn as sns
# import tensorflow as tf
from scipy.signal import convolve
from scipy.io import loadmat, savemat
# from bayes_opt import BayesianOptimization
# from bayes_opt.logger import JSONLogger
# from bayes_opt.event import Events
# from sklearn.utils import class_weight
import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU
# from keras.utils import to_categorical, plot_model
# from keras.regularizers import l2
# from sklearn.metrics import confusion_matrix


def downsampled_foliage_mask(data):
    threshold_to_clear_shadow = 0.8
    # kernel_size = 3     # change both if needed
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    data[data <= threshold_to_clear_shadow] = 0
    data = cv2.erode(data, kernel, iterations=1)  # this contains weed and potato foliage
    data = convolve(data, kernel, mode='valid')[::down_dim_size, ::down_dim_size]   # change kernel_size if needed
    save_name = os.path.join(save_loc, 'potato_masks', farm_rename + '_' + str(img_num) + '_kernel_size_' +
                             str(kernel_size) + '_' + str(down_dim_size) + '_downsampled_potatoes' + '.png')
    plt.imshow(data)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    data = data.flatten().astype(int)
    return data


def downsampled_image(filename):
    # kernel_size = 3     # change both if needed
    kernel = np.ones((kernel_size, kernel_size, 1))

    img0 = read_hyper(filename[0])[0]   # raw image
    img0 = img0[:, :, 1:-1]             # remove first and last band to match the 223 bands from previous year
    img1 = read_hyper(filename[1])[0]   # first derivative image
    img1 = img1[:, :, 1:-1]             # remove first and last band to match the 223 bands from previous year
    img2 = read_hyper(filename[2])[0]   # second derivative image
    img2 = img2[:, :, 1:-1]             # remove first and last band to match the 223 bands from previous year
    img = np.concatenate((img0, img1, img2), axis=2)
    del img0, img1, img2
    data = (convolve(img, kernel, mode='valid')[::down_dim_size, ::down_dim_size]
            .reshape(((2000 // down_dim_size) * (900 // down_dim_size), 223 * 3)))
    del img
    save_name = os.path.join(save_loc, 'potato_masks', farm_rename + '_' + str(img_num) + '_kernel_size_' +
                             str(kernel_size) + '_' + str(down_dim_size) + '_downsampled_data' + '.npy')
    save_name_mat = os.path.join(save_loc, 'potato_masks', farm_rename + '_' + str(img_num) + '_kernel_size_' +
                             str(kernel_size) + '_' + str(down_dim_size) + '_downsampled_data' + '.mat')
    np.save(save_name, data)
    savemat(save_name_mat, {'data': data})
    return data


def virus_or_not(img, idx):
    # thresh = 0.8
    model_name = os.path.join(info()['save_dir'], 'model_opt_virus_classifier.keras')
    model = keras.models.load_model(model_name)

    rgb_image = img[:, [112, 69, 26]]
    rgb_image /= np.max(rgb_image)
    rgb_image = rgb_image.reshape((2000 // down_dim_size, 900 // down_dim_size, 3))

    img = img[idx, :]
    pred_base = model.predict(img)

    thresh_list = [0.5, 0.6, 0.7]
    for thresh in thresh_list:
        pred = np.squeeze((pred_base >= thresh).astype(int))
        pred_idx = np.where(pred)[0]
        infected = len(pred_idx)

        # len(pred) got zero, fix it! - probably for norkotah 9
        infected_percentage = round(infected / len(pred), 6)    # round to 6 decimal places

        img_pred = rgb_image

        height, width, _ = img_pred.shape
        rows = idx[pred_idx] // width
        cols = idx[pred_idx] % width
        img_pred[rows, cols, :] = 1

        save_name = os.path.join(save_loc, 'virus_predictions', farm_rename + '_' + str(img_num) + '_inf_per_' +
                                 str(infected_percentage) + '_inf_' + str(infected) + '_in_' + str(len(pred)) +
                                 '_kernel_size_' + str(kernel_size) + '_' + str(down_dim_size) + '_thresh_' +
                                 str(thresh) + '.png')

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title('RGB Image')
        plt.subplot(1, 2, 2)
        plt.imshow(img_pred)
        plt.axis('off')
        plt.title('Predicted Image')
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

    return pred_base


if __name__ == "__main__":
    # choose from: BART, norota, umatilla
    farm_names = ['BART', 'norota', 'umatilla']
    # farm_names = ['norota']

    for farm_name in farm_names:
        kernel_size = 3
        down_dim_size = 3
        # down_dim_size = 10

        if farm_name == 'norota':
            farm_rename = 'norkotah'
        elif farm_name == 'BART':
            farm_rename = 'bart'
            # kernel_size = 10
            down_dim_size = 10
        else:
            farm_rename = farm_name

        base_path = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/'
                     'data_2024/processed_hyper_data/2024_07_31')
        save_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/'
                    'data_2024/processed_hyper_data/2024_07_31/saved_data')

        raw_files = glob.glob(os.path.join(base_path, 'radiance_reflectance_smoothed_band_removed_normalized',
                                           farm_name + '*.hdr'))
        raw_files.sort()
        ndvi_files = glob.glob(os.path.join(base_path, 'radiance___normalized_ndvi', farm_name + '*.hdr'))
        ndvi_files.sort()
        first_der = glob.glob(os.path.join(base_path, 'radiance___normalized_first_derivative', farm_name + '*.hdr'))
        first_der.sort()
        second_der = glob.glob(os.path.join(base_path, 'radiance___normalized_second_derivative', farm_name + '*.hdr'))
        second_der.sort()

        for count in range(len(ndvi_files)):
            ndvi_data = np.squeeze(read_hyper(ndvi_files[count])[0])
            if ndvi_data.shape != (2000, 900):
                continue

            match =  re.search(r'_L_(\d+)-radiance', ndvi_files[count])
            img_num = int(match.group(1))
            print('Processing', farm_rename, img_num)
            # get the downsampled potato or not potato foliage
            ndvi = downsampled_foliage_mask(ndvi_data)  # gives flatten of (666, 300)=199800 kernel of 3x3
            idx_potato = np.where(ndvi)[0]

            if idx_potato.size == 0:
                continue

            idx_save_name = os.path.join(save_loc, 'potato_masks', farm_rename + '_' + str(img_num) + '_kernel_size_' +
                             str(kernel_size) + '_' + str(down_dim_size) + '_downsampled_idx' + '.npy')
            idx_save_name_mat = os.path.join(save_loc, 'potato_masks', farm_rename + '_' + str(img_num) + '_kernel_size_' +
                                         str(kernel_size) + '_' + str(down_dim_size) + '_downsampled_idx' + '.mat')
            np.save(idx_save_name, idx_potato)
            savemat(idx_save_name_mat, {'idx': idx_potato})

            filenames = [raw_files[count], first_der[count], second_der[count]]

            img_save_name = os.path.join(save_loc, 'potato_masks', farm_rename + '_' + str(img_num) + '_kernel_size_' +
                                     str(kernel_size) + '_' + str(down_dim_size) + '_downsampled_data' + '.npy')

            if os.path.exists(img_save_name):
                image = np.load(img_save_name)
            else:
                image = downsampled_image(filenames)
            try:
                predictions = virus_or_not(image, idx_potato)
            except:
                print('Error in', farm_rename, img_num, 'for count =', count)
