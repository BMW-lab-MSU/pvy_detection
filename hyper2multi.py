import xml.etree.ElementTree as ET
import glob
import re
import numpy as np
import keras
import random
import cv2
import openpyxl
from utils import *
import seaborn as sns
import tensorflow as tf
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat, savemat
from scipy.interpolate import CubicSpline
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    reflectance_files = glob.glob(os.path.join(info()['general_dir'], 'raw_radiance_reflectance', '*.hdr'))
    reflectance_files.sort()

    transmission_file = os.path.join(info()['general_dir'], 'mapir_filter_transmission_data.xlsx')
    workbook = openpyxl.load_workbook(transmission_file)
    sheet = workbook.active  # select active sheet or specify if not the first one
    transposed_data = list(zip(*sheet.values))  # transpose to get the columns

    multi_data_folder = os.path.join(info()['save_dir'], 'mapir_multi_data')
    if not os.path.exists(multi_data_folder):
        os.mkdir(multi_data_folder)

    for hyper_file in reflectance_files:
        img_num = re.search(r'_(\d+)-', hyper_file).group()[1:3]
        print('Working on Image', img_num)

        # wv[0] = 387.12, wv[288] = 998.75
        [img_hyper, wv_hyper] = read_hyper(hyper_file)
        img = img_hyper[:, :, : 275].reshape(-1, 275)
        wv = wv_hyper[: 275]

        for wave_data_count in range(0, int(len(transposed_data) / 2)):
            wave_col = list(filter(lambda x: x is not None, transposed_data[wave_data_count * 2]))
            center_frequency = re.search(r'\d+', wave_col[0]).group()
            print('Hyper2Multi for', center_frequency, 'Hz -', wave_data_count + 1, '/', int(len(transposed_data) / 2))
            wave_col.pop(0)
            data_col = list(filter(lambda x: x is not None, list(transposed_data[wave_data_count * 2 + 1])))
            data_col.pop(0)

            spl = CubicSpline(wave_col, data_col)
            multi_data = spl(wv)

            img_new = np.matmul(img, multi_data) / sum(multi_data)
            filename = os.path.join(multi_data_folder, 'img_' + img_num + '_center_frequency_' + center_frequency + '.npy')
            np.save(filename, img_new)
