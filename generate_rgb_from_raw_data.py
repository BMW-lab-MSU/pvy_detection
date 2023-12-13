import os
import numpy as np
import matplotlib.pyplot as plt
from utils import info

if __name__=="__main__":
    folder_names = ['test_raw', 'train_raw', 'val_raw']
    data_folder = info()['save_dir']
    save_folder = os.path.join(data_folder, 'rgb_data')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for folder in folder_names:
        print(f"{folder}")
        path = os.path.join(data_folder, folder)
        files = os.listdir(path)
        for file in files:
            file_loc = os.path.join(path, file)
            print(f"{file_loc}")
            data = np.load(file_loc)
            rgb = np.dstack((data[:, :, 113], data[:, :, 70], data[:, :, 26]))
            rgb /= np.max(rgb)
            rgb = rgb.transpose((1, 0, 2))
            filename = os.path.join(save_folder, file + '.jpg')
            plt.imsave(filename, rgb)
