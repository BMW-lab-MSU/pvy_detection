import os.path
import glob
import keras
import numpy as np
from fontTools.merge.util import first
from scipy.io import savemat
from concurrent.futures import ThreadPoolExecutor
from utils import *
from scipy.signal import convolve
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt





if __name__ == "__main__":
    # load the files
    farm_names = ['norota', 'umatilla']
    down_dim_size = 3
    patch_size = (15, 15)   # consider this as one plant
    target_shape = (2000 // down_dim_size, 900 // down_dim_size)
    max_zeroes = 80
    threshold_to_clear_shadow = 0.8


    for farm_name in farm_names:
        if farm_name == 'norota':
            farm_rename = 'norkotah'
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
            # ndvi_data[ndvi_data <= threshold_to_clear_shadow] = 0
            if ndvi_data.shape != (2000, 900):
                continue

            match =  re.search(r'_L_(\d+)-radiance', ndvi_files[count])
            img_num = int(match.group(1))
            print('Processing', farm_rename, img_num)

            # load the data
            raw_data = read_hyper(raw_files[count])[0]
            raw_data = raw_data[:, :, 1:-1]     # remove first and last band to match the 223 bands from previous year
            first_data = read_hyper(first_der[count])[0]
            first_data = first_data[:, :, 1:-1]  # remove first and last band to match the 223 bands from previous year
            second_data = read_hyper(second_der[count])[0]
            second_data = second_data[:, :, 1:-1]  # remove first and last band to match the 223 bands from previous year

            patch_creator = PatchCreator(ndvi_data, raw_data, first_data, second_data, patch_size, target_shape,
                                         max_zeroes, threshold_to_clear_shadow)

            patches = patch_creator.create_patches()



    # do the ndvi first and just downsample by 3x


    # create a patch of 15x15 without any overlap if there is ndvi data and make them traceable

