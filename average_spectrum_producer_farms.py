import os, glob, re
import numpy as np
from utils import read_hyper, downsample_data, normalize_to_range
from scipy.io import loadmat
import matplotlib.pyplot as plt


if __name__ == "__main__":
    farm_names = ['norota', 'umatilla']
    down_dim_size = 3
    target_shape = (2000 // down_dim_size, 900 // down_dim_size)
    max_zeroes = 80
    threshold_to_clear_shadow = 0.8


    base_path = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/'
                 'data_2024/processed_hyper_data/2024_07_31')
    save_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/'
                'data_2024/processed_hyper_data/2024_07_31/saved_data')
    base_spectrum = loadmat('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY '
                            'Detection/MSU Flights/2023-07-12/data/extracted_data/matlab_data/'
                            'average_healthy_spectrum.mat')
    base_spectrum = base_spectrum['avg_healthy_data']
    base_spectrum = base_spectrum.reshape(223)
    new_min = np.min(base_spectrum)
    new_max = np.max(base_spectrum)

    for farm_name in farm_names:
        if farm_name == 'norota':
            farm_rename = 'Norkotah'
        else:
            farm_rename = 'Umatilla'

        raw_files = glob.glob(os.path.join(base_path, 'radiance_reflectance_smoothed_band_removed_normalized',
                                           farm_name + '*.hdr'))
        raw_files.sort()
        ndvi_files = glob.glob(os.path.join(base_path, 'radiance___normalized_ndvi', farm_name + '*.hdr'))
        ndvi_files.sort()

        for count in range(len(ndvi_files)):
            ndvi_data = np.squeeze(read_hyper(ndvi_files[count])[0])

            # Skip if shape doesn't match
            if ndvi_data.shape != (2000, 900):
                continue

            match = re.search(r'_L_(\d+)-radiance', ndvi_files[count])
            img_num = int(match.group(1))
            print('Geting Spectrum for', farm_rename, img_num)

            downsampled_ndvi = downsample_data(ndvi_data, target_shape, threshold_to_clear_shadow)
            r, c = np.where(downsampled_ndvi)

            if r.size == 0 or c.size == 0:
                continue

            raw_data, wv = read_hyper(raw_files[count])
            raw_data = raw_data[:, :, 1:-1]  # remove first and last band to match the 223 bands from previous year
            wv = wv[1 : -1]
            raw_data = downsample_data(raw_data, target_shape, threshold_to_clear_shadow)

            roi_data = raw_data[r, c, :]
            spectrum = np.mean(roi_data, axis=0)
            spectrum = normalize_to_range(spectrum, new_min, new_max)

            save_name = os.path.join(save_loc, 'spectrum', farm_rename + '_' + str(img_num) + '.png')

            plt.figure()
            plt.plot(wv, base_spectrum)
            plt.plot(wv ,spectrum)
            plt.ylim(0, 100)
            plt.xlim(400, 900)
            plt.legend(['Base Healthy Spectrum', 'Spectrum for ' + farm_rename + ' ' + str(img_num)])
            plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
