import os
import glob
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from utils import read_hyper, downsample_data, normalize_to_range
from scipy.io import loadmat
import matplotlib.pyplot as plt


def process_file(ndvi_file, raw_file, farm_rename, base_spectrum, new_min, new_max, target_shape, save_loc, threshold_to_clear_shadow):
    # Process NDVI data
    ndvi_data = np.squeeze(read_hyper(ndvi_file)[0])
    if ndvi_data.shape != (2000, 900):  # Skip if shape doesn't match
        return

    match = re.search(r'_L_(\d+)-radiance', ndvi_file)
    img_num = int(match.group(1))
    print('Getting Spectrum for', farm_rename, img_num)

    downsampled_ndvi = downsample_data(ndvi_data, target_shape, threshold_to_clear_shadow)
    r, c = np.where(downsampled_ndvi)
    if r.size == 0 or c.size == 0:
        return

    # Process raw data
    raw_data, wv = read_hyper(raw_file)
    raw_data = raw_data[:, :, 1:-1]  # Remove first and last bands
    wv = wv[1:-1]
    raw_data = downsample_data(raw_data, target_shape, threshold_to_clear_shadow)

    roi_data = raw_data[r, c, :]
    spectrum = np.mean(roi_data, axis=0)
    spectrum = normalize_to_range(spectrum, new_min, new_max)

    save_name = os.path.join(save_loc, 'spectrum', f'{farm_rename}_{img_num}.png')
    plt.figure()
    plt.plot(wv, base_spectrum)
    plt.plot(wv, spectrum)
    plt.ylim(0, 100)
    plt.xlim(400, 900)
    plt.legend(['Base Healthy Spectrum', f'Spectrum for {farm_rename} {img_num}'])
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


if __name__ == "__main__":
    farm_names = ['norota', 'umatilla']
    down_dim_size = 3
    target_shape = (2000 // down_dim_size, 900 // down_dim_size)
    threshold_to_clear_shadow = 0.8

    base_path = '/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/data_2024/processed_hyper_data/2024_07_31'
    save_loc = '/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/data_2024/processed_hyper_data/2024_07_31/saved_data'
    base_spectrum = loadmat('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/MSU Flights/2023-07-12/data/extracted_data/matlab_data/average_healthy_spectrum.mat')
    base_spectrum = base_spectrum['avg_healthy_data'].reshape(223)
    new_min = np.min(base_spectrum)
    new_max = np.max(base_spectrum)

    # Prepare file lists
    tasks = []
    for farm_name in farm_names:
        farm_rename = 'Norkotah' if farm_name == 'norota' else 'Umatilla'

        raw_files = sorted(glob.glob(os.path.join(base_path, 'radiance_reflectance_smoothed_band_removed_normalized', f'{farm_name}*.hdr')))
        ndvi_files = sorted(glob.glob(os.path.join(base_path, 'radiance___normalized_ndvi', f'{farm_name}*.hdr')))

        for ndvi_file, raw_file in zip(ndvi_files, raw_files):
            tasks.append((ndvi_file, raw_file, farm_rename, base_spectrum, new_min, new_max, target_shape, save_loc, threshold_to_clear_shadow))

    # Parallel processing
    with ProcessPoolExecutor() as executor:
        executor.map(lambda args: process_file(*args), tasks)
