import os
import random
import re

import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from utils import *
import cv2
import spectral as sp
import numpy as np
from scipy.ndimage import zoom
# from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


if __name__ == "__main__":
    base_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/'
                'MSU Flights/2023-07-12/data/sample_data_for_paper')

    # gets the specs
    names = ['specs_raw', 'specs_radiance_calibrated', 'specs_reflectance_calibrated', 'specs_smoothed', 'specs_normalized_final']

    loc = [os.path.join(base_loc, '2023_06_29_Potatoes_Pika_L_25.bil.hdr'),
               os.path.join(base_loc, '2023_06_29_Potatoes_Pika_L_25-RadianceConversion.bip.hdr'),
               os.path.join(base_loc, '2023_06_29_Potatoes_Pika_L_25-RadianceConversion-'
                                       'ReflectivityConversionFromRadiance.bip.hdr'),
               os.path.join(base_loc, '2023_06_29_Potatoes_Pika_L_25-RadianceConversion-'
                                      'ReflectivityConversionFromRadiance-SGFilter.bip.hdr'),
               os.path.join(base_loc, '2023_06_29_Potatoes_Pika_L_25-RadianceConversion-'
                                      'ReflectivityConversionFromRadiance-SGFilter-CropWavelengthsByBand-'
                                      'BadBandRemoval-NormalizeCube.bip.hdr')]

    for count, file in enumerate(loc):
        print(f'Working on {names[count]}')
        data, wv = read_hyper(file)

        if count == 2 or count == 3:
            data = data[:, :, 10:-56]
            wv = wv[10:-56]

        # x, y values for different types
        drySoil = data[1148, 697, :]
        wetSoil = data[127, 330, :]
        shadow = data[172, 382, :]
        healthy = data[723, 471, :]
        infected = data[1425, 765, :]

        plt.figure()
        plt.plot(wv, drySoil, marker = 'o', markevery=7)
        plt.plot(wv, wetSoil, marker = 's', markevery=7)
        plt.plot(wv, shadow, marker = '^', markevery=7)
        plt.plot(wv, healthy, marker = 'v', markevery=7)
        plt.plot(wv, infected, marker = '*', markevery=7)
        plt.legend(['Dry Soil', 'Wet Soil', 'Shadow', 'Healthy Plant', 'Infected Plant'])
        plt.savefig(os.path.join(base_loc, names[count] + '.png'), dpi=300)
        plt.close()

    # df = pd.read_csv(os.path.join(base_loc, 'spectrum.txt'), sep='\t', header=None, names=['wv', 'spec'])
    # plt.plot(df['wv'], df['spec'])
    # plt.xlim([300, 1050])
    # plt.savefig(os.path.join(base_loc, 'downwelling_spectrum.png' + '.png'), dpi=300)
    # plt.close()


    # # get field plot
    # field_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/'
    #              'MSU Flights/Potatoes_Altum/3_dsm_ortho/2_mosaic')
    #
    # red = os.path.join(field_loc, 'PotatoesGood_transparent_mosaic_red.tif')
    # green = os.path.join(field_loc, 'PotatoesGood_transparent_mosaic_green.tif')
    # blue = os.path.join(field_loc, 'PotatoesGood_transparent_mosaic_blue.tif')
    #
    # with rasterio.open(red) as src:
    #     r = src.read(1)
    #     r = r[3000:6500, 3000:8500]
    # with rasterio.open(green) as src:
    #     g = src.read(1)
    #     g = g[3000:6500, 3000:8500]
    # with rasterio.open(blue) as src:
    #     b = src.read(1)
    #     b = b[3000:6500, 3000:8500]
    #
    # # rgb = np.dstack((r, g, b))
    # # rgb = np.clip(rgb, 0, 1)
    #
    # red_channel = r
    # green_channel = g
    # blue_channel = b
    #
    # # Normalize to 0-255 if not already in that range
    # red_channel = np.clip((red_channel - np.min(red_channel)) / (np.max(red_channel) - np.min(red_channel)) * 255, 0,
    #                       255).astype(np.uint8)
    # green_channel = np.clip(
    #     (green_channel - np.min(green_channel)) / (np.max(green_channel) - np.min(green_channel)) * 255, 0, 255).astype(
    #     np.uint8)
    # blue_channel = np.clip((blue_channel - np.min(blue_channel)) / (np.max(blue_channel) - np.min(blue_channel)) * 255,
    #                        0, 255).astype(np.uint8)
    #
    # # Stack into an RGB image
    # rgb_image = cv2.merge((blue_channel, green_channel, red_channel))  # OpenCV uses BGR format
    #
    # # Save or display the image
    # cv2.imwrite(os.path.join(base_loc, 'field_plot_rgb.png'), rgb_image)
    # # cv2.imshow('RGB Image', rgb_image)
    # # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # # get label of the sample image 25
    # label_loc = os.path.join(base_loc, '2023_06_29_Potatoes_Pika_L_25-batch-RGB.png.npy')
    # label = np.load(label_loc)  # has unique vlaues array([0, 3, 4, 5, 6], dtype=uint8)
    # # 0/0 – background, 3/1 – healthy, 4/2 – infected, 5/3 – unknown, 6/4 – resistant
    # value_map = {0: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    # mapped_label = np.vectorize(value_map.get)(label)
    # # mapped_label = mapped_label.T
    # colors = ['black', 'green', 'red', 'gray', 'blue']  # Colors for background, healthy, infected, unknown, resistant
    # cmap = ListedColormap(colors)
    # bounds = [0, 1, 2, 3, 4, 5]  # Boundaries for the color mapping
    # norm = BoundaryNorm(bounds, cmap.N)
    #
    # plt.figure()
    # img = plt.imshow(mapped_label, cmap=cmap, norm=norm)
    #
    # # Create a custom colorbar
    # cbar = plt.colorbar(img, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
    # cbar.ax.set_yticklabels(['Background', 'Healthy', 'Infected', 'Unknown', 'Resistant'])  # Custom labels
    # plt.axis('off')
    # plt.savefig(os.path.join(base_loc, 'label_25.png'), dpi=300)
    # plt.close()