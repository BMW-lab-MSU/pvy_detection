import os
import re
import spectral as sp
import numpy as np
# from shapely.geometry import Polygon
from PIL import Image, ImageDraw


# read hyperspectral image by bands
def read_hyper(header_path):
    hdr = sp.envi.open(header_path)
    rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
    scale_factor = hdr.scale_factor
    wv = hdr.bands.centers
    # float32 is default and goes off-range for last few bands - so convert to float64
    # img = np.float64(hdr.read_bands(range(0, bands))) * scale_factor
    img = hdr.read_bands(range(0, bands)) * scale_factor
    # read_bands reads all the available bands
    return img, wv


# save extracted hyperspectral data
def save_extracted_data(file, save_loc):
    filename = file.split('/')[-1]
    print(f"Processing file: {filename}")
    img = read_hyper(file)[0]
    save_name = os.path.join(save_loc, filename + '.npy')  # save as numpy array
    np.save(str(save_name), img)
    del img


def concatenate_data_label(file_num, file_info):
    file_raw = file_info[0][file_num]
    file_first = file_info[1][file_num]
    file_second = file_info[2][file_num]
    file_label = file_info[3][file_num]

    idx = list(range(10, 175)) + list(range(185, 244))

    print(f"Processing file: {file_raw.split('/')[-1]}")
    img0 = read_hyper(file_raw)[0]
    print(f"Processing file: {file_first.split('/')[-1]}")
    img1 = read_hyper(file_first)[0]
    img1 = img1[:, :, idx]
    print(f"Processing file: {file_second.split('/')[-1]}")
    img2 = read_hyper(file_second)[0]
    img2 = img2[:, :, idx]

    img = np.concatenate((img0, img1, img2), axis=2)
    del img0, img1, img2

    print(f"Loading Label File : {file_label.split('/')[-1]}")
    label = np.load(file_label).flatten()

    labels_to_keep = [0, 3, 4, 6]
    to_keep_idx = np.where(np.isin(label, labels_to_keep))[0]
    label = label[to_keep_idx]

    img = img.reshape((1800000, 224 * 3))
    img = img[to_keep_idx, :]

    return img, label


# save the hyperspectral data and their derivatives (1st, 2nd) by pixels
def save_data_by_pixels(file_num, file_info, save_loc):
    print(file_num)
    [img, label] = concatenate_data_label(file_num, file_info)

    # make directories if they do not exist
    class_dir_loc = os.path.join(save_loc, 'background')
    print(class_dir_loc)
    if not os.path.exists(class_dir_loc):
        os.makedirs(class_dir_loc)
    class_dir_loc = os.path.join(save_loc, 'healthy')
    if not os.path.exists(class_dir_loc):
        os.makedirs(class_dir_loc)
    class_dir_loc = os.path.join(save_loc, 'infected')
    if not os.path.exists(class_dir_loc):
        os.makedirs(class_dir_loc)
    class_dir_loc = os.path.join(save_loc, 'resistant')
    if not os.path.exists(class_dir_loc):
        os.makedirs(class_dir_loc)

    file_raw = file_info[0][file_num]
    filename_base = 'Image_' + re.search(r'(\d+)-', file_raw.split('/')[-1]).group(1)

    for count in range(len(label)):
        filename = filename_base + '_Pixel_' + str(count)
        save_name = []
        if label[count] == 0:  # background
            class_dir_loc = os.path.join(save_loc, 'background')
            save_name = os.path.join(class_dir_loc, filename + '.npy')

        if label[count] == 3:  # healthy
            class_dir_loc = os.path.join(save_loc, 'healthy')
            save_name = os.path.join(class_dir_loc, filename + '.npy')

        if label[count] == 4:  # infected
            class_dir_loc = os.path.join(save_loc, 'infected')
            save_name = os.path.join(class_dir_loc, filename + '.npy')

        if label[count] == 6:  # resistant
            class_dir_loc = os.path.join(save_loc, 'resistant')
            save_name = os.path.join(class_dir_loc, filename + '.npy')

        print(f'Saving file :  {save_name}')
        np.save(save_name, img[count, :])


# decode the run length encoding to get the mask
def rle2mask(rle, source_width, source_height, left, top, target_width, target_height, filling_number):
    mask = np.array(np.zeros([source_height, source_width]), dtype=np.uint8)
    decoded = [0] * (target_width * target_height)  # create bitmap container
    decoded_idx = 0
    value = 0

    for i in rle:
        decoded[decoded_idx:decoded_idx + i] = [value] * i
        decoded_idx += i
        value = abs(value - 1)

    decoded = np.array(decoded, dtype=np.uint8)
    decoded = decoded.reshape((target_height, target_width))  # reshape to image size

    mask[top:top + decoded.shape[0], left:left + decoded.shape[1]] = decoded
    mask[mask == 1] = filling_number

    return mask


# get the polygon converted to mask
def poly2mask(points, source_width, source_height, filling_number):
    img = Image.new('L', (source_width, source_height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(points, outline=filling_number, fill=filling_number)
    mask = np.array(img)

    return mask


# label correction info
def label_correction():
    flip_axis = {
        '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 1, '6': 1, '7': 1, '8': 1, '9': 0, '10': 0, '11': 0, '12': 0,
        '13': 0, '14': 0, '15': 1, '16': 1, '17': 0, '18': 0, '19': 0, '20': 0, '21': 0, '22': 0, '23': 1, '24': 1,
        '25': 1, '26': 0, '27': 0, '28': 0, '29': 0, '30': 0,
    }

    return flip_axis


# store the filepaths and other required information
def info():
    file_directory = {
        'general_dir': '/media/SIDSSD/precisionag/Potatoes/MSU Flights/2023-07-12/data',
        'raw_data': '/media/SIDSSD/precisionag/Potatoes/MSU Flights/2023-07-12/data/hyper_data_raw_band_removed',
        'save_dir': '/media/SIDSSD/precisionag/Potatoes/MSU Flights/2023-07-12/data/extracted_data',
    }

    return file_directory
