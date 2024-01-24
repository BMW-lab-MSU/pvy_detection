import os
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
    save_name = os.path.join(save_loc, filename + '.npy')   # save as numpy array
    np.save(str(save_name), img)
    del img


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
