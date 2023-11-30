import os
import spectral as sp
import numpy as np


# read hyperspectral image by bands
def read_hyper(header_path):
    hdr = sp.envi.open(header_path)
    rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
    scale_factor = hdr.scale_factor
    wv = hdr.bands.centers
    img = np.float64(hdr.read_bands(range(0, bands))) * scale_factor
    return img, wv


# save extracted hyper data
def save_extracted_data(file, save_loc):
    filename = file.split('/')[-1]
    print(f"Processing file: {filename}")
    img = read_hyper(file)[0]
    save_name = os.path.join(save_loc, filename + '.npy')
    np.save(save_name, img)
    del img


def rle2mask(rle, source_width, source_height, left, top, target_width, target_height):
    img = np.array(np.zeros([source_height, source_width]))
    decoded = [0] * (target_width * target_height)  # create bitmap container
    decoded_idx = 0
    value = 0

    for v in rle:
        decoded[decoded_idx:decoded_idx + v] = [value] * v
        decoded_idx += v
        value = abs(value - 1)

    decoded = np.array(decoded, dtype=np.uint8)
    decoded = decoded.reshape((target_height, target_width))  # reshape to image size

    img[top:top + decoded.shape[0], left:left + decoded.shape[1]] = decoded

    return img


# store the filepaths and other required information
def info():
    file_directory = {
        'general_dir': '/media/SIDSSD/precisionag/Potatoes/MSU Flights/2023-07-12/data',
        'raw_data': '/media/SIDSSD/precisionag/Potatoes/MSU Flights/2023-07-12/data/raw_radiance_reflectance',
        'save_dir': '/media/SIDSSD/precisionag/Potatoes/MSU Flights/2023-07-12/data/extracted_data',
    }

    return file_directory
