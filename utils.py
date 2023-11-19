import spectral as sp
import numpy as np


# read hyperspectral image by bands
def read_hyper(header_path):
    hdr = sp.envi.open(header_path)
    rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
    scale_factor = hdr.scale_factor
    wv = hdr.bands.centers

    img = np.zeros((rows, cols, bands))
    for z in range(bands):
        img[:, :, z] = hdr.read_band(z) * scale_factor

    return img, wv


# store the filepaths and other required information
def info():
    file_directory = {
        'raw_data': '/media/SIDSSD/precisionag/Potatoes/MSU Flights/2023-07-12/raw_radiance_reflectance',
    }
