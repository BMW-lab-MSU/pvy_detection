import os
import random
import re

import cv2
import spectral as sp
import numpy as np
from scipy.ndimage import zoom
# from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def normalize_to_range(data, new_min, new_max):
    """
    Normalize data to the specified range

    :param data: 1D, 2D, or 3D array
    :param new_min: New minimum value
    :param new_max: New maximum value
    :return: Return the normalized data to the specified range
    """
    old_min = np.min(data)
    old_max = np.max(data)

    return (((data - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min


class PatchImageGenerator:
    def __init__(self, ndvi_data, patch_size, target_shape, patches_with_indices):
        """
        Initialize the class with NDVI data, patch size, and the patches with indices.

        :param ndvi_data: 2D array of NDVI data (downsampled).
        :param patch_size: Tuple (height, width) of patch size.
        :param target_shape: Tuple (height, width) of the entire image shape.
        :param patches_with_indices: List of patches with their respective indices (i, j).
        """
        self.ndvi_data = ndvi_data
        self.patch_size = patch_size
        self.target_shape = target_shape
        self.patches_with_indices = patches_with_indices

    def generate_patch_image(self):
        """
        Generate an image of patches with the rest of the image filled with value 2.

        :return: The generated image as a 2D array.
        """
        # Create an empty image filled with 2s
        image = np.full(self.target_shape, 2, dtype=int)

        # Iterate through the patches and place them in the image
        for patch_info in self.patches_with_indices:
            patch = patch_info['ndvi_patch']
            i, j = patch_info['index']

            # Create a binary version of the patch: 0 for zeroes, 1 for positive values
            binary_patch = np.where(patch > 0, 1, 0)

            # Insert the binary patch into the correct position in the image
            image[i:i + self.patch_size[0], j:j + self.patch_size[1]] = binary_patch

        return image


def downsample_data(data, target_shape, thresh):
    """
    Downsample a 2D or 3D array to the target shape

    :param data: A 2D (height x width) or 3D (height x width x bands) array
    :param target_shape: Tuple of (height, width)
    :param thresh: threshold value to just keep the foliage
    :return: downsampled data of target_shape
    """
    # print('Downsampling data with shape', data.shape)
    zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])

    if data.ndim == 3:  # for 3D hyperspectral data
        zoom_factors += (1, )   # no zoom for the third axis (bands)

    image = zoom(data, zoom_factors, order=3)  # cubic interpolation

    # apply threshold only to ndvi (2D) data
    if data.ndim == 2:
        image[image <= thresh] = 0

    return image


class PatchCreator:
    def __init__(self, ndvi, raw, first, second, patch_size, target_shape, max_zeroes, thresh):
        """
        Initialize the class with the data and downsample to the target shape

        :param ndvi: 2D array of (height x width) of NDVI values
        :param raw: 3D arrray of (height x width x bands) of raw hyperspectral data
        :param first: 3D arrray of (height x width x bands) of the first derivative of the raw data
        :param second: 3D arrray of (height x width x bands) of the second derivative of the raw data
        :param patch_size: Tuple (height, width) of the patch size of a plant
        :param target_shape: Tuple (height, width) to downsample to
        :param max_zeroes: Maximum allowed zeroes in a patch
        :param thresh: threshold value to just keep the foliage
        """

        # Downsample the data
        self.ndvi = downsample_data(ndvi, target_shape, thresh)
        self.raw = downsample_data(raw, target_shape, thresh)
        self.first = downsample_data(first, target_shape, thresh)
        self.second = downsample_data(second, target_shape, thresh)
        self.patch_size = patch_size
        self.max_zeroes = max_zeroes
        self.thresh = thresh

    def create_patch(self, row, col):
        """
        Create a patch of NDVI and the corresponding hyperspectral daa and check if they are valid

        :param row: Row index from the top left corner of the data
        :param col: Column index from the top left corner of the data
        :return: Dictionary containing the ndvi and the hyperspectral patch and the indices
        """
        # print('Creating Patches for row-col', row, col)
        # NDVI patch
        nvdi_patch = self.ndvi[row: row + self.patch_size[0], col: col + self.patch_size[1]]

        # check if valid
        if np.sum(nvdi_patch == 0) > self.max_zeroes:
            return None

        # Hyperspectral patches
        raw_patch = self.raw[row: row + self.patch_size[0], col: col + self.patch_size[1], :]
        first_patch = self.first[row: row + self.patch_size[0], col: col + self.patch_size[1], :]
        second_patch = self.second[row: row + self.patch_size[0], col: col + self.patch_size[1], :]

        # concatenate the hyperspectral patches
        concatenated_hyper_patch = np.concatenate((raw_patch, first_patch, second_patch), axis=-1)

        return {
            'ndvi_patch': nvdi_patch,
            'hyperspectral_patch': concatenated_hyper_patch,
            'index': (row, col)
        }


    def create_patches(self):
        """
        Genearate patches for the entire image

        :return: List of dictionaries containing NDVi and the hyperspectral patches and the indices
        """
        patches = []
        futures = []

        # with ProcessPoolExecutor() as executor:
        with ThreadPoolExecutor() as executor:
            for i in range(0, self.ndvi.shape[0] - self.patch_size[0] + 1, self.patch_size[0]):
                for j in range(0, self.ndvi.shape[1] - self.patch_size[1] + 1, self.patch_size[1]):
                    # print('Working on Index', i, j)
                    futures.append(executor.submit(self.create_patch, i, j))

            for future in futures:
                patch = future.result()
                if patch:
                    patches.append(patch)

        return patches


def get_keypoints_from_region(image, region):
    """
    Extracts SIFT keypoints and descriptors from a specific region of an image.
    Args:
    - image: The input image.
    - region: A tuple of (x, y, width, height) specifying the region.

    Returns:
    - keypoints: Keypoints detected in the region.
    - descriptors: Descriptors for the keypoints.
    """
    x, y, w, h = region
    roi = image[y:y + h, x:x + w]  # Crop the region of interest (ROI)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and descriptors in the region
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_roi, None)

    # Adjust keypoint positions to original image coordinates
    for kp in keypoints:
        kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

    return keypoints, descriptors


def match_features_and_align(image1, image2, keypoints1, descriptors1):
    """
    Matches SIFT features between two images and computes a homography to align them.
    Args:
    - image1: The first image (base image).
    - image2: The second image (to be aligned).
    - keypoints1: Keypoints detected in the first image.
    - descriptors1: Descriptors for the keypoints in the first image.

    Returns:
    - aligned_image: The aligned second image.
    """
    # Convert the second image to grayscale
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and descriptors in the second image
    sift = cv2.SIFT_create()
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Use FLANN-based matcher to find matching features
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use homography matrix to warp the second image
    height, width, _ = image1.shape
    aligned_image = cv2.warpPerspective(image2, H, (width, height))

    return aligned_image, H, good_matches, keypoints2


# align two images with given fixed points
def align_image_with_reference(src_img, dst_img, src_pts, dst_pts):
    h_matrix, status = cv2.findHomography(dst_pts, src_pts)
    height, width = src_img.shape[:2]
    aligned_image = cv2.warpPerspective(dst_img, h_matrix, (width, height))
    return aligned_image


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
    # print(f"Processing file: {file_first.split('/')[-1]}")
    img1 = read_hyper(file_first)[0]
    img1 = img1[:, :, idx]
    # print(f"Processing file: {file_second.split('/')[-1]}")
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

    # keep same amount of the labels as of the infected
    idx_0 = np.where(label == 0)[0]
    idx_3 = np.where(label == 3)[0]
    idx_4 = np.where(label == 4)[0]
    idx_6 = np.where(label == 6)[0]

    idx_counts = np.array([len(idx_0), len(idx_3), len(idx_4), len(idx_6)])
    zero_loc = np.where(idx_counts == 0)[0]
    no_zero_min = np.min(np.delete(idx_counts, zero_loc))

    if zero_loc.size > 0 and no_zero_min > 10000:
        no_zero_min = 10000

    random.seed(10)
    idx_0 = random.sample(list(idx_0), no_zero_min) if len(idx_0) > no_zero_min \
        else random.sample(list(idx_0), len(idx_0))
    idx_3 = random.sample(list(idx_3), no_zero_min) if len(idx_3) > no_zero_min \
        else random.sample(list(idx_3), len(idx_3))
    idx_4 = random.sample(list(idx_4), no_zero_min) if len(idx_4) > no_zero_min \
        else random.sample(list(idx_4), len(idx_4))
    idx_6 = random.sample(list(idx_6), no_zero_min) if len(idx_6) > no_zero_min \
        else random.sample(list(idx_6), len(idx_6))

    list_label_idx = [idx_0, idx_3, idx_4, idx_6]
    non_empty_list = [np.array(arr) for arr in list_label_idx if len(arr) > 0]

    if non_empty_list:
        label_idx = np.concatenate(non_empty_list)
    else:
        label_idx = np.array([])

    for idx in label_idx:
        filename = filename_base + '_Pixel_' + str(idx)
        save_name = []
        if label[idx] == 0:  # background
            class_dir_loc = os.path.join(save_loc, 'background')
            save_name = os.path.join(class_dir_loc, filename + '.npy')

        if label[idx] == 3:  # healthy
            class_dir_loc = os.path.join(save_loc, 'healthy')
            save_name = os.path.join(class_dir_loc, filename + '.npy')

        if label[idx] == 4:  # infected
            class_dir_loc = os.path.join(save_loc, 'infected')
            save_name = os.path.join(class_dir_loc, filename + '.npy')

        if label[idx] == 6:  # resistant
            class_dir_loc = os.path.join(save_loc, 'resistant')
            save_name = os.path.join(class_dir_loc, filename + '.npy')

        # print(f'Saving file :  {save_name}')
        np.save(save_name, img[idx, :])


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
        'general_dir': '/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY '
                       'Detection/MSU Flights/2023-07-12/data',
        'raw_data': '/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/MSU '
                    'Flights/2023-07-12/data/hyper_data_raw_band_removed',
        'save_dir': '/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/MSU '
                    'Flights/2023-07-12/data/extracted_data',
    }

    return file_directory
