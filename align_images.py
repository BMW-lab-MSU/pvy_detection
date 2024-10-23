import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *


if __name__ == "__main__":

    # change the data/location of the multidata
    files_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                  'Potato PVY Detection/data_2024/2024_07_10_potatoes_diseased_hort_farm/'
                  '3_dsm_ortho/2_mosaic/*.tif')
    # get the date from the file directory to save the aligned images
    pattern = r'\d{4}_\d{2}_\d{2}'
    save_date = re.search(pattern, files_loc).group(0)

    files = glob.glob(files_loc)

    save_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                'Potato PVY Detection/data_2024/aligned_data/')

    clipped_rgb_file = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                        'Potato PVY Detection/data_2024/clipped_data/bounded_2024_07_05_rgb.tif')

    ref_img = cv2.imread(clipped_rgb_file, cv2.IMREAD_COLOR)
    src_pts = np.array([[438, 193], [2012, 220], [1766, 4330], [211, 4259]], dtype=np.float32)

    regions = [(414, 170, 46, 46), (1990, 197, 45, 45), (1743, 4303, 51, 53), (186, 4234, 47, 50)]
    # regions = [(420, 178, 33, 32), (1998, 204, 27, 31), (1754, 4314, 26, 31), (194, 4243, 30, 31)]

    ######### test
    image1 = ref_img
    image2 = cv2.imread(files[1])

    # Extract keypoints and descriptors from each region and combine them
    keypoints1_all = []
    descriptors1_all = []

    for region in regions:
        keypoints, descriptors = get_keypoints_from_region(image1, region)
        keypoints1_all.extend(keypoints)
        if len(descriptors1_all) == 0:
            descriptors1_all = descriptors
        else:
            descriptors1_all = np.vstack((descriptors1_all, descriptors))

    # Align the second image based on the features from the first image
    aligned_image, H, matches, keypoints2 = match_features_and_align(image1, image2, keypoints1_all, descriptors1_all)

    # Display the result
    plt.figure(figsize=(15, 10))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), plt.title('Original Image 1')
    plt.subplot(122), plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)), plt.title('Aligned Image 2')
    plt.show()




#################################################
    if save_date == '2024_07_05':
        dst_pts = np.array([[1379, 1541], [2954, 1568], [2708, 5680], [1153, 5608]], dtype=np.float32)
    elif save_date == '2024_07_10':
        dst_pts = np.array([[1540, 1809], [3596.5, 1869], [3205, 7251], [1167, 7130]], dtype=np.float32)

    for file in files:
        pattern = r'_([^_/]+)\.tif'
        match = re.search(pattern, file)
        img_band = match.group(1).replace(" ", "")
        print(img_band)

        if img_band == "group1":
            img = cv2.imread(file, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        aligned_img = align_image_with_reference(ref_img, img, src_pts, dst_pts)

        save_name = save_loc + '/' + save_date + '_' + img_band + '.png'
        cv2.imwrite(save_name, aligned_img)

        if img_band == 'nir':
            nir = aligned_img.astype(np.float32)
        elif img_band == 'red':
            red = aligned_img.astype(np.float32)
        elif img_band == 'group1':
            rgb_image = aligned_img.astype(np.float32)

        plt.figure()
        plt.title('Aligned ' + img_band)
        plt.imshow(aligned_img)

        # plt.title(img_band)
        # plt.imshow(img)
        plt.show()

    ndvi = (nir - red) / (nir + red + 1e-6)  # adding small epsilon to avoid division by zero

    threshold = 0.2
    vegetation_mask = np.where(ndvi > threshold, 1, 0).astype(np.uint8)

    # apply the mask to the original image
    vegetation_only = cv2.bitwise_and(rgb_image, rgb_image, mask=vegetation_mask)

    save_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                'Potato PVY Detection/data_2024/aligned_vegetation/')
    save_name = save_loc + '/' + save_date + '_vegetation_only.png'
    cv2.imwrite(save_name, vegetation_only)






################### tests below

    nir_file = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                  'Potato PVY Detection/data_2024/2024_07_05_potatoes_diseased_hort_farm/'
                  '3_dsm_ortho/2_mosaic/2024_07_05_potatoes_diseased_hort_farm_transparent_mosaic_nir.tif')

    red_file = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                'Potato PVY Detection/data_2024/2024_07_05_potatoes_diseased_hort_farm/'
                '3_dsm_ortho/2_mosaic/2024_07_05_potatoes_diseased_hort_farm_transparent_mosaic_red.tif')

    rgb_image = cv2.imread(clipped_rgb_file, cv2.IMREAD_COLOR)
    nir_image = cv2.imread(nir_file, cv2.IMREAD_GRAYSCALE)
    red_image = cv2.imread(red_file, cv2.IMREAD_GRAYSCALE)

    src_pts = np.array([[438, 193], [2012, 220], [1766, 4330], [211, 4259]], dtype=np.float32)
    # dst_pts = np.array([[1378.5, 1541.5], [2953, 1568], [2708, 5680], [1153, 5608]], dtype=np.float32)
    dst_pts = np.array([[1379, 1541], [2954, 1568], [2708, 5680], [1153, 5608]], dtype=np.float32)

    aligned_nir = align_image_with_reference(rgb_image, nir_image, src_pts, dst_pts)
    aligned_red = align_image_with_reference(rgb_image, red_image, src_pts, dst_pts)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('RGB Image')
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('NIR Image')
    plt.imshow(nir_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Aligned NIR Image')
    plt.imshow(aligned_nir, cmap='gray')
    plt.show()

    nir = aligned_nir.astype(np.float32)
    red = aligned_red.astype(np.float32)

    ndvi = (nir - red) / (nir + red + 1e-6)     # adding small epsilon to avoid division by zero

    # Normalize NDVI to range [0, 255] for visualization
    ndvi_normalized = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX)

    threshold = 0.2
    vegetation_mask = np.where(ndvi > threshold, 1, 0).astype(np.uint8)

    # apply the mask to the original image
    vegetation_only = cv2.bitwise_and(rgb_image, rgb_image, mask=vegetation_mask)

    # Display the NDVI and the isolated vegetation
    plt.figure(figsize=(12, 6))

    # NDVI Visualization
    plt.subplot(1, 3, 1)
    plt.imshow(ndvi_normalized, cmap='gray')
    plt.title('NDVI')

    # Vegetation Mask
    plt.subplot(1, 3, 2)
    plt.imshow(vegetation_mask, cmap='gray')
    plt.title('Vegetation Mask')

    # Isolated Vegetation
    plt.subplot(1, 3, 3)
    plt.imshow(vegetation_only)
    plt.title('Vegetation Isolated')

    plt.show()

    save_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                'Potato PVY Detection/data_2024/clipped_data/bounded_vegetation_2024_07_05_rgb.png')
    cv2.imwrite(save_loc, vegetation_only)



