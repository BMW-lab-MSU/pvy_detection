import glob
import pandas as pd
from scipy.io import savemat
from scipy.signal import convolve
from utils import *

if __name__ == "__main__":
    multi_405 = glob.glob(os.path.join(info()['save_dir'], 'mapir_multi_data', '*405*'))
    multi_405.sort()
    multi_518 = glob.glob(os.path.join(info()['save_dir'], 'mapir_multi_data', '*518*'))
    multi_518.sort()
    multi_550 = glob.glob(os.path.join(info()['save_dir'], 'mapir_multi_data', '*550*'))
    multi_550.sort()
    multi_709 = glob.glob(os.path.join(info()['save_dir'], 'mapir_multi_data', '*709*'))
    multi_709.sort()
    multi_780 = glob.glob(os.path.join(info()['save_dir'], 'mapir_multi_data', '*780*'))
    multi_780.sort()
    multi_905 = glob.glob(os.path.join(info()['save_dir'], 'mapir_multi_data', '*905*'))
    multi_905.sort()

    csv_data = pd.read_csv(os.path.join(info()['save_dir'], 'matlab_data', 'compressed_virus_data.csv'))
    img_num_list = csv_data['21'].tolist()
    pixel_list = csv_data['8183'].tolist()
    label_list = csv_data['0'].tolist()

    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size, 1))

    random.seed(10)
    idx = list(range(len(multi_405)))
    random.shuffle(idx)
    idx_test = idx[15:]
    idx_val = idx[11:15]

    trainX = np.empty((0, 6))
    trainY = np.empty(0)
    trainImg = np.empty(0)
    valX = trainX
    valY = trainY
    valImg = trainImg
    testX = trainX
    testY = trainY
    testImg = trainImg

    for img_count in range(len(multi_405)):
        img_num = int(multi_405[img_count].split('_')[-4])
        print('Processing Image', img_num)
        idx = [index for index, value in enumerate(img_num_list) if value == img_num]
        roi_pixel = [pixel_list[i] for i in idx]
        roi_label = [label_list[i] for i in idx]
        roi_img = np.ones(len(roi_label)) * img_num

        multi_data = np.stack((np.load(multi_405[img_count]).reshape((2000, 900)),
                               np.load(multi_518[img_count]).reshape((2000, 900)),
                               np.load(multi_550[img_count]).reshape((2000, 900)),
                               np.load(multi_709[img_count]).reshape((2000, 900)),
                               np.load(multi_780[img_count]).reshape((2000, 900)),
                               np.load(multi_905[img_count]).reshape((2000, 900))), axis=2)

        down_sampled_img = convolve(multi_data, kernel, mode='valid')[::kernel_size, ::kernel_size].reshape(18000, 6)
        data = down_sampled_img[roi_pixel, :]

        if img_count in idx_val:
            valX = np.concatenate((valX, data), axis=0)
            valY = np.concatenate((valY, roi_label), axis=0)
            valImg = np.concatenate((valImg, roi_img), axis=0)
        elif img_count in idx_test:
            testX = np.concatenate((testX, data), axis=0)
            testY = np.concatenate((testY, roi_label), axis=0)
            testImg = np.concatenate((testImg, roi_img), axis=0)
        else:
            trainX = np.concatenate((trainX, data), axis=0)
            trainY = np.concatenate((trainY, roi_label), axis=0)
            trainImg = np.concatenate((trainImg, roi_img), axis=0)

    save_name_npz = os.path.join(info()['save_dir'], 'mapir_multi_data', 'downsampled_multi.npz')
    save_name_mat = os.path.join(info()['save_dir'], 'mapir_multi_data', 'downsampled_multi.mat')
    np.savez(save_name_npz, trainX=trainX, trainY=trainY, trainImg=trainImg, valX=valX,
             valY=valY, valImg=valImg, testX=testX, testY=testY, testImg=testImg)
    savemat(save_name_mat, {'trainX': trainX, 'trainY': trainY, 'trainImg': trainImg, 'valX': valX,
                            'valY': valY, 'valImg': valImg, 'testX': testX, 'testY': testY, 'testImg': testImg})
