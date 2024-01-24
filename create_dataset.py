import glob
import random
from utils import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def process_file(save_directory, data_kind, data):
    save_loc = os.path.join(save_directory, data_kind)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    # use partial as there is a fixed value input
    partial_processed = partial(save_extracted_data, save_loc=save_loc)

    # use process instead of threads, else the program will run out of memory
    with ProcessPoolExecutor() as executor:
        executor.map(partial_processed, data)


def get_data(dir_raw, dir_first, dir_second, dir_labels, dir_save):
    random.seed(10)

    # 60% for training, 20% for validation and 20% for testing
    random.shuffle(dir_raw)
    split_idx_trn = round(len(dir_raw) * 0.6)
    split_idx_val = round(len(dir_raw) * 0.2)

    train_files_raw = dir_raw[: split_idx_trn]
    train_files_first = dir_first[:, split_idx_trn]
    train_files_second = dir_second[:, split_idx_trn]
    train_files_labels = dir_labels[:, split_idx_trn]

    val_files_raw = dir_raw[split_idx_trn: split_idx_trn + split_idx_val]
    val_files_first = dir_first[split_idx_trn: split_idx_trn + split_idx_val]
    val_files_second = dir_second[split_idx_trn: split_idx_trn + split_idx_val]
    val_files_labels = dir_labels[split_idx_trn: split_idx_trn + split_idx_val]

    test_files_raw = dir_raw[split_idx_trn + split_idx_val:]
    test_files_first = dir_first[split_idx_trn + split_idx_val:]
    test_files_second = dir_second[split_idx_trn + split_idx_val:]
    test_files_labels = dir_labels[split_idx_trn + split_idx_val:]

    # create data split directories
    dir_name = os.path.join(dir_save, 'train_data')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dir_name = os.path.join(dir_save, 'val_data')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dir_name = os.path.join(dir_save, 'test_data')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print('---------- Extracting Training Data ----------')
    data_kind = 'train_raw'
    process_file(save_directory, data_kind, train_files)

    print('---------- Extracting Validation Data ----------')
    data_kind = 'val_raw'
    process_file(save_directory, data_kind, val_files)

    print('---------- Extracting Testing Data ----------')
    data_kind = 'test_raw'
    process_file(save_directory, data_kind, test_files)

    # wavelengths should be the same for all the images collected by the same camera
    wvl = read_hyper(train_files[0])[1]

    print('---------- Saving other relevant Data ----------')
    save_loc = os.path.join(save_directory, 'other_data')
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    save_loc = os.path.join(save_loc, 'other_data.npz')  # .npz is used to save multiple variables
    np.savez(save_loc, train_files=train_files, val_files=val_files, test_files=test_files, wvl=wvl)


if __name__ == "__main__":
    # get raw hyperspectral value files
    raw_data_files = os.path.join(info()['raw_data'], '*.hdr')
    raw_data_files = glob.glob(raw_data_files)
    raw_data_files.sort()

    # get first derivative files
    first_derivative_files = os.path.join(info()['general_dir'], 'first_derivative', '*.hdr')
    first_derivative_files = glob.glob(first_derivative_files)
    first_derivative_files.sort()

    # get first derivative files
    second_derivative_files = os.path.join(info()['general_dir'], 'second_derivative', '*.hdr')
    second_derivative_files = glob.glob(second_derivative_files)
    second_derivative_files.sort()

    # get the labels
    labels_files = os.path.join(info()['save_dir'], 'labels', '*.npy')
    labels_files = glob.glob(labels_files)
    labels_files.sort()

    save_dir = info()['save_dir']

    # get_data(dir_list, save_dir)
