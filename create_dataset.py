import glob
import random
from utils import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def process_file(dir_name, data):
    idx = np.arange(len(data[0]))

    # use partial as there is a fixed value input
    partial_processed = partial(save_data_by_pixels, file_info=data, save_loc=dir_name)

    # use process instead of threads, else the program will run out of memory
    with ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(partial_processed, idx)


def get_data(dir_raw, dir_first, dir_second, dir_labels, dir_save):
    random.seed(10)
    shuffled_indices = list(range(len(dir_raw)))
    random.shuffle(shuffled_indices)

    # Use the same shuffled index list for all four lists
    dir_raw = [dir_raw[i] for i in shuffled_indices]
    dir_first = [dir_first[i] for i in shuffled_indices]
    dir_second = [dir_second[i] for i in shuffled_indices]
    dir_labels = [dir_labels[i] for i in shuffled_indices]

    # 60% for training, 20% for validation and 20% for testing
    split_idx_trn = round(len(dir_raw) * 0.6)
    split_idx_val = round(len(dir_raw) * 0.2)

    train_files_raw = dir_raw[: split_idx_trn]
    train_files_first = dir_first[: split_idx_trn]
    train_files_second = dir_second[: split_idx_trn]
    train_files_labels = dir_labels[: split_idx_trn]
    train_files = [train_files_raw, train_files_first, train_files_second, train_files_labels]

    val_files_raw = dir_raw[split_idx_trn: split_idx_trn + split_idx_val]
    val_files_first = dir_first[split_idx_trn: split_idx_trn + split_idx_val]
    val_files_second = dir_second[split_idx_trn: split_idx_trn + split_idx_val]
    val_files_labels = dir_labels[split_idx_trn: split_idx_trn + split_idx_val]
    val_files = [val_files_raw, val_files_first, val_files_second, val_files_labels]

    test_files_raw = dir_raw[split_idx_trn + split_idx_val:]
    test_files_first = dir_first[split_idx_trn + split_idx_val:]
    test_files_second = dir_second[split_idx_trn + split_idx_val:]
    test_files_labels = dir_labels[split_idx_trn + split_idx_val:]
    test_files = [test_files_raw, test_files_first, test_files_second, test_files_labels]

    # create data split directories
    dir_train = os.path.join(dir_save, 'train_data')
    if not os.path.exists(dir_train):
        os.makedirs(dir_train)

    dir_val = os.path.join(dir_save, 'val_data')
    if not os.path.exists(dir_val):
        os.makedirs(dir_val)

    dir_test = os.path.join(dir_save, 'test_data')
    if not os.path.exists(dir_test):
        os.makedirs(dir_test)

    print('---------- Extracting Training Data Files ----------')
    process_file(dir_train, train_files)

    print('---------- Extracting Validation Data Files ----------')
    process_file(dir_val, val_files)

    print('---------- Extracting Testing Data Files ----------')
    process_file(dir_test, test_files)


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

    get_data(raw_data_files, first_derivative_files, second_derivative_files, labels_files, save_dir)
