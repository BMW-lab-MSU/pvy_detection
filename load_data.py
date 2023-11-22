import glob
import random
from utils import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def process_file(save_directory, data_kind, data):
    save_loc = os.path.join(save_directory, data_kind)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    partial_processed = partial(save_extracted_data, save_loc=save_loc)

    with ProcessPoolExecutor() as executor:
        executor.map(partial_processed, data)


def get_data(header_directory, save_directory):
    random.seed(10)

    random.shuffle(header_directory)
    split_idx_trn = round(len(header_directory) * 0.6)
    split_idx_val = round(len(header_directory) * 0.2)

    train_files = header_directory[: split_idx_trn]
    val_files = header_directory[split_idx_trn: split_idx_trn + split_idx_val]
    test_files = header_directory[split_idx_trn + split_idx_val:]

    print('---------- Extracting Training Data ----------')
    data_kind = 'train_raw'

    process_file(save_directory, data_kind, train_files)

    print('---------- Extracting Validation Data ----------')
    data_kind = 'val_raw'
    process_file(save_directory, data_kind, val_files)

    print('---------- Extracting Testing Data ----------')
    data_kind = 'test_raw'
    process_file(save_directory, data_kind, test_files)

    wvl = read_hyper(train_files[0])[1]

    print('---------- Saving other relevant Data ----------')
    save_loc = os.path.join(save_directory, 'other_data')
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    save_loc = os.path.join(save_loc , 'other_data.npz')
    np.savez(save_loc, train_files=train_files, val_files=val_files, test_files=test_files, wvl=wvl)


if __name__ == "__main__":
    info = info()
    data_folder = info['raw_data']
    save_dir = info['save_dir']
    ext_data_folder = os.path.join(data_folder, '*.hdr')
    dir_list = glob.glob(ext_data_folder)
    dir_list.sort()
    get_data(dir_list, save_dir)

