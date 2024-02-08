import glob
import keras
from utils import *
import matplotlib.pyplot as plt
from keras.utils import to_categorical

if __name__ == "__main__":

    idx_test = [7, 15]

    ticks = [0, 1, 2, 3, 4]
    tick_labels = ['background', 'healthy', 'infected', 'resistant', 'other']
    # tick_labels = ['background', 'healthy', 'infected', 'other']

    labels_to_keep = [0, 3, 4, 6]   # 0 bck, 3 neg, 4 pos, 6 res
    # labels_to_keep = [0, 3, 4]  # 0 bck, 3 neg, 4 pos, 6 res
    # labels_to_keep = [3, 4, 6]  # 0 bck, 3 neg, 4 pos, 6 res
    # labels_to_keep = [3, 4]  # 0 bck, 3 neg, 4 pos, 6 res

    model = keras.models.load_model(os.path.join(info()['save_dir'], 'model_4_classes.keras'))

    for i in idx_test:
        all_files = glob.glob(os.path.join(info()['raw_data'], '*.hdr'))
        all_files.sort()
        all_first = glob.glob(os.path.join(info()['general_dir'], 'first_derivative', '*.hdr'))
        all_first.sort()
        all_second = glob.glob(os.path.join(info()['general_dir'], 'second_derivative', '*.hdr'))
        all_second.sort()
        all_labels = glob.glob(os.path.join(info()['save_dir'], 'labels', '*.npy'))
        all_labels.sort()

        idx = list(range(10, 175)) + list(range(185, 244))

        # TEST DATA ACCUMULATION

        print(f"Processing Test Data, Index number: {i}, filename: {all_files[i]}")
        img0 = read_hyper(all_files[i])[0]
        img1 = read_hyper((all_first[i]))[0]
        img1 = img1[:, :, idx]
        img2 = read_hyper(all_second[i])[0]
        img2 = img2[:, :, idx]
        img = np.concatenate((img0, img1, img2), axis=2)
        del img0, img1, img2
        label = np.load(all_labels[i]).flatten()
        to_keep_idx = np.where(np.isin(label, labels_to_keep))[0]
        # y_val = to_categorical(label[to_keep_idx])
        y_val = to_categorical(label)
        y_val = y_val[:, labels_to_keep]
        img = img.reshape((1800000, 224 * 3))
        # img = img[to_keep_idx, :]

        predicted = model.predict(img)

        # Convert predictions and true labels to class labels
        predicted_classes = np.argmax(predicted, axis=1)
        true_classes = np.argmax(y_val, axis=1)

        pp = (np.ones(predicted_classes.shape) * np.max(ticks)).astype(int)
        pp[to_keep_idx] = predicted_classes[to_keep_idx]
        tt = (np.ones(true_classes.shape) * np.max(ticks)).astype(int)
        tt[to_keep_idx] = true_classes[to_keep_idx]

        pp = pp.reshape(2000, 900)
        tt = tt.reshape(2000, 900)

        image_name = all_files[i].split('/')[-1].split('-')[0]

        # plot and save the true labeled image
        plt.imshow(tt.transpose(), cmap='viridis')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.title('True Label for ' + image_name)
        plt.savefig(
            os.path.join(info()['save_dir'], 'predictions', 'true_labels_for_image_' + str(i) + '_with_classes_'
                         + ''.join(map(str, labels_to_keep)) + '.png'))
        plt.close()

        # plot and save the predicted labeled image
        plt.imshow(pp.transpose(), cmap='viridis')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        plt.title('Predicted Label for ' + image_name)
        plt.savefig(
            os.path.join(info()['save_dir'], 'predictions', 'predicted_labels_for_image_' + str(i) + '_with_classes_'
                         + ''.join(map(str, labels_to_keep)) + '.png'))
        plt.close()

