import glob
import keras
import cv2
from utils import *
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

if __name__ == "__main__":
    data_save_name = os.path.join(info()['save_dir'], 'potato_yes_no_data.npz')

    if os.path.exists(data_save_name):
        print('Loading Saved Data')
        data = np.load(data_save_name)
        train_data = data['train_data']
        train_labels = data['train_labels']
        val_data = data['val_data']
        val_labels = data['val_labels']
    else:
        ndvi_files = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized_ndvi', '*.hdr'))
        ndvi_files.sort()
        all_labels = glob.glob(os.path.join(info()['save_dir'], 'labels', '*.npy'))
        all_labels.sort()
        raw_files = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized', '*.hdr'))
        raw_files.sort()

        threshold_to_clear_shadow = 0.8
        kernel = np.ones((3, 3), np.uint8)

        train_data = np.empty((0, 223))
        train_labels = np.empty(0)  # there will be 2 labels, 0 and 1

        val_data = train_data
        val_labels = train_labels

        random.seed(10)
        idx = list(range(len(all_labels)))
        random.shuffle(idx)
        idx_test = idx[15:]
        idx_val = idx[11:15]

        for count in range(len(ndvi_files)):
            if count in idx_test:
                continue

            data = np.squeeze(read_hyper(ndvi_files[count])[0])
            data[data <= threshold_to_clear_shadow] = 0
            data_eroded = cv2.erode(data, kernel, iterations=1).flatten()  # this contains weeds and potato foliage

            label = np.load(all_labels[count]).flatten()
            foliage_with_weeds = np.logical_or(data_eroded, label)
            true_indices = np.where(foliage_with_weeds)[0]
            # find the number of weeds per image
            not_potato_indices = np.where(label[true_indices] == 0)[0]
            # print(f"Number of weeds in Image ", count, " is ", num_weeds)
            if np.sum(not_potato_indices) >= 50000:
                not_potato_indices = not_potato_indices[:50000]

            potato_indices = np.where(label != 0)[0]
            # take same amount of potato labels randomly
            random.seed(10)
            potato_indices = random.sample(list(potato_indices), len(not_potato_indices))

            img = read_hyper(raw_files[count])[0].reshape((1800000, 223))
            combined_indices = np.hstack((potato_indices, not_potato_indices))
            random.seed(10)
            np.random.shuffle(combined_indices)
            img = img[combined_indices, :]
            label = label[combined_indices]
            label[label != 0] = 1  # change the label to 1 for potatoes

            if count in idx_val:
                val_data = np.concatenate((val_data, img), axis=0)
                val_labels = np.concatenate((val_labels, label), axis=0)
            else:
                train_data = np.concatenate((train_data, img), axis=0)
                train_labels = np.concatenate((train_labels, label), axis=0)

        np.savez(data_save_name, train_data=train_data, train_labels=train_labels, val_data=val_data,
                 val_labels=val_labels)

    # input shape of the data
    input_size = train_data.shape[1]

    # create the model to tune
    def create_model(num_units, num_layers, learning_rate, batch_size):
        num_units = round(num_units)
        num_layers = round(num_layers)
        batch_size = 2 ** round(batch_size)

        model = Sequential()
        model.add(Dense(num_units, input_dim=input_size, activation='relu'))
        for _ in range(num_layers - 1):
            model.add(Dense(num_units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=20, batch_size=batch_size, verbose=2, validation_data=(val_data, val_labels), shuffle=True)
        _, accuracy = model.evaluate(test_data, test_labels, verbose=2)
        return -accuracy    # minimize negative accuracy

    def optimized_model(num_units, num_layers, learning_rate, batch_size):
        model = Sequential()
        model.add(Dense(num_units, input_dim=input_size, activation='relu'))
        for _ in range(num_layers - 1):
            model.add(Dense(num_units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=20, batch_size=batch_size, verbose=2,
                  validation_data=(val_data, val_labels), shuffle=True)
        return model

    # num_units=100, num_layers=2, learning_rate=0.001, batch_size=32
    model_save_name = os.path.join(info()['save_dir'], 'model_opt_potato_not_potato.keras')
    if os.path.exists(model_save_name):
        print('Loading Saved Model')
        model = keras.models.load_model(model_save_name)
    else:
        model = optimized_model(num_units=100, num_layers=2, learning_rate=0.001, batch_size=32)
        model.save(model_save_name)
        del train_data, train_labels, val_data, val_labels

    ndvi_files = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized_ndvi', '*.hdr'))
    ndvi_files.sort()
    all_labels = glob.glob(os.path.join(info()['save_dir'], 'labels', '*.npy'))
    all_labels.sort()
    raw_files = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized', '*.hdr'))
    raw_files.sort()
    threshold_to_clear_shadow = 0.8
    kernel = np.ones((3, 3), np.uint8)
    test_data = np.empty((0, 223))
    test_labels = np.empty(0)  # there will be 2 labels, 0 and 1
    idx_test = [15, 13, 1, 18]

    for count in idx_test:
        data = np.squeeze(read_hyper(ndvi_files[count])[0])
        data[data <= threshold_to_clear_shadow] = 0
        data_eroded = cv2.erode(data, kernel, iterations=1).flatten()  # this contains weeds and potato foliage

        label = np.load(all_labels[count]).flatten()
        foliage_with_weeds = np.logical_or(data_eroded, label)
        true_indices = np.where(foliage_with_weeds)[0]

        # find the number of weeds per image
        not_potato_indices = np.where(label[true_indices] == 0)[0]
        potato_indices = np.where(label != 0)[0]
        combined_indices = np.hstack((potato_indices, not_potato_indices))

        img = read_hyper(raw_files[count])[0].reshape((1800000, 223))
        img = img[combined_indices, :]
        label = label[combined_indices]
        label[label != 0] = 1  # change the label to 1 for potatoes

        img_true = np.zeros(2000 * 900)
        img_true[combined_indices] = label

        pred = model.predict(img)
        threshold = 0.5
        pred = np.squeeze((pred >= threshold).astype(int))

        img_pred = np.zeros(2000 * 900)
        img_pred[combined_indices] = pred

        acc = np.sum(np.equal(label, pred)) / len(label)

        plt.subplot(1, 2, 1)
        plt.imshow(img_true.reshape((2000, 900)))
        plt.subplot(1, 2, 2)
        plt.imshow(img_pred.reshape((2000, 900)))
        img_num = raw_files[count].split('_')[-1].split('-')[0]
        plt.suptitle('True and Predictions for Image ' + str(img_num) + '; Accuracy: ' + str(acc))
        plt_save_name = os.path.join(info()['save_dir'], 'True_and_Predictions_for_Image_' + str(img_num) + '.png')
        plt.savefig(plt_save_name)
        plt.close()
