import os
import glob
import numpy as np
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose, LayerNormalization, Dropout, Embedding
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam


# disbale gpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# enable multi-threaded cpu processing
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

print("Available CPUs:", os.cpu_count())
print("Using CPU only:", tf.config.list_physical_devices('GPU') == [])

print("Physical devices:", tf.config.list_physical_devices())


# # import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
#
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB max
# )

# Function to normalize image for visualization
def normalize_img(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

# load the hyperspectral and ndvi images
def load_hsi_ndvi(hsi_path, ndvi_path):
    ndvi = np.squeeze(read_hyper(ndvi_path)[0])

    try:
        match = re.search(r'_L_(\d+)-radiance', ndvi_path)
        img_num = int(match.group(1))
        print('Loading Data for Image', img_num)
    except Exception as e:
        print('Could not print the image number due to :', e)

    hsi = read_hyper(hsi_path)[0]
    if hsi.shape[2] != 223:
        hsi = hsi[:, :, 1:-1]  # remove first and last band to match the 223 bands from previous year
    return hsi, ndvi

# Patch Extraction
def extract_patches(image, patch_size):
    patches = []
    locations = []
    for i in range(0, image.shape[0] - patch_size[0] + 1, patch_size[0]):
        for j in range(0, image.shape[1] - patch_size[1] + 1, patch_size[1]):
            patches.append(image[i:i+patch_size[0], j:j+patch_size[1], :])
            locations.append((i, j))
    return np.array(patches), locations

# NDVI Masking
def apply_ndvi_mask(patches, ndvi_patches, threshold=0.8):
    filtered_patches = []
    filtered_locations = []
    for i in range(len(ndvi_patches)):
        if np.mean(ndvi_patches[i]) > threshold:
            filtered_patches.append(patches[i])
            filtered_locations.append(i)  # Store index of valid patches
    return np.array(filtered_patches), filtered_locations

# Autoencoder Model
# def build_autoencoder(input_shape):
#     input_layer = Input(shape=input_shape)
#     x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
#     x = MaxPooling2D((2, 2), padding='same')(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = MaxPooling2D((2, 2), padding='same')(x)
#     x = Flatten()(x)
#     latent = Dense(128, activation='relu')(x)
#
#     x = Dense(np.prod(input_shape), activation='relu')(latent)
#     # x = Reshape((input_shape[0] // 4, input_shape[1] // 4, input_shape[2] // 4, 64))(x)
#     # x = Reshape((input_shape[0] // 4, input_shape[1] // 4, 64))(x)
#     x = Reshape((input_shape[0] // 4, input_shape[1] // 4, input_shape[2] // 4))(x)  # Fix reshape
#     x = UpSampling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = UpSampling2D((2, 2))(x)
#     output_layer = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)
#
#     return Model(inputs=input_layer, outputs=output_layer)

def build_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    latent = Dense(128, activation='relu')(x)  # Latent representation

    # Compute channels dynamically
    num_channels = input_shape[2]
    channels_after_reshape = (num_channels * (input_shape[0] * input_shape[1])) // (input_shape[0] // 4 * input_shape[1] // 4)

    x = Dense(np.prod(input_shape), activation='relu')(latent)
    x = Reshape((input_shape[0] // 4, input_shape[1] // 4, channels_after_reshape))(x)  # Dynamically computed shape
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    output_layer = Conv2D(input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)

    return Model(inputs=input_layer, outputs=output_layer)


# Batch Processing
def process_images(hsi_paths, ndvi_paths, patch_size):
    all_features = []
    all_labels = []

    autoencoder = None
    encoder = None

    for hsi_path, ndvi_path in zip(hsi_paths, ndvi_paths):
        hsi, ndvi = load_hsi_ndvi(hsi_path, ndvi_path)

        hsi_patches, _ = extract_patches(hsi, patch_size)
        ndvi_patches, _ = extract_patches(ndvi[..., None], patch_size)
        filtered_patches, _ = apply_ndvi_mask(hsi_patches, ndvi_patches)

        print('Length of filtered patches', len(filtered_patches))

        if len(filtered_patches) == 0:
            continue

        scaler = MinMaxScaler()
        reshaped_patches = filtered_patches.reshape(-1, np.prod(filtered_patches.shape[1:]))
        scaled_patches = scaler.fit_transform(reshaped_patches).reshape(filtered_patches.shape)

        # Initialize Autoencoder once
        if autoencoder is None:
            autoencoder = build_autoencoder(patch_size + (223,))
            autoencoder.compile(optimizer="adam", loss="mse")
            encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-4].output)

        # Train Autoencoder
        # autoencoder.fit(scaled_patches, scaled_patches, epochs=10, batch_size=16, validation_split=0.2)
        autoencoder.fit(scaled_patches, scaled_patches, epochs=10, batch_size=8, validation_split=0.2)

        # Extract features
        features = encoder.predict(scaled_patches)
        all_features.append(features)

    # Clustering
    # all_features = np.concatenate(all_features)
    # Convert list to numpy array and reshape to (num_samples, num_features)
    all_features = np.concatenate(all_features, axis=0)  # Stack all feature batches
    all_features = all_features.reshape(all_features.shape[0], -1)  # Flatten spatial dimensions

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(all_features)

    return autoencoder, encoder, kmeans, all_features, labels


# Process and visualize image
def process_and_visualize(hsi_path, ndvi_path, patch_size, encoder, kmeans):
    hsi, ndvi = load_hsi_ndvi(hsi_path, ndvi_path)
    hsi_patches, patch_locations = extract_patches(hsi, patch_size)
    ndvi_patches, _ = extract_patches(ndvi[..., None], patch_size)
    filtered_patches, valid_indices = apply_ndvi_mask(hsi_patches, ndvi_patches)
    valid_locations = [patch_locations[i] for i in valid_indices]

    # Normalize and select RGB bands (113, 70, 26)
    rgb_bands = [113, 70, 26]
    rgb_image = np.stack([normalize_img(hsi[:, :, band]) for band in rgb_bands], axis=-1)

    plt.imshow(rgb_image)
    plt.savefig('raw_23data_16.png', dpi=300)

    plt.imshow(normalize_img(ndvi))
    plt.savefig('ndvi_23data_16.png', dpi=300)

    # scale and encode features for clustering
    scaler = MinMaxScaler()
    reshaped_patches = filtered_patches.reshape(-1, np.prod(filtered_patches.shape[1:]))
    scaled_patches = scaler.fit_transform(reshaped_patches).reshape(filtered_patches.shape)

    features = encoder.predict(scaled_patches)

    features = np.concatenate(features, axis=0)  # Stack all feature batches
    features = features.reshape(features.shape[0], -1)  # Flatten spatial dimensions

    labels = kmeans.fit_predict(features)

    # Reconstruct cluster map for entire image
    cluster_map = np.full((hsi.shape[0], hsi.shape[1]), -1)  # Initialize empty cluster map
    for idx, loc in enumerate(valid_locations):
        cluster_map[loc[0]: loc[0] + patch_size[0], loc[1]: loc[1] + patch_size[1]] = labels[idx]

    # Display the RGB composite
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_image)
    ax.set_title("RGB Composite with Clusters")

    # Overlay the cluster map with transparency
    ax.imshow(cluster_map, cmap="viridis", alpha=0.5)
    # plt.colorbar(label="Cluster Label", ax=ax)
    plt.show()
    plt.savefig('cluster_23data_16.png', dpi=300)


# Predict on New Images
def predict_new_image(hsi_path, ndvi_path, patch_size, autoencoder, encoder, kmeans):
    hsi, ndvi = load_hsi_ndvi(hsi_path, ndvi_path)

    hsi_patches = extract_patches(hsi, patch_size)
    ndvi_patches = extract_patches(ndvi[..., None], patch_size)
    filtered_patches = apply_ndvi_mask(hsi_patches, ndvi_patches)

    scaler = MinMaxScaler()
    reshaped_patches = filtered_patches.reshape(-1, np.prod(filtered_patches.shape[1:]))
    scaled_patches = scaler.fit_transform(reshaped_patches).reshape(filtered_patches.shape)

    features = encoder.predict(scaled_patches)
    predictions = kmeans.predict(features)

    return predictions.reshape(hsi.shape[0] // patch_size[0], hsi.shape[1] // patch_size[1])


if __name__ == "__main__":
    farm_names = ['norota', 'umatilla']

    farm_name = farm_names[0]

    base_path = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/'
                 'data_2024/processed_hyper_data/2024_07_31')
    save_loc = ('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/Potato PVY Detection/'
                'data_2024/processed_hyper_data/2024_07_31/saved_data')

    test = 1

    if test == 1:
        raw_files = glob.glob(os.path.join(base_path, 'test_samples',
                                           farm_name + '*.hdr'))
        raw_files.sort()
        ndvi_files = glob.glob(os.path.join(base_path, 'test_samples_ndvi', farm_name + '*.hdr'))
        ndvi_files.sort()
    else:
        raw_files = glob.glob(os.path.join(base_path, 'radiance_reflectance_smoothed_band_removed_normalized',
                                           farm_name + '*.hdr'))
        raw_files.sort()
        ndvi_files = glob.glob(os.path.join(base_path, 'radiance___normalized_ndvi', farm_name + '*.hdr'))
        ndvi_files.sort()

    # patch_size = (64, 64)
    patch_size = (32, 32)

    # autoencoder, encoder, kmeans, features, labels = process_images(raw_files, ndvi_files, patch_size)


    # ADD THE 23 DATA LOCATION
    raw_files = glob.glob(os.path.join('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                                       'Potato PVY Detection/MSU Flights/2023-07-12/data/'
                                       'smoothed_clipped_normalized', '*.hdr'))
    raw_files.sort()
    ndvi_files = glob.glob(os.path.join('/media/SIDSSD/precisionag/01__Big Projects/Precision Disease Management/'
                                       'Potato PVY Detection/MSU Flights/2023-07-12/data/'
                                       'smoothed_clipped_normalized_ndvi', '*.hdr'))
    ndvi_files.sort()


    # SPECIFY IMAGE NUMBER TO TEST
    img_num = 16
    a = [raw_files[img_num]]
    b = [ndvi_files[img_num]]

    autoencoder, encoder, kmeans, features, labels = process_images(a, b, patch_size)

    process_and_visualize(a[0], b[0], patch_size, encoder, kmeans)


    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    print(f"Silhouette Score: {silhouette}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # np.save("features.npy", features)
    # np.save("labels.npy", labels)







