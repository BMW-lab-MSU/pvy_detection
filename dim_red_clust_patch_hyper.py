import os
import xml.etree.ElementTree as ET
import glob
import numpy as np
import keras
import random
import cv2
import csv
import pandas as pd
import umap
import joblib

from utils import *
import seaborn as sns
import tensorflow as tf
from scipy.signal import convolve
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from concurrent.futures import ProcessPoolExecutor
from scipy.io import loadmat, savemat
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU, Input
from keras.utils import to_categorical, plot_model
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, SpectralClustering, Birch, AgglomerativeClustering, AffinityPropagation, \
    OPTICS
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import kneighbors_graph
import faiss
import hdbscan


def save_labels(csv_name, label_files):
    csv_name = csv_name
    csv_header = ['img_num', 'pixel', 'label']  # 0 for healthy, 1 for infected

    with open(csv_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

    for count in range(len(label_files)):
        img_num = int(re.search(r'_(\d+)-batch', label_files[count]).group(1))
        # print('Saving Data for', img_num)
        img = np.load(labels_files[count]).reshape(-1)
        idx_healthy = np.where(img == 3)[0]
        idx_infected = np.where(img == 4)[0]

        with open(csv_name, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(np.vstack(([img_num] * len(idx_healthy), idx_healthy, [0] * len(idx_healthy))).T)
            writer.writerows(np.vstack(([img_num] * len(idx_infected), idx_infected, [1] * len(idx_infected))).T)


def save_leaves_data(files, save_name):
    data = np.empty((0, 223))
    train_idx = np.empty(0)
    train_pix = np.empty(0)
    for count in range(len(files)):
        img_num = int(re.search(r'_(\d+)-Reflectance', files[count]).group(1))
        print('Saving Data for', img_num)
        img = read_hyper(files[count])[0].reshape((1800000, 223))
        idx = np.where(img_nums == img_num)[0]
        pix = pixels[idx]
        label = labels[idx]
        img = img[pix, :]

        leaves_save_name = os.path.join(save_name, 'leaves_shadows_' + str(img_num) + '.npz')
        np.savez(leaves_save_name, img=img, pix=pix, label=label)

        if img_num in train_img:
            data = np.concatenate((data, img), axis=0)
            train_idx = np.concatenate((train_idx, idx), axis=0)
            train_pix = np.concatenate((train_pix, pix), axis=0)

    train_data_name = os.path.join(save_name, 'train_data_leaves_shadows.npz')
    np.savez(train_data_name, data=data, train_idx=train_idx, train_pix=train_pix)
    return data


def perform_dim_reduction(data, n, batch, base_name):
    # PCA
    print('Doing  PCA')
    ipca = IncrementalPCA(n_components=n, batch_size=batch)
    for i in range(0, data.shape[0], batch):
        print('Processing from', i, 'to', i + batch)
        ipca.partial_fit(data[i: i + batch])
    reduced_data_pca = ipca.transform(data)
    # joblib.dump(ipca, 'LOCATION.pkl')
    # ipca = joblib.load('LOCATION.pkl')
    # new_data_pca =  ipca.transform(new_data)
    save_name = os.path.join(base_name, 'reduced_data_pca.npy')
    np.save(save_name, reduced_data_pca)

    # Truncated SVD - useful for sparse data
    print('Doing  Truncated SVD')
    svd = TruncatedSVD(n_components=n)
    reduced_data_svd = svd.fit_transform(data)
    # new_data_svd = svd.transform(new_data)
    save_name = os.path.join(base_name, 'reduced_data_svd.npy')
    np.save(save_name, reduced_data_svd)

    # Random Projections - useful for high dimensional data
    print('Doing  Random Projection')
    rp = SparseRandomProjection(n_components=n)
    reduced_data_rp = rp.fit_transform(data)
    # new_data_rp = rp.transform(new_data)
    save_name = os.path.join(base_name, 'reduced_data_rp.npy')
    np.save(save_name, reduced_data_rp)

    # Autoencoders
    print('Doing  Autoencoder')
    input_dim = data.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(n, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.2)

    reduced_data_ae = encoder.predict(data)
    # autoencoder.save('LOCATION.h5')
    # encoder.save('LOCATION.h5')
    # encoder = load_model('LOCATION.h5')
    # new_data_ae = encoder.predict(new_data)
    save_name = os.path.join(base_name, 'reduced_data_ae.npy')
    np.save(save_name, reduced_data_ae)

    # takes very long time, so commented out
    # # tSNE - takes very long time - maybe run with 2 components
    # # 'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.
    # print('Doing  TSNE')
    # tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
    # reduced_data_tsne = tsne.fit_transform(data)
    # save_name = os.path.join(base_name, 'reduced_data_tsne.npy')
    # np.save(save_name, reduced_data_tsne)
    #
    # # umap
    # print('Doing  UMAP')
    # reducer = umap.UMAP(n_components=2)
    # reduced_data_umap = reducer.fit_transform(data)
    # save_name = os.path.join(base_name, 'reduced_data_umap.npy')
    # np.save(save_name, reduced_data_umap)


def perform_clustering(multiple_data, data_name, base_name):
    for i in range(len(multiple_data)):
        scalar = StandardScaler()
        data_scaled = scalar.fit_transform(multiple_data[i])

        # mini batch kmeans
        cluster_name = 'kmeans'
        print('Performing', cluster_name)
        mini_batch_kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=20000)
        cluster_kmeans = mini_batch_kmeans.fit_predict(data_scaled)
        # save scaler and mini_batch_kmeans by pkl
        # new_data_scaled = scaler.transform(new_data)
        # new_cluster_kmeans = mini_batch_kmeans.predict(new_data_scaled)
        save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')
        np.save(str(save_name), cluster_kmeans)

        # hdbscan
        cluster_name = 'hdbscan'
        print('Performing', cluster_name)
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
        cluster_hdbscan = hdbscan_clusterer.fit_predict(data_scaled)
        # save scaler and hdbscan_clusterer by pkl
        # new_data_scaled = scaler.transform(new_data)
        # HDBSCAN does not have a direct predict method like KMeans
        # Instead, use the approximate_predict function
        # _, new_cluster_hdbscan = hdbscan.approximate_predict(hdbscan_clusterer, new_data_scaled)
        save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')
        np.save(str(save_name), cluster_hdbscan)

        # # spectral clustering - takes long
        # print('Performing', cluster_name)
        # cluster_name = 'spectral_clustering'
        # spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_jobs=1)
        # cluster_spectral = spectral.fit_predict(data_scaled)
        # save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')
        # np.save(str(save_name), cluster_spectral)

        # # birch - data too big?
        # cluster_name = 'birch'
        # print('Performing', cluster_name)
        # birch = Birch(n_clusters=2)
        # cluster_birch = birch.fit_predict(data_scaled)
        # save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')
        # np.save(str(save_name), cluster_birch)

        # # Agglomerative - takes long
        # cluster_name = 'agglomerative'
        # print('Performing', cluster_name)
        # connectivity = kneighbors_graph(data_scaled, n_neighbors=30, include_self=False)
        # # agglomerative clustering with connectivity constraint
        # agg_clustering = AgglomerativeClustering(n_clusters=2, connectivity=connectivity)
        # cluster_agglomerative = agg_clustering.fit_predict(data_scaled)
        # save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')
        # np.save(str(save_name), cluster_agglomerative)

        # # Affinity Propagation - too big?
        # cluster_name = 'affinity'
        # print('Performing', cluster_name)
        # affinity_propagation = AffinityPropagation(random_state=42)
        # cluster_affinity = affinity_propagation.fit_predict(data_scaled)
        # save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')
        # np.save(str(save_name), cluster_affinity)

        # # OPTICS - takes long too
        # cluster_name = 'optics'
        # print('Performing', cluster_name)
        # optics = OPTICS(min_samples=50, xi=0.05, min_cluster_size=0.1)
        # cluster_optics = optics.fit_predict(data_scaled)
        # save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')
        # np.save(str(save_name), cluster_optics)

        # FAISS
        try:
            cluster_name = 'faiss'
            print('Performing', cluster_name)
            kmeans = faiss.Kmeans(d=data_scaled.shape[1], k=2, niter=20, verbose=True)
            kmeans.train(data_scaled.astype(np.float32))
            _, cluster_faiss = kmeans.index.search(data_scaled.astype(np.float32), 1)
            cluster_faiss = cluster_faiss.flatten()
            # joblib.dump(scaler, 'scaler.pkl')
            # faiss.write_index(kmeans.index, 'faiss_kmeans_index.bin')
            # new_data_scaled = scaler.transform(new_data)
            # _, new_cluster_faiss = index.search(new_data_scaled.astype(np.float32), 1)
            # new_cluster_faiss = new_cluster_faiss.flatten()
            save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')
            np.save(str(save_name), cluster_faiss)
        except Exception as e:
            print('Could not perform', cluster_name, 'for', data_name[i], '\nThe error is:', e)


def save_clustered_images(img_num, clustered_files, counter):
    print('Saving Image', img_num)
    leaves_save_name = os.path.join(info()['save_dir'], 'clustered', 'leaves_shadows_' + str(img_num) + '.npz')
    a = np.load(leaves_save_name)
    aa = a['img']
    base_name = os.path.join(info()['save_dir'], 'clustered', 'dim_reduced', 'clustered', str(img_num))
    if not os.path.exists(base_name):
        os.makedirs(base_name)
    # cluster_name = 'kmeans'
    # data_name = ['ae', 'pca', 'rp', 'svd']
    # i = 1
    # save_name = os.path.join(base_name, cluster_name + '_' + data_name[i] + '.npy')

    for f in range(len(clustered_files)):
        save_name = clustered_files[f]
        kpca = np.load(save_name)
        test_cluster = kpca[counter: counter + aa.shape[0]]
        pix = a['pix']
        testimg = np.ones((2000, 900)) * 2
        testimg = testimg.reshape(-1)
        testimg[pix] = test_cluster
        testimg = testimg.reshape((2000, 900))
        plt.imshow(testimg)
        fname = os.path.splitext(os.path.basename(clustered_files[f]))[0]
        save_name = os.path.join(base_name, fname + '.tiff')
        plt.savefig(save_name)
        plt.close()

    return counter + aa.shape[0]


if __name__ == "__main__":
    # get the labels    # 3 (healthy), 4 (infected), 5 (unknown), 6 (resistant)
    labels_files = os.path.join(info()['save_dir'], 'labels', '*.npy')
    labels_files = glob.glob(labels_files)
    labels_files.sort()

    csv_file_name = os.path.join(info()['save_dir'], 'clustered', 'susceptible_indices.csv')

    need_to_save_labels = 0
    if need_to_save_labels:
        save_labels(csv_file_name, labels_files)

    train_img = [21, 22, 26, 31, 34, 39, 40, 42, 43, 44, 47, 48]

    df = pd.read_csv(csv_file_name)

    img_nums = np.array(df['img_num'])
    pixels = np.array(df['pixel'])
    labels = np.array(df['label'])

    raw_files = glob.glob(os.path.join(info()['general_dir'], 'smoothed_clipped_normalized', '*.hdr'))
    raw_files.sort()

    train_save_dir = os.path.join(info()['save_dir'], 'clustered')

    need_to_save_leaves = 0
    if need_to_save_leaves:
        training_data = save_leaves_data(raw_files, train_save_dir)
    else:
        train_save_name = os.path.join(train_save_dir, 'train_data_leaves_shadows.npz')
        training_data = np.load(train_save_name)
        training_data = training_data['data']

    need_to_do_dim_red = 0
    base_save_name = os.path.join(info()['save_dir'], 'clustered', 'dim_reduced')
    n_components = 10
    batch_size = 20000
    if need_to_do_dim_red:
        perform_dim_reduction(training_data, n_components, batch_size, base_save_name)
    else:
        ae = np.load(os.path.join(base_save_name, 'reduced_data_ae.npy'))
        pca = np.load(os.path.join(base_save_name, 'reduced_data_pca.npy'))
        rp = np.load(os.path.join(base_save_name, 'reduced_data_rp.npy'))
        svd = np.load(os.path.join(base_save_name, 'reduced_data_svd.npy'))

    all_reduced = [ae, pca, rp, svd]
    data_names = ['ae', 'pca', 'rp', 'svd']
    base_cluster_save_name = os.path.join(info()['save_dir'], 'clustered', 'dim_reduced', 'clustered')

    # multiple_data = [ae, pca, rp, svd]
    # data_name = ['ae', 'pca', 'rp', 'svd']
    # base_name = os.path.join(info()['save_dir'], 'clustered', 'dim_reduced', 'clustered')

    need_to_cluster = 0
    if need_to_cluster:
        perform_clustering(all_reduced, data_names, base_cluster_save_name)

    clustered_data_files = glob.glob(os.path.join(base_cluster_save_name, '*.npy'))

    # save the images
    start_idx = 0
    for i in range(len(train_img)):
        start_idx = save_clustered_images(train_img[i], clustered_data_files, start_idx)

    train_csv = os.path.join(info()['save_dir'], 'clustered', 'leaves_shadows_train.csv')
