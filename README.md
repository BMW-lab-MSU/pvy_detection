# PVY Detection
Potato Virus Y detection using a hyperspectral camera

## Citation
This software was used for the publication titled *"Unmanned Aerial Vehicle-Based Hyperspectral Imaging for Potato Virus Y Detection: Machine Learning Insights"* with authors S. B. Nesar, P. W. Nugent, N. K. Zidack, and B. M. Whitaker. Please cite the paper when using this code for research purposes.

The data required to run this software is archived on Zenodo at: 
- https://doi.org/10.5281/zenodo.15417758 and
- https://doi.org/10.5281/zenodo.15420134.

## Project Overview
This repository contains the code for detecting **Potato Virus Y (PVY)** using hyperspectral data. The primary objective is to classify potato plants based on their infection status using machine learning algorithms, leveraging data collected from UAVs (Unmanned Aerial Vehicles) equipped with hyperspectral imaging sensors.

### Key Features:
- **Data Processing**: Code for handling and preprocessing hyperspectral images, including feature generation and dataset splitting.
- **Machine Learning**: Scripts for training and evaluating models that classify PVY-infected plants.
- **Model Evaluation**: Tools for testing model performance and visualizing results.

## Code Overview

This repository consists of several Python and MATLAB scripts that serve different functions in the PVY detection pipeline:

### Python Scripts

1. **[`load_data.py`](./load_data.py)**  
   Reads the `hyper_data_raw_band_removed` files and generates `.npy` files for training, validation, and testing datasets.

2. **[`create_labels.py`](./create_labels.py)**  
   Generates labels from the `annotations.xml` file and converts them into `.npy` files for each image. The label classes are:
   - 0 – Background
   - 3 – Healthy
   - 4 – Infected
   - 5 – Unknown
   - 6 – Resistant

3. **[`create_dataset.py`](./create_dataset.py)**  
   Combines raw data with first and second derivative data, then separates them by class (background, healthy, infected, and resistant) to create a pixelated dataset.

4. **[`virus_classifier.py`](./virus_classifier.py)**  
   Classifies potato vs non-potato pixels using NDVI and smoothed, clipped, and normalized hyperspectral data. This script generates `.npz` and `.mat` files containing downsampled compressed images labeled as infected, healthy, and resistant.

### MATLAB Scripts

5. **[`generate_csv_data.m`](./matlab_data/generate_csv_data.m)**  
   Generates a CSV file (`compressed_virus_data.csv`) containing data for healthy and infected samples from susceptible plants only.

6. **[`gen_susceptible_train_test_data.m`](./matlab_data/gen_susceptible_train_test_data.m)**  
   Generates a balanced training set of healthy and infected samples and a test set with images 26, 31, 39, and 43. It outputs the `susceptible_train_test_data.mat` file.

7. **[`train_multiple_models.m`](./matlab_data/train_multiple_models.m)**  
   Trains six different machine learning models with optimization and saves the trained models in `trained_models.mat`.

8. **[`test_multiple_models.m`](./matlab_data/test_multiple_models.m)**  
   Runs predictions on the test set, saves results, and updates the model files in `updated_trained_models.mat`.

9. **[`gen_images_with_trained_models.m`](./matlab_data/gen_images_with_trained_models.m)**  
   Predicts the labels for all images using all trained models and generates output images.

10. **[`gen_spectrum.m`](./matlab_data/gen_spectrum.m)**  
   Generates the mean spectrum for both healthy and infected plants.

11. **[`feature_selection.m`](./matlab_data/feature_selection.m)**  
   Performs feature importance testing to assess the relevance of different features in classification.


