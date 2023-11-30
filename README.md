# pvy_detection
potato virus Y detection using a hyperspectral camera


#### Steps:
- The raw hyperspectral data should be radiance and reflectance calibrated
- `load_data.py` loads the calibrated raw data and creates a training, validation, and test set
  - The output files are saved as NumPy arrays with a dimension of 2000 x 900 x 300
- `create_labels.py` creates the label matrix for each of the images with a dimension of 2000 x 900
