# Citation
This software was used for the publication titled "Unmanned Aerial Vehicle-Based Hyperspectral Imaging for Potato Virus Y Detection: Machine Learning Insights" with authors S. B. Nesar, P. W. Nugent, N. K. Zidack, and B. M. Whitaker. Please cite the paper when using this code for research purposes.

The data required to run this software is archived on Zenodo at https://doi.org/10.5281/zenodo.15417758 and https://doi.org/10.5281/zenodo.15420134.

# pvy_detection
potato virus Y detection using a hyperspectral camera

#### Steps:
- The raw hyperspectral data should be radiance and reflectance calibrated
- `load_data.py` loads the calibrated raw data and creates a training, validation, and test set
  - The output files are saved as NumPy arrays with a dimension of 2000 x 900 x 300
- `create_labels.py` creates the label matrix for each of the images with a dimension of 2000 x 900
