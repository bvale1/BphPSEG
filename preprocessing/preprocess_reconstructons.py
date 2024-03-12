import numpy as np
import h5py, logging, torch, os, json
from dataloader import load_sim, heatmap
from feature_extractor import feature_extractor

# This is a script to preprocess the data from the simulations
# The script is more basic than extract_features.py as it preprocesses the raw 
# images
# Billy Vale - https://github.com/bvale1 - 2023-11

# - preprocessed dataset is saved as a hdf5 file with a group for each sample
# - group names correspond to the cluster id
# - each group has features of shape (n_channels, x, z) and two
# labels of shape (x, z), one is total concentration and the other is binary
# - also saves an (x, z) binary mask to segment the background from the sample
# - a config file is saved as a json file with details of the dataset