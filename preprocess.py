import numpy as np
import h5py, logging, torch, os, json
from dataloader import load_sim
from feature_extractor import feature_extractor

# This is a script to preprocess the data from the simulations
# The pipeline fits an exponential curve to each pixel at each wavelength
# The fit parameters make up features for machine learning
# Billy Vale - https://github.com/bvale1 - 2023-11

# - preprocessed dataset is saved as a hdf5 file with a group for each sample
# - group names correspond to the cluster id
# - each group has features of shape (n_channels, x, z) and two
# labels of shape (x, z), one is total concentration and the other is binary
# - also save a binary mask to segment the background from the sample
# - a config file is saved as a json file with details of the dataset

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    paths = ['']
    # TODO get all the paths from dataset folder
    dataset_cfg = {
        'dataset_name' : 'homogeneous_cylinders',
        'BphP_SVM_RF_XGB_git_hash' : None,
        'feature_names' : [
            'A_680nm', 'k_680nm', 'b_680nm', 'R_sqr_680nm', 'diff_680nm', 'range_680nm',
            'A_770nm', 'k_770nm', 'b_770nm', 'R_sqr_770nm', 'diff_770nm', 'range_770nm'
        ]
    }
    
    with open(dataset_cfg['dataset_name'] + '_config.json', 'w') as f:
        json.dump(dataset_cfg, f)
            
    # process one sample at a time,
    # the option to run in parallel may be implemented later
    for path in paths:
        [data, sim_cfg] = load_sim(path, args=['p0_tr', 'ReBphP_PCM_c_tot'])
        cluster_id = '.'.join(path.split('.')[-2:])
        
        fe = feature_extractor(data['p0_tr'][0,0])
        fe.NLS_scipy()
        fe.filter_features()
        fe.R_squared()
        fe.differetial_image()
        fe.range_image()
        
        features_680nm, keys_680nm = fe.get_features(asTensor=True)
        
        fe = feature_extractor(data['p0_tr'][0,1])
        fe.NLS_scipy()
        fe.filter_features()
        fe.R_squared()
        fe.differetial_image()
        fe.range_image()
        
        features_770nm, keys_770nm = fe.get_features(asTensor=True)
        
        features = torch.cat([features_680nm, features_770nm], dim=1)
        
        with h5py.File(dataset_cfg['dataset_name'] + '.h5', 'a+') as f:
            group = f.create_group(cluster_id)
            group.create_dataset('features', data=features.numpy())
            group.create_dataset('c_tot', data=data['ReBphP_PCM_c_tot'].numpy())
            group.create_dataset('c_mask', data=data['ReBphP_PCM_c_tot'].numpy() > 0.0)
            group.create_dataset('bg_mask', data=data['bg_mask'][0,0].numpy())
        