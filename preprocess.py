import numpy as np
import h5py, logging, torch, os, json
from dataloader import load_sim, heatmap
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
    root_dir = 'E:/cluster_MSOT_simulations/BphP_phantom/'
    
    samples = os.listdir(root_dir)
    
    dataset_cfg = {
        'dataset_name' : 'homogeneous_cylinders',
        'BphP_SVM_RF_XGB_git_hash' : None, # TODO: get git hash automatically
        'feature_names' : [
            'A_680nm', 'k_680nm', 'b_680nm', 'R_sqr_680nm', 'diff_680nm', 'range_680nm',
            'A_770nm', 'k_770nm', 'b_770nm', 'R_sqr_770nm', 'diff_770nm', 'range_770nm',
            'radial_dist'
        ]
    }
    
    with open(dataset_cfg['dataset_name'] + '_config.json', 'w') as f:
        json.dump(dataset_cfg, f)
    
    with h5py.File(dataset_cfg['dataset_name'] + '.h5', 'w') as f:
        logging.info(f"creating {dataset_cfg['dataset_name'] + '.h5'}")
        
    # process one sample at a time,
    # the option to run in parallel may be implemented later
    for sample in samples:
        [data, sim_cfg] = load_sim(
            os.path.join(root_dir, sample),
            args=['p0_tr', 'ReBphP_PCM_c_tot', 'bg_mask']
        )
        cluster_id = '.'.join(sample.split('.')[-2:])
        
        # divide by total energy delivered [Pa] -> [Pa J^-1]
        data['p0_tr'] *= 1/np.asarray(sim_cfg['LaserEnergy'])[:,:,:,np.newaxis,np.newaxis]
        
        fe = feature_extractor(data['p0_tr'][0,0], mask=data['bg_mask'])
        fe.NLS_scipy()
        fe.threshold_features()
        fe.filter_features()
        fe.R_squared()
        fe.differetial_image()
        fe.range_image()
        
        features_680nm, keys_680nm = fe.get_features(asTensor=True)
        
        fe = feature_extractor(data['p0_tr'][0,1], mask=data['bg_mask'])
        fe.NLS_scipy()
        fe.threshold_features()
        fe.filter_features()
        fe.R_squared()
        fe.differetial_image()
        fe.range_image()
        fe.radial_distance()
        
        features_770nm, keys_770nm = fe.get_features(asTensor=True)
        
        features = torch.cat([features_680nm, features_770nm], dim=0)
        
        # uncomment to visually inspect features
        heatmap(
            features, 
            labels=dataset_cfg['feature_names'],
            title=cluster_id,
            dx=sim_cfg['dx'],
            sharescale=False,
            cmap='cool'
        )
        heatmap(
            data['p0_tr'][0,0],
            dx=sim_cfg['dx'],
            title=cluster_id + ' 770nm', 
            sharescale=True
        )
        heatmap(
            data['p0_tr'][0,1],
            dx=sim_cfg['dx'],
            title=cluster_id + ' 680nm',
            sharescale=True
        )
        
        with h5py.File(dataset_cfg['dataset_name'] + '.h5', 'r+') as f:
            group = f.create_group(cluster_id)
            group.create_dataset('features', data=features.numpy())
            group.create_dataset('c_tot', data=data['ReBphP_PCM_c_tot'])
            group.create_dataset('c_mask', data=data['ReBphP_PCM_c_tot'] > 0.0)
            group.create_dataset('bg_mask', data=data['bg_mask'])
        