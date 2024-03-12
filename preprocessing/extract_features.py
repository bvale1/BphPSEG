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
    #root_dir = '/mnt/f/cluster_MSOT_simulations/BphP_phantom/' # from ubuntu
    root_dir = 'F:/cluster_MSOT_simulations/BphP_phantom/' # from windows
    
    
    dataset_cfg = {
        'dataset_name' : '20240305_homogeneous_cylinders',
        'git_hash' : None, # TODO: get git hash automatically
        'feature_names' : [
            'A_680nm', 'k_680nm', 'b_680nm', 'R_sqr_680nm', 'diff_680nm', 'range_680nm',
            'A_770nm', 'k_770nm', 'b_770nm', 'R_sqr_770nm', 'diff_770nm', 'range_770nm'
            #'radial_dist' # <- depricating this feature as it doesn't improve model performance
        ],
        'feature_normalisation_params' : {}, # for maxs, mins and means
        'image_normalisation_params' : {}, # for maxs, mins and means
        'concentration_normalisation_params' : {} # for maxs, mins and means
    }
    
    # create dataset directory if it doesn't exist
    if not os.path.exists(dataset_cfg['dataset_name']):
        logging.info(f"dataset not found, creating directory {dataset_cfg['dataset_name']}")
        os.makedirs(dataset_cfg['dataset_name'])
    
    # create dataset.h5 if it doesn't exist
    groups = []
    if not os.path.exists(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5')):
        with h5py.File(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5'), 'w') as f:
            logging.info(f"creating {os.path.join(dataset_cfg['dataset_name'], 'dataset.h5')}")
    else: # or get list of already processed samples
        with h5py.File(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5'), 'r') as f:
            logging.info(f"opening {os.path.join(dataset_cfg['dataset_name'], 'dataset.h5')}")
            groups = list(f.keys())
        
    
    # check which samples are already processed and saved under dataset_cfg['dataset_name']
    files = os.listdir(root_dir)
    samples = []
    for file in files:
        if '.out' in file or '.log' in file or '.error' in file:
            pass
        else:
            samples.append(file)
        
    logging.info(f'raw data located in {root_dir}, {samples}')
    logging.info(f'{len(samples)} samples found')
    
    [data, sim_cfg] = load_sim(
        os.path.join(root_dir, samples[0]),
        args=['p0_tr', 'ReBphP_PCM_c_tot', 'bg_mask']
    )
    
    dataset_cfg['dx'] = sim_cfg['dx'] # save the pixel size and config json (these need to be consistant throughout the dataset)
    with open(os.path.join(dataset_cfg['dataset_name'], 'config.json'), 'w') as f:
        json.dump(dataset_cfg, f)
    
        
    # this script also calculates the overall min, max and mean of each input
    # and output channel so they can be used to normalise.
    # it is important to apply the exact same transformations to the data 
    # when training for regression/pixel level prediction (i.e. same
    # transformation parameters)
    feature_max = np.empty((len(samples), len(dataset_cfg['feature_names'])))
    feature_min = feature_max.copy()
    feature_mean = feature_max.copy()
    image_max = np.empty(len(samples))
    image_min = image_max.copy()
    image_mean = image_max.copy()
    c_max = np.empty(len(samples))
    c_min = c_max.copy()
    c_mean = c_max.copy()
        
    # process one sample at a time,
    # the option to run in parallel may be implemented later
    for i, sample in enumerate(samples):
        cluster_id = '.'.join(sample.split('.')[-2:])     
            
        # load the data
        logging.info(f'sample {cluster_id}, {i+1}/{len(samples)}')
        [data, sim_cfg] = load_sim(
            os.path.join(root_dir, sample),
            args=['p0_tr', 'ReBphP_PCM_c_tot', 'bg_mask']
        )
        
        # sanity check: this generally catches simulations that fail without raising exceptions
        if np.any(~np.isfinite(data['p0_tr'])):
            logging.info(f'non-finite values found in sample {cluster_id}, skipping, consider removing from dataset')
            continue
        percent_zero = np.sum(data['p0_tr'] == 0.0) / np.prod(data['p0_tr'].shape)
        logging.info(f'{percent_zero*100:.2f}% of pixels are zero')
        if percent_zero > 0.5:
            logging.info(f'>50% of pixels are zero, skipping sample {cluster_id}, consider removing from dataset')
            continue
        
        # divide by total energy delivered [Pa] -> [Pa J^-1]
        data['p0_tr'] *= 1/np.asarray(sim_cfg['LaserEnergy'])[:,:,:,np.newaxis,np.newaxis]
        
        image_max[i] = np.max(data['p0_tr'])
        image_min[i] = np.min(data['p0_tr'])
        image_mean[i] = np.mean(data['p0_tr'])
        c_max[i] = np.max(data['ReBphP_PCM_c_tot'])
        c_min[i] = np.min(data['ReBphP_PCM_c_tot'])
        c_mean[i] = np.mean(data['ReBphP_PCM_c_tot'])
        
        # feature extraction with scipy optimize takes a long time, it's best to avoid repeating unecessarily
        if cluster_id in groups:
            logging.info(f'sample {cluster_id} already processed, skipping feaure extraction')
            with h5py.File(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5'), 'r') as f:
                features = f[cluster_id]['features'][()]
                
        else:
            logging.info('computing 680nm features')
            fe = feature_extractor(data['p0_tr'][0,0], mask=data['bg_mask'])
            fe.NLS_scipy()
            fe.threshold_features()
            fe.filter_features()
            fe.R_squared()
            fe.differetial_image()
            fe.range_image()
            
            features_680nm, keys_680nm = fe.get_features(asTensor=True)
            
            logging.info('computing 770nm features')
            fe = feature_extractor(data['p0_tr'][0,1], mask=data['bg_mask'])
            fe.NLS_scipy()
            fe.threshold_features()
            fe.filter_features()
            fe.R_squared()
            fe.differetial_image()
            fe.range_image()
            # fe.radial_distance(sim_cfg['dx']) # <- depricating this feature as it doesn't improve model performance
            
            features_770nm, keys_770nm = fe.get_features(asTensor=True)
            
            features = torch.cat([features_680nm, features_770nm], dim=0).numpy()
        
        # max, min and mean of each channel (keep channel dimension, axis=0)
        feature_max[i] = np.max(features, axis=(1, 2))
        feature_min[i] = np.min(features, axis=(1, 2))
        feature_mean[i] = np.mean(features, axis=(1, 2))
        
        # example: uncomment to plot features
        '''
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
        '''
        if cluster_id not in groups:
            with h5py.File(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5'), 'r+') as f:
                group = f.create_group(cluster_id)
                group.create_dataset('features', data=features)
                group.create_dataset('images', data=data['p0_tr'])
                group.create_dataset('c_tot', data=data['ReBphP_PCM_c_tot'])
                group.create_dataset('c_mask', data=data['ReBphP_PCM_c_tot'] > 0.0)
                group.create_dataset('bg_mask', data=data['bg_mask'])
        
        
    dataset_cfg['image_normalisation_params']['max'] = np.max(image_max).tolist()
    dataset_cfg['image_normalisation_params']['min'] = np.min(image_min).tolist()
    dataset_cfg['image_normalisation_params']['mean'] = np.mean(image_mean).tolist()
    dataset_cfg['feature_normalisation_params']['max'] = np.max(features, axis=0).tolist()
    dataset_cfg['feature_normalisation_params']['min'] = np.min(features, axis=0).tolist()
    dataset_cfg['feature_normalisation_params']['mean'] = np.mean(features, axis=0).tolist()
    dataset_cfg['concentration_normalisation_params']['max'] = np.max(c_max).tolist()
    dataset_cfg['concentration_normalisation_params']['min'] = np.min(c_min).tolist()
    dataset_cfg['concentration_normalisation_params']['mean'] = np.mean(c_mean).tolist()
    
    with open(os.path.join(dataset_cfg['dataset_name'], 'config.json'), 'w') as f:
        json.dump(dataset_cfg, f)