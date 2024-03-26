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
# - each group has features of shape (n_channels, x, z) and two, raw images
# of shape (wavelengths, pulses, x, z) and labels of shape (x, z), one is the
# spatial concentration and the other is a binary mask
# - also saves a binary mask to segment the background from the sample
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
        'feature_normalisation_params' : {}, # max, min, std and mean
        'image_normalisation_params' : {}, # max, min, std and mean
        'concentration_normalisation_params' : {} # max, min, std and mean
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
        
    logging.info(f'raw data located in {root_dir}')#', {samples}')
    logging.info(f'{len(samples)} samples found')
    
    [data, sim_cfg] = load_sim(
        os.path.join(root_dir, samples[0]),
        args=['p0_tr', 'ReBphP_PCM_c_tot', 'bg_mask']
    )
    shape = data['p0_tr']
    
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
        
        if cluster_id not in groups:
            with h5py.File(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5'), 'r+') as f:
                group = f.create_group(cluster_id)
                group.create_dataset('features', data=features)
                group.create_dataset('images', data=data['p0_tr'])
                group.create_dataset('c_tot', data=data['ReBphP_PCM_c_tot'])
                group.create_dataset('c_mask', data=data['ReBphP_PCM_c_tot'] > 0.0)
                group.create_dataset('bg_mask', data=data['bg_mask'])
        
        features[np.isnan(features)] = 0.0
        # max, min and mean of each channel (keep channel dimension, axis=0)
        feature_max[i] = np.max(features, axis=(1, 2))
        feature_min[i] = np.min(features, axis=(1, 2))
        feature_mean[i] = np.mean(features, axis=(1, 2))
        
        # example of how to visualise features extracted
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
        
    
    image_max = np.max(image_max)
    image_min = np.min(image_min)
    image_mean = np.mean(image_mean)
    feature_max = np.max(feature_max, axis=0)
    feature_min = np.min(feature_min, axis=0)
    feature_mean = np.mean(feature_mean, axis=0)
    c_max = np.max(c_max)
    c_min = np.min(c_min)
    c_mean = np.mean(c_mean)
    
    dataset_cfg['image_normalisation_params']['max'] = [image_max]
    dataset_cfg['image_normalisation_params']['min'] = [image_min]
    dataset_cfg['image_normalisation_params']['mean'] = [image_mean]
    dataset_cfg['feature_normalisation_params']['max'] = feature_max.tolist()
    dataset_cfg['feature_normalisation_params']['min'] = feature_min.tolist()
    dataset_cfg['feature_normalisation_params']['mean'] = feature_mean.tolist()
    dataset_cfg['concentration_normalisation_params']['max'] = [c_max]
    dataset_cfg['concentration_normalisation_params']['min'] = [c_min]
    dataset_cfg['concentration_normalisation_params']['mean'] = [c_mean]
        
    # to compute the standared deviations 
    # sqrt( (np.sum(x-(np.sum(x)/n))**2)/(n-1) )
    # a two pass approach is used, reason: storing the sum of (x**2) 
    # is likely to cause floating point errors
    
    # ssr = sum of squared residuals
    feature_ssr = np.empty((len(samples), len(dataset_cfg['feature_names'])))
    image_ssr = 0
    c_ssr = 0

    for i, sample in enumerate(samples):
        cluster_id = '.'.join(sample.split('.')[-2:])     
            
        # load the data
        with h5py.File(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5'), 'r') as f:
            features = f[cluster_id]['features'][()]
            images = f[cluster_id]['images'][()]
            c = f[cluster_id]['c_tot'][()]
        
        features[np.isnan(features)] = 0.0
        feature_ssr = np.sum( (features - feature_mean[:,np.newaxis,np.newaxis])**2, axis=(1, 2))
        image_ssr += np.sum( (images - image_mean)**2 )
        c_ssr += np.sum( (c - c_mean)**2 )
    
    features_n = len(samples) * np.prod(features.shape[-2:])
    feature_std = np.sqrt( feature_ssr/(features_n-1) )
    images_n = len(samples) * np.prod(images.shape)
    images_std = np.sqrt( image_ssr/(images_n-1) )
    c_n = len(samples) * np.prod(c.shape)
    c_std = np.sqrt( c_ssr/(c_n-1) )
    
    dataset_cfg['image_normalisation_params']['std'] = [images_std]
    dataset_cfg['feature_normalisation_params']['std'] = feature_std.tolist()
    dataset_cfg['concentration_normalisation_params']['std'] = [c_std]
        
    
    logging.info(f'dataset config with global normalisation/standaredisation parameters: {dataset_cfg}')
        
    with open(os.path.join(dataset_cfg['dataset_name'], 'config.json'), 'w') as f:
        json.dump(dataset_cfg, f)
        