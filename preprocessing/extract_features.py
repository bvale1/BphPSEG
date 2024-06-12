import numpy as np
import h5py, logging, torch, os, json
from dataloader import load_sim, heatmap
from feature_extractor import feature_extractor
from scipy.ndimage import median_filter

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
    root_dir = 'F:/cluster_MSOT_simulations/BphP_phantom_with_noise/' # from windows
    
    
    dataset_cfg = {
        'dataset_name' : '20240517_BphP_cylinders_noise_std6',#'20240502_BphP_cylinders_noise_std2',#'20240517_BphP_cylinders_no_noise',
        'git_hash' : None, # TODO: get git hash automatically
        'recon_key' : 'p0_tr', #'noisy_p0_tr', # reconstructions used to extract features
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
        args=[dataset_cfg['recon_key'], 'ReBphP_PCM_c_tot', 'bg_mask']
    )
    shape = data[dataset_cfg['recon_key']]
    
    dataset_cfg['dx'] = sim_cfg['dx'] # save the pixel size and config json (these need to be consistant throughout the dataset)
    with open(os.path.join(dataset_cfg['dataset_name'], 'config.json'), 'w') as f:
        json.dump(dataset_cfg, f)
    
        
    # this script calculates the overall min, max and mean of each input
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
    
    protein_n = 0 # total number of pixels containing proteins
    
    # process one sample at a time,
    # the option to run in parallel may be implemented later
    for i, sample in enumerate(samples):
        cluster_id = '.'.join(sample.split('.')[-2:])     
            
        # load the data
        logging.info(f'sample {cluster_id}, {i+1}/{len(samples)}')
        [data, sim_cfg] = load_sim(
            os.path.join(root_dir, sample),
            args=[dataset_cfg['recon_key'], 'ReBphP_PCM_c_tot', 'bg_mask']
        )
        
        # sanity check: this generally catches simulations that fail without raising exceptions
        if np.any(~np.isfinite(data[dataset_cfg['recon_key']])):
            logging.info(f'non-finite values found in sample {cluster_id}, skipping, consider removing from dataset')
            continue
        percent_zero = np.sum(data[dataset_cfg['recon_key']] == 0.0) / np.prod(data[dataset_cfg['recon_key']].shape)
        logging.info(f'{percent_zero*100:.2f}% of pixels are zero')
        if percent_zero > 0.5:
            logging.info(f'>50% of pixels are zero, skipping sample {cluster_id}, consider removing from dataset')
            continue
        
        # divide by total energy delivered [Pa] -> [Pa J^-1]
        data[dataset_cfg['recon_key']] *= 1/np.asarray(sim_cfg['LaserEnergy'])[:,:,:,np.newaxis,np.newaxis]
        # apply 3x3 median filter
        shape = data[dataset_cfg['recon_key']].shape
        data[dataset_cfg['recon_key']] = data[dataset_cfg['recon_key']].reshape((np.prod(shape[:-2]),) + tuple(shape[-2:]))
        for j in range(int(np.prod(shape[:-2]))):
            data[dataset_cfg['recon_key']][j] = median_filter(
                data[dataset_cfg['recon_key']][j],
                size=3
            )
        data[dataset_cfg['recon_key']] = data[dataset_cfg['recon_key']].reshape(shape)
        
        image_max[i] = np.max(data[dataset_cfg['recon_key']])
        image_min[i] = np.min(data[dataset_cfg['recon_key']])
        image_mean[i] = np.mean(data[dataset_cfg['recon_key']])
        c_max[i] = np.max(data['ReBphP_PCM_c_tot'])
        c_min[i] = np.min(data['ReBphP_PCM_c_tot'])
        c_mean[i] = np.mean(data['ReBphP_PCM_c_tot'])
        protein_n += np.sum(data['ReBphP_PCM_c_tot'] > 0.0)
        
        # feature extraction with scipy optimize takes a long time, it's best to avoid repeating unecessarily
        if cluster_id in groups:
            logging.info(f'sample {cluster_id} already processed, skipping feaure extraction')
            with h5py.File(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5'), 'r') as f:
                features = f[cluster_id]['features'][()]
                
        else:
            logging.info('computing 680nm features')
            fe = feature_extractor(data[dataset_cfg['recon_key']][0,0], mask=data['bg_mask'])
            fe.NLS_scipy()
            fe.threshold_features()
            fe.filter_features()
            fe.R_squared()
            fe.differetial_image()
            fe.range_image()
            
            features_680nm, keys_680nm = fe.get_features(asTensor=True)
            
            logging.info('computing 770nm features')
            fe = feature_extractor(data[dataset_cfg['recon_key']][0,1], mask=data['bg_mask'])
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
                group.create_dataset('images', data=data[dataset_cfg['recon_key']])
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
            data[dataset_cfg['recon_key']][0,0],
            dx=sim_cfg['dx'],
            title=cluster_id + ' 770nm', 
            sharescale=True
        )
        heatmap(
            data[dataset_cfg['recon_key']][0,1],
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
    
    # To approximate the SNR, the script also keeps track of the RMS of
    # the sensor data associated with the phantoms (true signal), which is
    # roughly t=1.875e-05 s and t=3.625e-05 s or sample index 750 to 1450, 
    # also computes statistics such as the min, max, mean and std of the ratio
    # between the intrinsic absorption coefficient and protein concentration in pixels
    # containing proteins
    SNR_params = {
        'true_signal_RMS': 0,
        'mup_mua_min': np.empty(len(samples)), 'mup_mua_max': np.empty(len(samples)),
        'mup_mua_mean': 0#, 'c_mua_std': np.empty(len(samples))
    }
        
    # to compute the standared deviations 
    # sqrt( (np.sum(x-(np.sum(x)/n))**2)/(n-1) )
    # a two pass approach is used, reason: storing the sum of (x**2) 
    # is likely to cause floating point errors
    
    # ssr = sum of squared residuals
    feature_ssr = np.empty((len(samples), len(dataset_cfg['feature_names'])))
    image_ssr = 0
    c_ssr = 0
    
    # same applies with RMS sqrt( (np.sum(x**2/n)) )
    # signal_ss_n = sum of squared values / n
    signal_ss_n = np.float64(0.0)
    
    # to compute statistics on the ratio between the range and the mean
    # of the photoswitching signals at 770 nm
    signals_range_over_mean = np.array([], dtype=np.float32)

    # Far-red absorbing protein molar absorption coefficient at 770 nm is 7530
    # 680 nm is 153 [M^-1 cm^-1]=[m^2 mol^-1]
    delta_epsilon_a_770nm = 7530 - 153
    epsilon_a_Pfr_770nm = 7530

    for i, sample in enumerate(samples):
        cluster_id = '.'.join(sample.split('.')[-2:])
            
        logging.info(f'sample {cluster_id}, {i+1}/{len(samples)}')
        
        # load the data
        [data, sim_cfg] = load_sim(
            os.path.join(root_dir, sample),
            args=['ReBphP_PCM_c_tot', 'sensor_data', 'background_mua_mus']
        )
        
        with h5py.File(os.path.join(dataset_cfg['dataset_name'], 'dataset.h5'), 'r') as f:
            features = f[cluster_id]['features'][()]
            images = f[cluster_id]['images'][()]
            c = f[cluster_id]['c_tot'][()]
        
        mask = data['ReBphP_PCM_c_tot'] > 0.0
        if np.sum(mask) > 0:
            mup_mua = epsilon_a_Pfr_770nm * data['ReBphP_PCM_c_tot'] / (data['background_mua_mus'][1,0])# + 153 * data['ReBphP_PCM_c_tot'][mask])
            SNR_params['mup_mua_max'][i] = np.max(mup_mua[mask])
            SNR_params['mup_mua_min'][i] = np.min(mup_mua[mask])
            SNR_params['mup_mua_mean'] += np.sum(mup_mua[mask]) / protein_n
            
            signals_range_over_mean = np.concatenate(
                (
                    signals_range_over_mean,
                    (np.abs((images[0,1,0,mask] - images[0,1,-1,mask])) / (1e-8 + np.mean(images[0,1,:,mask], axis=1))).flatten()
                )
            )
        
        features[np.isnan(features)] = 0.0
        feature_ssr = np.sum( (features - feature_mean[:,np.newaxis,np.newaxis])**2, axis=(1, 2))
        image_ssr += np.sum( (images - image_mean)**2 )
        c_ssr += np.sum( (c - c_mean)**2 )
        signal_ss_n += np.sum(data['sensor_data'][:,:,:,:,750:1450]**2, dtype=np.float64) / (np.float64(1450-750) * 256 * np.prod(images.shape[:-2], dtype=np.float64))
    
    features_n = len(samples) * np.prod(features.shape[-2:])
    feature_std = np.sqrt( feature_ssr/(features_n-1) )
    images_n = len(samples) * np.prod(images.shape)
    images_std = np.sqrt( image_ssr/(images_n-1) )
    c_n = len(samples) * np.prod(c.shape)
    c_std = np.sqrt( c_ssr/(c_n-1) )
    
    SNR_params['true_signal_RMS'] = np.sqrt(signal_ss_n / len(samples))
    SNR_params['mup_mua_max'] = np.max(SNR_params['mup_mua_max'])
    SNR_params['mup_mua_min'] = np.min(SNR_params['mup_mua_min'])
    logging.info(f'770nm signal to noise ratio parameters: {SNR_params}')
    
    dataset_cfg['image_normalisation_params']['std'] = [images_std]
    dataset_cfg['feature_normalisation_params']['std'] = feature_std.tolist()
    dataset_cfg['concentration_normalisation_params']['std'] = [c_std]
        
    logging.info(f'dataset config with global normalisation/standardisation parameters: {dataset_cfg}')
    
    logging.info(f'mean range(770 nm)={np.mean(signals_range_over_mean, dtype=np.float64)}')
    logging.info(f'std range(770 nm)={np.std(signals_range_over_mean, dtype=np.float64)}')
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8,4), tight_layout=True)
    ax.hist(signals_range_over_mean, bins=2000, density=True)
    ax.set_xlim(0, 0.5)    
    ax.grid(True)
    ax.set_axisbelow(True)
        
    with open(os.path.join(dataset_cfg['dataset_name'], 'config.json'), 'w') as f:
        json.dump(dataset_cfg, f)
        