import numpy as np
import h5py, torch
import matplotlib.pyplot as plt
from preprocessing.dataloader import *
from preprocessing.feature_extractor import *

# validation is focused on wavelength index 1 (770 nm) due to the 
# higher signal-to-noise ratio

# load experimental reconstructions (viewMSOT backprojections)
exp_file = '20230623_clara_upgrade_experiment_backprojections.h5'
with h5py.File(exp_file, 'r') as f:
    exp_recons = f['p0_bp'][()]

# load simulated reconstructions (k-wave iterative time reversal)
sim_folder = '20240325_Clara_exp_bgmus_0.c158729.p0'
#sim_folder = '20240325_Clara_exp_bgmus_500.c158728.p0'
#sim_folder = '20240326_Clara_exp_bg_mua_1_mus_0.c158870.p0'

# index sim_data as [cycle, wavelength, pulse, x, z]
[sim_data, sim_cfg] = load_sim(sim_folder)

# preprocessing:
# average experimental images over 5 cycles 
exp_recons = np.mean(exp_recons, axis=0)
# apply laser energy correction to simulated data
sim_data['p0_tr'] /= np.asarray(sim_cfg['LaserEnergy'])[:,:,:,np.newaxis,np.newaxis]

# extract the region of interest (ROI) from the experimental reconstructions
fe = feature_extractor(exp_recons[1], roi=(85, 142, 12))
exp_roi_mask = torch.reshape(fe.mask, fe.image_size).numpy()
# get mean and std signal intensity of all pixels in the ROI at each pulse
print(f'roi data shape {fe.data.shape}, averging over roi')
exp_roi_stds = torch.std(fe.data, dim=0)
fe.data = torch.mean(fe.data, dim=0, keepdim=True)
exp_roi_means = fe.data[0]
print(f'roi data mean {fe.data}, std {exp_roi_stds}')
# fit the exponential decay
fe.NLS_scipy()
fe.R_squared()
exp_features = fe.features
print('exp_fit: ', exp_features)

# repeat for simulated data
fe = feature_extractor(sim_data['p0_tr'][0,1], roi=(90, 145, 12))
sim_roi_mask = torch.reshape(fe.mask, fe.image_size).numpy()
# get mean and std signal intensity of all pixels in the ROI at each pulse
print(f'roi data shape {fe.data.shape}, averging over roi')
sim_roi_stds = torch.std(fe.data, dim=0)
fe.data = torch.mean(fe.data, dim=0, keepdim=True)
sim_roi_means = fe.data[0]
print(f'roi data mean {fe.data}, std {sim_roi_stds}')
# fit the exponential decay
fe.NLS_scipy()
fe.R_squared()
sim_features = fe.features
print('sim_fit: ', sim_features)

# make plots to verify the ROI's are in the correct location
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(
    exp_recons[1,0]+np.mean(exp_recons[1,0])*exp_roi_mask, 
    cmap='gray', interpolation='none')
ax[0].set(title='experiment with overlayed ROI')
ax[1].imshow(
    sim_data['p0_tr'][0,1,0]+np.mean(sim_data['p0_tr'][0,1,0])*sim_roi_mask, 
    cmap='gray', interpolation='none')
ax[1].set(title='simulation with overlayed ROI')
plt.savefig('roi_overlay.png')

# plot experimental and simulated reconstructions
(fig, _, _) = heatmap(exp_recons[1,0::7], cbar_label='a.u.', labels=['pulse 1', 'pulse 7', 'pulse 14'])
plt.savefig('exp_recons.png')
(fig, _, _) = heatmap(sim_data['p0_tr'][0,1,0::7], cbar_label=r'Pa J$^{-1}$', labels=['pulse 1', 'pulse 7', 'pulse 14'])
plt.savefig('sim_recons.png')
# simualted optical properties of ReBphP at (680nm, 770nm)
# simulated 'eta' photoswitching efficiencies are somewhat arbitrary
ReBphP_PCM = {
    'Pr': {
        'epsilon_a': [7299.7136439689275, 153.01497801013753],
        'eta': [0.004, 0.0]
    }, 
    'Pfr': {
        'epsilon_a': [6307.054943431987, 7530.003941623696],
        'eta': [0.0, 0.002]}
    }
# index background_mua_mus as [1, ..., lambda]->[mu_a, mu_s]->[x]->[z]
# sample absorption coefficient (mu_a) at 770 nm is a linear combination of
# background absorption and ReBphP absorption
mu_a = (sim_data['background_mua_mus'][1,0]
    + (sim_data['ReBphP_PCM_Pr_c'][0,1] * ReBphP_PCM['Pr']['epsilon_a'][1])
    + (sim_data['ReBphP_PCM_Pfr_c'][0,1] * ReBphP_PCM['Pfr']['epsilon_a'][1]))
(fig, _, _) = heatmap(mu_a[0::7], cbar_label=r'm$^{-1}$', labels=['pulse 1', 'pulse 7', 'pulse 14'])
plt.savefig('sim_mu_a.png')


# create plot comparing the normalised decay curves
fig, ax = plt.subplots(1,1, figsize=(5,5))
n_exp = np.arange(len(exp_roi_means))
t_exp = np.linspace(0, len(exp_roi_means)-1, 1000)
exp_fit = exp_features['A'][0] * torch.exp(-exp_features['k'][0] * t_exp) + exp_features['b'][0]
# normalise the experimental decay curve
exp_fit = (exp_fit - exp_features['b'][0]) / exp_features['A'][0]
exp_roi_means = (exp_roi_means - exp_features['b'][0]) / exp_features['A'][0] 
exp_roi_stds = (exp_roi_stds) / exp_features['A'][0]
ax.plot(t_exp+1, exp_fit, label='experiment', color='#1F77B4')
ax.errorbar(n_exp+1, exp_roi_means, yerr=exp_roi_stds, fmt='o', color='#1F77B4')

n_sim = np.arange(len(sim_roi_means))
t_sim = np.linspace(0, len(sim_roi_means)-1, 1000)
sim_fit = sim_features['A'][0] * torch.exp(-sim_features['k'][0] * t_sim) + sim_features['b'][0]
# normalise the simulated decay curve
sim_fit = (sim_fit - sim_features['b'][0]) / sim_features['A'][0]
sim_roi_means = (sim_roi_means - sim_features['b'][0]) / sim_features['A'][0]
sim_roi_stds = (sim_roi_stds) / sim_features['A'][0]
ax.plot(t_sim+1, sim_fit, label='simulation', color='#D62728')
ax.errorbar(n_sim+1, sim_roi_means, yerr=sim_roi_stds, fmt='o', color='#D62728')

ax.set(xlabel='pulse number', ylabel='normalised ROI signal intensity')
ax.legend()
ax.grid(True)
ax.set_axisbelow(True)
fig.tight_layout()
plt.savefig('decay_curves.png')

