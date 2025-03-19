
import torch
from custom_pytorch_utils.custom_datasets import BphP_MSOT_Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(
    '/'.join(os.path.dirname(__file__).split('/')[:-1])
))
from preprocessing.dataloader import heatmap, load_sim


def get_datasets(path):
    dataset = BphP_MSOT_Dataset(
        path, 
        gt_type='regression', 
        input_type='images', 
        x_transform=None,
        y_transform=None
    )

    return dataset, (random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # train/val/test sets are always the same
    ))

# Use to plot and visualize the results of the simulation (before preprocessing)
#path = r'\\\\wsl$\\Ubuntu-22.04\\home\\wv00017\\python_BphP_MSOT_sim\\20231123_BphP_phantom_test'
path = '/home/wv00017/python_BphP_MSOT_sim/20231123_BphP_phantom_test'

#[data, sim_cfg] = load_sim(path, args='all')

#fig, _, _ = heatmap(
#    data['p0_tr'][0,:,0],
#    title='initial pressure reconstructions',
#    dx=sim_cfg['dx'],
#    labels=['pulse 1 (680nm)', 'pulse 1 (770nm)'],
#    cbar_label=r'Pa J$^{-1}$'
#)
#plt.savefig('initial_pressure_reconstructions.png')
#fig, _, _ = heatmap(
#    data['background_mua_mus'][:,0],
#    title='absorption coefficient',
#    dx=sim_cfg['dx'],
#    labels=[r'$\mu_a$(680nm)', r'$\mu_a$(770nm)'],
#    cbar_label=r'm$^{-1}$'
#)
#plt.savefig('absorption_coefficient.png')
#fig, _, _ = heatmap(
#    data['background_mua_mus'][:,1],
#    title='scattering coefficient',
#    dx=sim_cfg['dx'],
#    labels=[r'$\mu_s$(680nm)', r'$\mu_s$(770nm)'],
#    cbar_label=r'm$^{-1}$'
#)
#plt.savefig('scattering_coefficient.png')
#fig, _, _ = heatmap(
#    data['ReBphP_PCM_c_tot'],
#    title='total protein concentration',
#    dx=sim_cfg['dx'],
#    cbar_label=r'$10^{3}$ M'
#)
#plt.savefig('total_protein_concentration.png')

# plot preprocessed data
dataset, (_, _, test_dataset) = get_datasets(
    'preprocessing/20240517_BphP_cylinders_no_noise/'
)
dataset1 = test_dataset[6][0]

_, (_, _, test_dataset) = get_datasets(
    'preprocessing/20240502_BphP_cylinders_noise_std2/'
)
dataset2 = test_dataset[6][0]

_, (_, _, test_dataset) = get_datasets(
    'preprocessing/20240517_BphP_cylinders_noise_std6/'
)
dataset3 = test_dataset[6][0]

images = np.array([dataset1[0], dataset2[0], dataset3[0],
                   dataset1[15], dataset2[15], dataset3[15],
                   dataset1[16], dataset2[16], dataset3[16],
                   dataset1[31], dataset2[31], dataset3[31]])  

titles = [r'$\lambda_{1}$, n=1', r'$\lambda_{1}$, n=1', r'$\lambda_{1}$, n=1',
          r'$\lambda_{1}$, n=16', r'$\lambda_{1}$, n=16', r'$\lambda_{1}$, n=16',
          r'$\lambda_{2}$, n=1', r'$\lambda_{2}$, n=1', r'$\lambda_{2}$, n=1',
          r'$\lambda_{2}$, n=16', r'$\lambda_{2}$, n=16', r'$\lambda_{2}$, n=16']

images = np.array([dataset3[0],
                   dataset3[15],
                   dataset3[16],
                   dataset3[31]]) 

titles = [r'$\lambda_{1}$, n=1',
          r'$\lambda_{1}$, n=16',
          r'$\lambda_{2}$, n=1',
          r'$\lambda_{2}$, n=16']

fig, _, _ = heatmap(
    images,
    dx=dataset.config['dx'],
    rowmax=1,
    labels=titles,
    sharescale=True,
    cbar_label=''#'PA signal amplitude (Pa J$^{-1}$)'
)
plt.savefig('reconstructed_images_example.png')
