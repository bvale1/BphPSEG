import numpy as np
import matplotlib.pyplot as plt
import json
import os
import h5py


plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"

# add project root (folder above) to path to import custom modules
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_root)

from custom_pytorch_utils.custom_transforms import *
from custom_pytorch_utils.custom_datasets import *

sample_name = 'c143423.p31'
feature_names = [r'$A$', r'$k$', r'$b$', r'$R_{\mathrm{f}}^{2}$', r'DI', r'RI']

datasets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocessing')
datasets = [
    os.path.join(datasets_path, '20240517_BphP_cylinders_no_noise/'),
    os.path.join(datasets_path, '20240502_BphP_cylinders_noise_std2/'),
    os.path.join(datasets_path, '20240517_BphP_cylinders_noise_std6/')
]
noise_levels = [r'Dataset 1 (no noise)', r'Dataset 2 (SNR$_{\mathrm{dB}}=18.4)$', r'Dataset 3 (SNR$_{\mathrm{dB}}=8.8)$']

# plot feature images from sample c143423.p31 for each noise level
with h5py.File(os.path.join(datasets[0], 'dataset.h5'), 'r') as f:
    #features1 = np.flip(f['c143423.p31']['features'][()], axis=1)
    features1 = f['c143423.p31']['features'][()]
    images1 = f['c143423.p31']['images'][()]
with h5py.File(os.path.join(datasets[1], 'dataset.h5'), 'r') as f:
    #features2 = np.flip(f['c143423.p31']['features'][()], axis=1)
    features2 = f['c143423.p31']['features'][()]
    images2 = f['c143423.p31']['images'][()]
with h5py.File(os.path.join(datasets[2], 'dataset.h5'), 'r') as f:
    #features3 = np.flip(f['c143423.p31']['features'][()], axis=1)
    features3 = f['c143423.p31']['features'][()]
    images3 = f['c143423.p31']['images'][()]
with open(os.path.join(os.path.dirname(datasets[0]), 'config.json'), 'r') as f:
    config = json.load(f)
    dx = config['dx'] * 1e3   # convert from m to mm
shape = [256, 256]
extent = [-dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2]
fig = plt.figure(figsize=(20, 20))
subfigs = fig.subfigures(nrows=3, ncols=1)
all_features = [features1, features2, features3]
all_images = [images1, images2, images3]

for g, (subfig, label, dataset) in enumerate(zip(subfigs, noise_levels, all_features)):
    subfig.suptitle(label, fontsize=40, y=1.07)
    axs = subfig.subplots(2, 6, gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    subfig.subplots_adjust(wspace=0.0, hspace=0.0)

    for wl in range(2):                  # rows inside group
        for feature, name in enumerate(feature_names):   # 6 columns
            ax = axs[wl, feature]
            #ax.set_axis_off()
            img = dataset[feature + wl * len(feature_names), :, :]
            img[np.isnan(img)] = 0.0
            ax.imshow(
                img,
                cmap="viridis",
                extent=extent,
                origin="lower",
            )
            ax.tick_params(axis='both', labelsize=25)
            if wl == 0:
                ax.set_title(name, fontsize=40)
            if feature == 0 and wl == 0:
                ax.set_ylabel("z (mm)", fontsize=30)
                ax.text(
                    -0.75, 0.4, r"$\lambda_{1}$",
                    transform=ax.transAxes,
                    rotation=0,
                    ha='left', va='bottom',
                    fontsize=35,
                    clip_on=False,
                )
            elif feature == 0 and wl == 1:
                ax.set_ylabel("z (mm)", fontsize=30)
                ax.text(
                    -0.75, 0.4, r"$\lambda_{2}$",
                    transform=ax.transAxes,
                    rotation=0,
                    ha='left', va='bottom',
                    fontsize=35,
                    clip_on=False,
                )
            else:
                ax.set_yticks([])
            if g == 2 and wl == 1:
                ax.set_xlabel(r"x (mm)", fontsize=30)
                ax.tick_params(axis='x', bottom=True, labelbottom=True)
            else:
                ax.tick_params(axis='x', bottom=True, labelbottom=False)

fig.savefig("fig_10_features_mosaic.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.close(fig)

x1, z1 = 195, 105
x2, z2 = 90, 50
x3, z3 = 180, 165 #175, #175

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(18, 22), gridspec_kw={"wspace": 0.3, "hspace": 0.05})
axs[0, 0].set_title('Dataset 1\n' + '(no noise)' + '\n' + r'$\lambda_{1}, i=1$ (Pa J$^{-1}$)', fontsize=30, y=1.05)
axs[0, 0].plot((x1-128)*dx, (z1-128)*dx, 'rx', markersize=20, markeredgewidth=4)
axs[0, 0].plot((x2-128)*dx, (z2-128)*dx, 'gx', markersize=20, markeredgewidth=4)
axs[0, 0].plot((x3-128)*dx, (z3-128)*dx, 'bx', markersize=20, markeredgewidth=4)
axs[0, 1].set_title('Dataset 2\n' + r'(SNR$_{\mathrm{dB}}=18.4)$' + '\n' + r'$\lambda_{1}, i=1$ (Pa J$^{-1}$)', fontsize=30, y=1.05)
axs[0, 2].set_title('Dataset 3\n' + r'(SNR$_{\mathrm{dB}}=8.8)$' + '\n' + r'$\lambda_{1}, i=1$ (Pa J$^{-1}$)', fontsize=30, y=1.05)
for y in range(4):
    for x in range(3):
        
        if y == 0:
            i = 0
            j = 0
        elif y == 1:
            i = 0
            j = 15
            ax = axs[y, x].set_title(r'$\lambda_{1}, i=16$ (Pa J$^{-1}$)', fontsize=30, y=1.05)
        elif y == 2:
            i = 1
            j = 0
            ax = axs[y, x].set_title(r'$\lambda_{2}, i=1$ (Pa J$^{-1}$)', fontsize=30, y=1.05) # 1 or 16?
        elif y == 3:
            i = 1
            j = 15
            ax = axs[y, x].set_title(r'$\lambda_{2}, i=16$ (Pa J$^{-1}$)', fontsize=30, y=1.05) # 16 or 32?
        
        img = axs[y, x].imshow(all_images[x][0, i, j, :, :], extent=extent, origin='lower', cmap='binary_r')
        cbar = fig.colorbar(img, ax=axs[y, x], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)
        
        if x == 0:
            axs[y, x].set_ylabel(r"z (mm)", fontsize=25)
            axs[y, x].tick_params(axis='y', left=True, labelleft=True, labelsize=20)
        else:
            axs[y, x].set_yticks([])
        
        if y == 3:
            axs[y, x].set_xlabel(r"x (mm)", fontsize=25)
            axs[y, x].tick_params(axis='x', bottom=True, labelbottom=True, labelsize=20)
        else:
            axs[y, x].set_xticks([])


fig.savefig("fig_8_noise_level_images.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.close(fig)