import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import json
import os
import h5py
import sys
import torch
from transformers import SegformerForSemanticSegmentation

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"
    
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_root)

from pytorch_models.BphPQUANT import BphPQUANT
from pytorch_models.BphPSEG import BphPSEG
from pytorch_models.MLP import inherit_mlp_class_from_parent
from pytorch_models.BphP_segformer import inherit_segformer_class_from_parent
import custom_pytorch_utils.augment_models_func as amf
from custom_pytorch_utils.custom_transforms import *
from custom_pytorch_utils.custom_datasets import *

segformers_regression = [
    os.path.join(os.path.dirname(__file__), 'segformer_regression_ckpts/noise_std0/segformer_checkpoint.pt'),
    os.path.join(os.path.dirname(__file__), 'segformer_regression_ckpts/noise_std2/segformer_checkpoint.pt'),
    os.path.join(os.path.dirname(__file__), 'segformer_regression_ckpts/noise_std6/segformer_checkpoint.pt'),
]
mlps_regression = [
    os.path.join(os.path.dirname(__file__), 'mlp_regression_ckpts/noise_std0/mlp_checkpoint.pt'),
    os.path.join(os.path.dirname(__file__), 'mlp_regression_ckpts/noise_std2/mlp_checkpoint.pt'),
    os.path.join(os.path.dirname(__file__), 'mlp_regression_ckpts/noise_std6/mlp_checkpoint.pt'),
]
segformers_binary = [
    os.path.join(os.path.dirname(__file__), 'segformer_binary_ckpts/noise_std0/segformer_checkpoint.pt'),
    os.path.join(os.path.dirname(__file__), 'segformer_binary_ckpts/noise_std2/segformer_checkpoint.pt'),
    os.path.join(os.path.dirname(__file__), 'segformer_binary_ckpts/noise_std6/segformer_checkpoint.pt'),
]
mlps_binary = [
    os.path.join(os.path.dirname(__file__), 'mlp_binary_ckpts/noise_std0/mlp_checkpoint.pt'),
    os.path.join(os.path.dirname(__file__), 'mlp_binary_ckpts/noise_std2/mlp_checkpoint.pt'),
    os.path.join(os.path.dirname(__file__), 'mlp_binary_ckpts/noise_std6/mlp_checkpoint.pt'),
]

feature_names = [r'$A$', r'$k$', r'$b$', r'$R_{\mathrm{f}}^{2}$', r'DI', r'RI']

datasets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocessing')
datasets = [
    os.path.join(datasets_path, '20240517_BphP_cylinders_no_noise/'),
    os.path.join(datasets_path, '20240502_BphP_cylinders_noise_std2/'),
    os.path.join(datasets_path, '20240517_BphP_cylinders_noise_std6/')
]
noise_levels = [r'Dataset 1 (no noise)', r'Dataset 2 ($SNR_{\mathrm{dB}}=18.4)$', r'Dataset 3 ($SNR_{\mathrm{dB}}=8.8)$']

with open(os.path.join(os.path.dirname(datasets[0]), 'config.json'), 'r') as f:
    config = json.load(f)
    dx = config['dx'] * 1e3   # convert from m to mm
shape = [256, 256]
extent = [-dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2]


summary_dict = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'per_sample_metrics.json'), 'r'))
all_samples = summary_dict['binary']['noise_std0']['RF']['features']['bg']['sample_names']

concentration_segformer = [[], [], []]
concentration_mlp = [[], [], []]
sensitivity_segformer = [[], [], []]
sensitivity_mlp = [[], [], []]

for i, noise_level in enumerate(['noise_std0', 'noise_std2', 'noise_std6']):
    with open(os.path.join(os.path.dirname(datasets[i]), 'config.json'), 'r') as f:
        config = json.load(f)
    
    # binary segmantic segmentation
    BphP_segformer = inherit_segformer_class_from_parent(BphPSEG)
    MLP = inherit_mlp_class_from_parent(BphPSEG)
    (_, _, _, _, _, _, test_dataset, _, _, _) = create_dataloaders(
        datasets[i], "features", "binary", 'MinMax', 16, config
    )

    # load segformer model for this noise level
    model_segformer = BphP_segformer(
        SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640'),
        in_channels=12, out_channels=2, seed=1,
    )
    model_segformer.load_state_dict(torch.load(segformers_binary[i], map_location='cpu', weights_only=True))
    model_segformer.eval()

    for sample_idx in range(len(test_dataset)):
        (X, Y, _, _, sample_name) = test_dataset[sample_idx]
        # get the concentration for this sample from the sample name
        with h5py.File(os.path.join(datasets[i], 'dataset.h5'), 'r') as f:
            conc = f[sample_name]['c_tot'][()] * 1e-3 * 1e7 # [mols m^-3] -> [M] -> [1e-7 M]
        has_protein = conc > 0

        with torch.no_grad():
            Y_pred = model_segformer.forward(X.unsqueeze(0)).squeeze().detach().cpu().numpy()

        Y_pred = np.argmax(Y_pred, axis=0)
        Y, Y_pred = np.argmax(Y.numpy(), axis=0).astype(bool), Y_pred.astype(bool)
        for unique_conc in np.unique(conc):
            conc_mask = conc == unique_conc
            has_protein_conc = has_protein[conc_mask]
            if not np.any(has_protein_conc):
                continue

            Y_pred_conc = Y_pred[conc_mask][has_protein_conc]
            Y_conc = Y[conc_mask][has_protein_conc]
            concentration_segformer[i].append(unique_conc)
            sensitivity_segformer[i].append(np.mean(Y_pred_conc == Y_conc))
        
    
    model_mlp = MLP(
        in_channels=12, out_channels=2, seed=1
    )
    mlp_state = torch.load(mlps_binary[i], map_location='cpu', weights_only=True)
    if isinstance(mlp_state, dict) and 'state_dict' in mlp_state:
        mlp_state = mlp_state['state_dict']
    has_batchnorm_weights = any('layer1.1.' in key or 'layer2.1.' in key for key in mlp_state)
    if not has_batchnorm_weights:
        amf.remove_batchnorm(model_mlp)
    model_mlp.load_state_dict(mlp_state)
    model_mlp.eval()

    for sample_idx in range(len(test_dataset)):
        (X, Y, _, _, sample_name) = test_dataset[sample_idx]
        # get the concentration for this sample from the sample name
        with h5py.File(os.path.join(datasets[i], 'dataset.h5'), 'r') as f:
            conc = f[sample_name]['c_tot'][()] * 1e-3 * 1e7 # [mols m^-3] -> [M] -> [1e-7 M]
        has_protein = conc > 0

        with torch.no_grad():
            Y_pred = model_mlp.forward(X.unsqueeze(0)).squeeze().detach().cpu().numpy()

        Y_pred = np.argmax(Y_pred, axis=0)
        Y, Y_pred = np.argmax(Y.numpy(), axis=0).astype(bool), Y_pred.astype(bool)
        for unique_conc in np.unique(conc):
            conc_mask = conc == unique_conc
            has_protein_conc = has_protein[conc_mask]
            if not np.any(has_protein_conc):
                continue

            Y_pred_conc = Y_pred[conc_mask][has_protein_conc]
            Y_conc = Y[conc_mask][has_protein_conc]
            concentration_mlp[i].append(unique_conc)
            sensitivity_mlp[i].append(np.mean(Y_pred_conc == Y_conc))

# Plot one panel per noise level.
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
# make all text on the figure larger
plt.rcParams.update({'font.size': 20})

for i, ax in enumerate(axes):
    segformer_x = np.asarray(concentration_segformer[i])
    segformer_y = np.asarray(sensitivity_segformer[i])
    mlp_x = np.asarray(concentration_mlp[i])
    mlp_y = np.asarray(sensitivity_mlp[i])

    ax.scatter(segformer_x, segformer_y, s=20, alpha=0.7, color='tab:blue', label='SegFormer-B5')
    ax.scatter(mlp_x, mlp_y, s=20, alpha=0.7, color='tab:orange', label='MLP')
    ax.set_title(noise_levels[i])
    ax.set_xlabel(r'Concentration ($10^{-7}$ M)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.set_ylim(-0.02, 1.02)
    #ax.grid(True, alpha=0.25)

axes[0].set_ylabel('Sensitivity', fontsize=20)
axes[0].legend(loc='lower right')

fig.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_13_sensitivity_vs_conc.pdf')
fig.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight')
plt.close(fig)

print(f'Saved: {output_path}')