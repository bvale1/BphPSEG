import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import json
import os
import h5py
import sys
from transformers import SegformerForSemanticSegmentation

project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_root)

from pytorch_models.BphPQUANT import BphPQUANT
from pytorch_models.BphPSEG import BphPSEG
from pytorch_models.MLP import inherit_mlp_class_from_parent
from pytorch_models.BphP_segformer import inherit_segformer_class_from_parent
import custom_pytorch_utils.augment_models_func as amf
from custom_pytorch_utils.custom_transforms import *
from custom_pytorch_utils.custom_datasets import *

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"
    
#sample_name = 'c143423.p31' # median
#sample_name = 'c145187.p4' # best, but has no proteins
#sample_name = 'c139810.p13' # second best, but has no proteins
sample_name = 'c156467.p12' # best
#sample_name = 'c139810.p11' # worst
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
# get index of sample in list of all samples
sample_idx = all_samples.index(sample_name)

with h5py.File(os.path.join(datasets[0], 'dataset.h5'), 'r') as f:
    gt_reg = f[sample_name]['c_tot'][()] * 1e-3 # [mol m^-3] -> [M]
    gt_reg = gt_reg * 1e7 # [M] -> [10^-7 M]
    gt_bin = f[sample_name]['c_mask'][()].astype(bool)

R2 = []
DI = []
pred_mlp_regression = []
pred_segformer_regression = []
err_mlp_regression = []
err_segformer_regression = []
pred_mlp_binary = []
pred_segformer_binary = []
err_mlp_binary = []
err_segformer_binary = []
for i, noise_level in enumerate(['noise_std0', 'noise_std2', 'noise_std6']):
    with open(os.path.join(os.path.dirname(datasets[i]), 'config.json'), 'r') as f:
        config = json.load(f)
    # regression
    BphP_segformer = inherit_segformer_class_from_parent(BphPQUANT)
    MLP = inherit_mlp_class_from_parent(BphPQUANT)
    (_, _, _, _, _, _, test_dataset, _, normalise_y, normalise_x) = create_dataloaders(
        datasets[i], "features", "regression", 'MinMax', 16, config
    )
    (X, Y, bg_mask, inclusion_mask, _) = test_dataset[sample_idx]
    X_inv = normalise_x.inverse(X)
    R2.append(X[9]) # R^2 at lambda 2, 770 nm
    DI.append(X[10]) # DI at lambda 2, 770 nm
    # load segformer model for this noise level
    model = BphP_segformer(
        SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640'),
        in_channels=12, out_channels=1,
        y_transform=normalise_y,
        wandb_log=None, git_hash=None, seed=1,
    )
    model.load_state_dict(torch.load(segformers_regression[i], map_location='cpu', weights_only=True))
    model.eval()
    Y_pred = model.forward(X.unsqueeze(0)).squeeze() * 1e-3 * 1e7 # [mol m^-3] -> [10^-7 M]
    Y_pred = normalise_y.inverse(Y_pred).squeeze().detach().numpy()
    pred_segformer_regression.append(Y_pred)
    err_segformer_regression.append(Y_pred - gt_reg)
    
    model = MLP(
        in_channels=12, out_channels=1,
        y_transform=normalise_y,
        wandb_log=None, git_hash=None, seed=1
    )
    mlp_state = torch.load(mlps_regression[i], map_location='cpu', weights_only=True)
    if isinstance(mlp_state, dict) and 'state_dict' in mlp_state:
        mlp_state = mlp_state['state_dict']
    has_batchnorm_weights = any('layer1.1.' in key or 'layer2.1.' in key for key in mlp_state)
    if not has_batchnorm_weights:
        amf.remove_batchnorm(model)
    model.load_state_dict(mlp_state)
    model.eval()
    Y_pred = model.forward(X.unsqueeze(0)).squeeze() * 1e-3 * 1e7 # [mol m^-3] -> [10^-7 M]
    Y_pred = normalise_y.inverse(Y_pred).squeeze().detach().numpy()
    pred_mlp_regression.append(Y_pred)
    err_mlp_regression.append(Y_pred - gt_reg)

    # binary segmantic segmentation
    BphP_segformer = inherit_segformer_class_from_parent(BphPSEG)
    MLP = inherit_mlp_class_from_parent(BphPSEG)
    (_, _, _, _, _, _, test_dataset, _, _, _) = create_dataloaders(
        datasets[i], "features", "binary", 'MinMax', 16, config
    )
    (X, Y, bg_mask, inclusion_mask, _) = test_dataset[sample_idx]
    # load segformer model for this noise level
    model = BphP_segformer(
        SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640'),
        in_channels=12, out_channels=2,
        wandb_log=None, git_hash=None, seed=1,
    )
    model.load_state_dict(torch.load(segformers_binary[i], map_location='cpu', weights_only=True))
    model.eval()
    Y_pred = model.forward(X.unsqueeze(0)).squeeze().detach().numpy()
    Y_pred = np.argmax(Y_pred, axis=0)
    Y, Y_pred = np.argmax(Y.numpy(), axis=0).astype(bool), Y_pred.astype(bool)
    # plot a visualisation of the confusion matrix
    confusion_array = np.stack([
        np.logical_not(Y_pred) * np.logical_not(Y), # TN
        Y_pred * np.logical_not(Y), # FP
        np.logical_not(Y_pred) * Y, # FN
        Y_pred * Y], # TP
        axis=0
    )
    confusion_array = np.sum(
        confusion_array * np.arange(1,5).reshape(4, 1, 1), axis=0
    )
    pred_segformer_binary.append(Y_pred)
    err_segformer_binary.append(confusion_array)
    
    model = MLP(
        in_channels=12, out_channels=2,
        wandb_log=None, git_hash=None, seed=1
    )
    mlp_state = torch.load(mlps_binary[i], map_location='cpu', weights_only=True)
    if isinstance(mlp_state, dict) and 'state_dict' in mlp_state:
        mlp_state = mlp_state['state_dict']
    has_batchnorm_weights = any('layer1.1.' in key or 'layer2.1.' in key for key in mlp_state)
    if not has_batchnorm_weights:
        amf.remove_batchnorm(model)
    model.load_state_dict(mlp_state)
    model.eval()
    Y_pred = model.forward(X.unsqueeze(0)).squeeze().detach().numpy()
    Y_pred = np.argmax(Y_pred, axis=0)
    Y_pred = Y_pred.astype(bool)
    confusion_array = np.stack([
        np.logical_not(Y_pred) * np.logical_not(Y), # TN
        Y_pred * np.logical_not(Y), # FP
        np.logical_not(Y_pred) * Y, # FN
        Y_pred * Y], # TP
        axis=0
    )
    confusion_array = np.sum(
        confusion_array * np.arange(1,5).reshape(4, 1, 1), axis=0
    )
    pred_mlp_binary.append(Y_pred)
    err_mlp_binary.append(confusion_array)
    
# parameters for plotting binary confusion matrix
labels = [  'TN'   ,  'FP'   ,  'FN'  ,   'TP'    ]
confusion_colors = [ 'white' , 'black' ,'salmon','limegreen']
confusion_cmap = mpl_colors.ListedColormap(confusion_colors)
# other parameters for plotting
IMAGE_WIDTH_IN = 2.64
IMAGE_HEIGHT_IN = 3.08
COLORBAR_HEIGHT_IN = 3.08
DISPLAY_IMAGE_HEIGHT_IN = min(IMAGE_WIDTH_IN, IMAGE_HEIGHT_IN)

fig = plt.figure(figsize=(24, 28))
FIG_WIDTH_IN, FIG_HEIGHT_IN = fig.get_size_inches()

ROW_GAP_IN = 0.42
ROW_GAP_01_IN = ROW_GAP_IN + 0.7
ROW_GAP_OTHER_IN = ROW_GAP_IN - 0.5 + 0.2
TOP_MARGIN_IN = 1.40
CBAR_WIDTH_IN = 0.144
CBAR_PAD_IN = 0.096
TITLE_OFFSET_IN = 0.224
HEADER_SHIFT_DOWN_IN = 0.3
ROW1_TITLE_UP_IN = 0.6
ROW1_PANEL_TITLE_DOWN_IN = 0.6


def add_axes_inch(left_in, bottom_in, width_in, height_in):
    return fig.add_axes([
        left_in / FIG_WIDTH_IN,
        bottom_in / FIG_HEIGHT_IN,
        width_in / FIG_WIDTH_IN,
        height_in / FIG_HEIGHT_IN,
    ])

noise_levels = [
    'dataset 1\n' + r'(no noise)',
    'dataset 2\n' + r'($SNR_{\mathrm{dB}}=18.4)$',
    'dataset 3\n' + r'($SNR_{\mathrm{dB}}=8.8)$'
]

row_defs_left = [
    {
        'title': 'Ground truth \n concentration ' + r' $ c_{\mathrm{tot}}$ ($10^{-7}$ M)',
        'data': [gt_reg],
        'row_type': 'top',
    },
    {
        'title': 'Coefficient of determination ' + r'$R_{\mathrm{f}}^{2}$ at $\lambda_{2}$ (770 nm)',
        'data': R2,
        'row_type': 'r2',
    },
    {'title': r'MLP predictions ($10^{-7}$ M)', 'data': pred_mlp_regression, 'row_type': 'pred'},
    {'title': r'MLP errors ($10^{-7}$ M)', 'data': err_mlp_regression, 'row_type': 'err'},
    {'title': r'SegFormer-B5 predictions ($10^{-7}$ M)', 'data': pred_segformer_regression, 'row_type': 'pred'},
    {'title': r'SegFormer-B5 errors ($10^{-7}$ M)', 'data': err_segformer_regression, 'row_type': 'err'},
]

row_defs_right = [
    {
        'title': 'Ground truth \n segmentation mask',
        'data': [gt_bin],
        'row_type': 'top',
    },
    {
        'title': 'Difference image ' + r'DI at $\lambda_{2}$ (770 nm) (Pa J$^{-1}$)',
        'data': DI,
        'row_type': 'r2',
    },
    {'title': r'MLP predictions (binary)', 'data': pred_mlp_binary, 'row_type': 'pred_bin'},
    {'title': r'MLP errors (confusion)', 'data': err_mlp_binary, 'row_type': 'confusion'},
    {'title': r'SegFormer-B5 predictions (binary)', 'data': pred_segformer_binary, 'row_type': 'pred_bin'},
    {'title': r'SegFormer-B5 errors (confusion)', 'data': err_segformer_binary, 'row_type': 'confusion'},
]

y_top_edge_in = FIG_HEIGHT_IN - TOP_MARGIN_IN
row_three_width_in = 3 * IMAGE_WIDTH_IN
panel_width_in = row_three_width_in + CBAR_PAD_IN + CBAR_WIDTH_IN
panel_gap_in = 0.8
total_panels_width_in = 2 * panel_width_in + panel_gap_in
left_panel_left_in = (FIG_WIDTH_IN - total_panels_width_in) / 2.0
right_panel_left_in = left_panel_left_in + panel_width_in + panel_gap_in

row_bottoms_in = [y_top_edge_in - IMAGE_HEIGHT_IN]
for row_idx in range(1, len(row_defs_left)):
    gap_in = ROW_GAP_01_IN if row_idx == 1 else ROW_GAP_OTHER_IN
    row_bottoms_in.append(row_bottoms_in[-1] - IMAGE_HEIGHT_IN - gap_in)

def plot_panel(panel_left_in, row_defs, show_y_axis=True, pred_vmin=None, pred_vmax=None):
    row_three_left_in = panel_left_in
    panel_inner_width_in = panel_width_in - (CBAR_PAD_IN + CBAR_WIDTH_IN)
    for row_idx, row in enumerate(row_defs):
        y_bottom_in = row_bottoms_in[row_idx]
        y_title_in = y_bottom_in + IMAGE_HEIGHT_IN + TITLE_OFFSET_IN - HEADER_SHIFT_DOWN_IN
        if row_idx == 1:
            y_title_in += ROW1_TITLE_UP_IN
        title_x = (panel_left_in + panel_width_in / 2.0) / FIG_WIDTH_IN
        if row_idx == 1:
            for i in range(3):
                header_x = (row_three_left_in + i * IMAGE_WIDTH_IN + IMAGE_WIDTH_IN / 2.0) / FIG_WIDTH_IN
                fig.text(header_x, y_title_in / FIG_HEIGHT_IN, noise_levels[i], ha='center', va='bottom', fontsize=20)
            row_title_y_in = y_title_in - ROW1_PANEL_TITLE_DOWN_IN
            fig.text(title_x, row_title_y_in / FIG_HEIGHT_IN, row['title'], ha='center', va='bottom', fontsize=20)
        elif row['row_type'] == 'top':
            top_title_x = (panel_left_in + (panel_inner_width_in - IMAGE_WIDTH_IN) / 2.0 + IMAGE_WIDTH_IN / 2.0) / FIG_WIDTH_IN
            fig.text(top_title_x, y_title_in / FIG_HEIGHT_IN, row['title'], ha='center', va='bottom', fontsize=20)
        else:
            fig.text(title_x, y_title_in / FIG_HEIGHT_IN, row['title'], ha='center', va='bottom', fontsize=20)

        if row['row_type'] == 'top':
            x_img_in = panel_left_in + (panel_inner_width_in - IMAGE_WIDTH_IN) / 2.0
            ax = add_axes_inch(x_img_in, y_bottom_in, IMAGE_WIDTH_IN, IMAGE_HEIGHT_IN)
            if show_y_axis:
                top_vmin = pred_vmin
                top_vmax = pred_vmax
            else:
                top_vmin, top_vmax = 0, 1
            img = ax.imshow(row['data'][0], extent=extent, origin='lower', cmap='binary', vmin=top_vmin, vmax=top_vmax)
            ax.set_box_aspect(1)
            if show_y_axis:
                ax.set_ylabel('z (mm)', fontsize=18)
            else:
                ax.set_ylabel('')
                ax.set_yticks([])
            ax.set_xticks([])
            ax.tick_params(axis='both', labelsize=18)

            if show_y_axis:
                image_display_bottom_in = y_bottom_in + (IMAGE_HEIGHT_IN - DISPLAY_IMAGE_HEIGHT_IN) / 2.0
                cbar_height_in = min(COLORBAR_HEIGHT_IN, DISPLAY_IMAGE_HEIGHT_IN)
                cbar_bottom_in = image_display_bottom_in + (DISPLAY_IMAGE_HEIGHT_IN - cbar_height_in) / 2.0
                cax = add_axes_inch(
                    x_img_in + IMAGE_WIDTH_IN + CBAR_PAD_IN,
                    cbar_bottom_in,
                    CBAR_WIDTH_IN,
                    cbar_height_in,
                )
                cbar = fig.colorbar(img, cax=cax)
                cbar.ax.tick_params(labelsize=18)
            continue

        row_data = row['data']
        if row['row_type'] == 'pred':
            vmin = pred_vmin if pred_vmin is not None else min(np.min(data) for data in row_data)
            vmax = pred_vmax if pred_vmax is not None else max(np.max(data) for data in row_data)
        elif row['row_type'] == 'err':
            vmax = max(np.max(np.abs(data)) for data in row_data)
            vmin = -vmax
        elif row['row_type'] == 'confusion':
            vmin, vmax = 1, 4
        elif row['row_type'] == 'pred_bin':
            vmin, vmax = 0, 1
        else:
            vmin = None
            vmax = None

        for i in range(3):
            x_img_in = row_three_left_in + i * IMAGE_WIDTH_IN
            ax = add_axes_inch(x_img_in, y_bottom_in, IMAGE_WIDTH_IN, IMAGE_HEIGHT_IN)

            if row['row_type'] == 'err':
                img = ax.imshow(row_data[i], extent=extent, origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
            elif row['row_type'] == 'confusion':
                img = ax.imshow(row_data[i], extent=extent, origin='lower', cmap=confusion_cmap, vmin=vmin, vmax=vmax)
            elif row['row_type'] == 'pred_bin':
                img = ax.imshow(row_data[i], extent=extent, origin='lower', cmap='binary', vmin=vmin, vmax=vmax)
            elif row['row_type'] == 'r2':
                img = ax.imshow(row_data[i], extent=extent, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            else:
                img = ax.imshow(row_data[i], extent=extent, origin='lower', cmap='binary', vmin=vmin, vmax=vmax)

            ax.set_box_aspect(1)
            ax.tick_params(axis='both', labelsize=18)

            if row['row_type'] == 'r2' and row_idx != 1:
                ax.set_title(row['title'], fontsize=20)
            if i == 0 and show_y_axis:
                ax.set_ylabel('z (mm)', fontsize=18)
            else:
                ax.set_yticks([])
            if row_idx == 5:
                ax.set_xlabel('x (mm)', fontsize=18)
            else:
                ax.set_xticks([])

        if row['row_type'] != 'confusion' and not (row['row_type'] == 'pred_bin' and not show_y_axis):
            image_display_bottom_in = y_bottom_in + (IMAGE_HEIGHT_IN - DISPLAY_IMAGE_HEIGHT_IN) / 2.0
            cbar_height_in = min(COLORBAR_HEIGHT_IN, DISPLAY_IMAGE_HEIGHT_IN)
            cbar_bottom_in = image_display_bottom_in + (DISPLAY_IMAGE_HEIGHT_IN - cbar_height_in) / 2.0
            cax = add_axes_inch(
                row_three_left_in + row_three_width_in + CBAR_PAD_IN,
                cbar_bottom_in,
                CBAR_WIDTH_IN,
                cbar_height_in,
            )
            cbar = fig.colorbar(img, cax=cax)
            cbar.ax.tick_params(labelsize=18)
        elif row['row_type'] == 'confusion':
            legend_handles = [
                plt.Line2D([0], [0], marker='s', linestyle='none',
                           markerfacecolor=color, markeredgecolor='black', markersize=10)
                for color in confusion_colors
            ]
            ax.legend(
                legend_handles,
                labels,
                loc='center left',
                bbox_to_anchor=(1.05, 0.5),
                frameon=False,
                fontsize=16,
                borderaxespad=0.0,
            )


left_pred_vmax = max(
    np.max(pred_mlp_regression),
    np.max(pred_segformer_regression),
)
plot_panel(left_panel_left_in, row_defs_left, show_y_axis=True, pred_vmin=0, pred_vmax=left_pred_vmax)
plot_panel(right_panel_left_in, row_defs_right, show_y_axis=False)
        
#fig.savefig("fig_14_test_sample_predictions.pdf", format="pdf", dpi=600, bbox_inches="tight")
fig.savefig("fig_16_best_c156467.p12.pdf", format="pdf", dpi=600, bbox_inches="tight")
#fig.savefig("fig_17_worst_c139810.p11.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.close(fig)