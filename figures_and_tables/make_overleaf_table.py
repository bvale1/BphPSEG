import json
import itertools
import os
import numpy as np


MODELS = [
    'RF',
    'XGB',
    'mlp',
    'Unet',
    'deeplabv3_resnet101',
    'segformerb5',
]

INPUT_TYPES = [
    'features',
    'images'
]

GT_TYPES = [
    'binary',
    'regression',
]

NOISE_LEVELS = [
    'noise_std0',
    'noise_std2',
    'noise_std6',
]

def make_overleaf_table(
        summary_dict: dict,
        gt_type: str,
        bg_inc: str,
        metric1: str,
        metric2: str,
        caption: str,
        section_title: str,
        label: str,
        unit_scale: float | None = None,
        metric1_header: str | None = None,
        metric2_header: str | None = None,
) -> str:

    if metric1_header is None:
        metric1_header = metric1
    if metric2_header is None:
        metric2_header = metric2
        
    
    lines = []

    head = rf"""
\begin{{table}}[H]
\centering
\footnotesize
\caption{{{caption}}}
\vspace{{0.25cm}}
%\begin{{adjustbox}}{{width=1\textwidth}}
\begin{{tabular}}{{|l|l|l|l|l|l|l|l|}}
\hline
\multicolumn{{8}}{{|l|}}{{{section_title}}} \\
\hline
\multirow{{2}}{{*}}{{Model}} & \multirow{{2}}{{*}}{{Input}} & \multicolumn{{2}}{{|l|}}{{Dataset 1 (no noise)}} & \multicolumn{{2}}{{|l|}}{{Dataset 2 ($\textrm{{SNR}}_{{\textrm{{dB}}}}=18.4$)}} & \multicolumn{{2}}{{|l|}}{{Dataset 3 ($\textrm{{SNR}}_{{\textrm{{dB}}}}=8.8$)}} \\
\cline{{3-8}}
& & {metric1_header} & {metric2_header} & {metric1_header} & {metric2_header} & {metric1_header} & {metric2_header} \\
\hline
"""
    lines.append(head.strip('\n'))

    for model, input_type in itertools.product(MODELS, INPUT_TYPES):
        if model in ['mlp', 'XGB', 'RF'] and input_type == 'images':
            continue
        elif model in ['mlp', 'XGB', 'RF']:
            input_type1 = 'pixel-level'
            input_type2 = 'features'
        elif model in ['Unet', 'deeplabv3_resnet101', 'segformerb5'] and input_type == 'features':
            input_type1 = 'feature'
            input_type2 = 'images'
        else:
            #input_type1 = 'images'
            input_type1 = r'\multirow{2}{*}{images}'
            input_type2 = ''
            
        if model in ['mlp', 'XGB', 'RF']:
            #model_name1 = model.upper()
            model_name1 = r'\multirow{2}{*}{' + model.upper() + r'}'
            model_name2 = ''
        elif model == 'Unet':
            model_name1 = 'U-Net-'
            model_name2 = 'ResNet-101'
        elif model == 'deeplabv3_resnet101':
            model_name1 = 'DeepLabV3-'
            model_name2 = 'ResNet-101'
        elif model == 'segformerb5':
            #model_name1 = 'SegFormer-B5'
            model_name1 = r'\multirow{2}{*}{SegFormer-B5}'
            model_name2 = ''
        else:
            model_name1 = model
            model_name2 = ''

        means1, ci951, means2, ci952 = [], [], [], []
        for noise_level in NOISE_LEVELS:
            try:
                means1.append(summary_dict[gt_type][noise_level][model][input_type][bg_inc][metric1]['mean'])
            except KeyError:
                means1.append(float('nan'))
            try:
                ci951.append(summary_dict[gt_type][noise_level][model][input_type][bg_inc][metric1]['ci95'])
            except KeyError:
                ci951.append(float('nan'))
            try:
                means2.append(summary_dict[gt_type][noise_level][model][input_type][bg_inc][metric2]['mean'])
            except KeyError:
                means2.append(float('nan'))
            try:
                ci952.append(summary_dict[gt_type][noise_level][model][input_type][bg_inc][metric2]['ci95'])
            except KeyError:
                ci952.append(float('nan'))
            if unit_scale is not None:
                means1[-1] = means1[-1] * unit_scale
                ci951[-1] = ci951[-1] * unit_scale
                means2[-1] = means2[-1] * unit_scale
                ci952[-1] = ci952[-1] * unit_scale
        
        try:
            row1 = f"{model_name1} & {input_type1} & {means1[0]:.3f} & {means2[0]:.3f} & {means1[1]:.3f} & {means2[1]:.3f} & {means1[2]:.3f} & {means2[2]:.3f} \\\\" 
            row2 = f"{model_name2} & {input_type2} & $\\pm$ {ci951[0]:.3f} & $\\pm$ {ci952[0]:.3f} & $\\pm$ {ci951[1]:.3f} & $\\pm$ {ci952[1]:.3f} & $\\pm$ {ci951[2]:.3f} & $\\pm$ {ci952[2]:.3f} \\\\" 
        except:
            breakpoint()
        lines.append(row1)
        lines.append(row2)
        lines.append("\\hline")

    foot = rf"""
\end{{tabular}}
%\end{{adjustbox}}
\label{{{label}}}
\end{{table}}
"""
    lines.append(foot.strip('\n'))

    return '\n'.join(lines)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'summary_metrics.json'), 'r') as f:
        summary_dict = json.load(f)

    tables = [
        {
            'gt_type': 'binary',
            'bg_inc': 'inclusion',
            'metric1': 'Sensitivity',
            'metric2': 'Specificity',
            'caption': 'The test sensitivity and specificity on the inhomogeneities, of the models applied in the binary semantic segmentation of the photoswitchable proteins for the 3 datasets. Values are shown as mean $\pm$ 95 \% confidence interval of the mean over 5 runs.',
            'section_title': 'Binary semantic segmentation test sensitivity and specificity (inhomogeneities)',
            'label': 'tab:binary_inhomogeneities_sensitivity_specificity',
        },
        {
            'gt_type': 'binary',
            'bg_inc': 'bg',
            'metric1': 'Sensitivity',
            'metric2': 'Specificity',
            'caption': 'The test sensitivity and specificity on the backgrounds, of the models applied in the binary semantic segmentation of the photoswitchable proteins for the 3 datasets. Values are shown as mean $\\pm$ 95 \\% confidence interval of the mean over 5 runs.',
            'section_title': 'Binary semantic segmentation test sensitivity and specificity (background)',
            'label': 'tab:binary_background_sensitivity_specificity',
        },
        {
            'gt_type': 'regression',
            'bg_inc': 'inclusion',
            'metric1': 'RMSE',
            'metric2': 'MAE',
            'caption': 'The test RMSE and MAE on the inhomogeneities, of the models applied in the regression of the photoswitchable proteins for the 3 datasets. Values are shown as mean $\\pm$ 95 \\% confidence interval of the mean over 5 runs. Two outliers (both U-Net-ResNet101 trained with feature images on dataset 1 with no noise) were excluded from the results because training failed to converge.',
            'section_title': 'Regression test RMSE and MAE (inhomogeneities)',
            'label': 'tab:regression_inhomogeneities_rmse_mae',
            'unit_scale': 1e-3 * 1e9, # scale RMSE and MAE from [M] to [10^-9 M]
            'metric1_header': 'RMSE ($10^{-9}$ M)',
            'metric2_header': 'MAE ($10^{-9}$ M)',
        },
        {
            'gt_type': 'regression',
            'bg_inc': 'bg',
            'metric1': 'RMSE',
            'metric2': 'MAE',
            'caption': 'The test RMSE and MAE on the backgrounds, of the models applied in the regression of the photoswitchable proteins for the 3 datasets. Values are shown as mean $\\pm$ 95 \\% confidence interval of the mean over 5 runs. Two outliers (both U-Net-ResNet101 trained with feature images on dataset 1 with no noise) were excluded from the results because training failed to converge.',
            'section_title': 'Regression test RMSE and MAE (background)',
            'label': 'tab:regression_background_rmse_mae',
            'unit_scale': 1e-3 * 1e9, # scale RMSE and MAE from [M] to [10^-9 M]
            'metric1_header': 'RMSE ($10^{-9}$ M)',
            'metric2_header': 'MAE ($10^{-9}$ M)',
        },
    ]

    for table_cfg in tables:
        print(make_overleaf_table(summary_dict=summary_dict, **table_cfg))
        print()
        
    # print mean RMSE drop of U-Net-ResNet101, DeepLabV3-ResNet101 and SegFormer-B5 from features to images
    for noise_level in NOISE_LEVELS:
        unet_features = summary_dict['regression'][noise_level]['Unet']['features']['inclusion']['RMSE']['mean']
        unet_img = summary_dict['regression'][noise_level]['Unet']['images']['inclusion']['RMSE']['mean']
        deeplabv3_features = summary_dict['regression'][noise_level]['deeplabv3_resnet101']['features']['inclusion']['RMSE']['mean']
        deeplabv3_img = summary_dict['regression'][noise_level]['deeplabv3_resnet101']['images']['inclusion']['RMSE']['mean']
        segformer_features = summary_dict['regression'][noise_level]['segformerb5']['features']['inclusion']['RMSE']['mean']
        segformer_img = summary_dict['regression'][noise_level]['segformerb5']['images']['inclusion']['RMSE']['mean']
        unet_drop = 100 * (unet_features - unet_img) / unet_features
        deeplabv3_drop = 100 * (deeplabv3_features - deeplabv3_img) / deeplabv3_features
        segformer_drop = 100 * (segformer_features - segformer_img) / segformer_features
        print(f"{noise_level}: mean RMSE drop: {(unet_drop + deeplabv3_drop + segformer_drop) / 3:.2f} %")
        
    # print mean IoU and MAE of CNNs vs segformerb5 for binary semantic segmentation of inclusion
    CNNs_bg_IoU = []
    CNNs_inc_IoU = []
    CNNs_bg_MAE = []
    CNNs_inc_MAE = []
    for noise_level, model in itertools.product(NOISE_LEVELS, ['deeplabv3_resnet101', 'Unet']):
        IoU_bg = summary_dict['binary'][noise_level][model]['images']['bg']['IOU']['mean']
        IoU_inc = summary_dict['binary'][noise_level][model]['images']['inclusion']['IOU']['mean']
        MAE_bg = summary_dict['regression'][noise_level][model]['images']['bg']['MAE']['mean'] * 1e-3 * 1e9 # scale MAE from [M] to [10^-9 M]
        MAE_inc = summary_dict['regression'][noise_level][model]['images']['inclusion']['MAE']['mean'] * 1e-3 * 1e9 # scale MAE from [M] to [10^-9 M]
        CNNs_bg_IoU.append(IoU_bg)
        CNNs_inc_IoU.append(IoU_inc)
        CNNs_bg_MAE.append(MAE_bg)
        CNNs_inc_MAE.append(MAE_inc)
    segformer_bg_IoU = []
    segformer_inc_IoU = []
    segformer_bg_MAE = []
    segformer_inc_MAE = []
    for noise_level in NOISE_LEVELS:
        IoU_bg = summary_dict['binary'][noise_level]['segformerb5']['images']['bg']['IOU']['mean']
        IoU_inc = summary_dict['binary'][noise_level]['segformerb5']['images']['inclusion']['IOU']['mean']
        MAE_bg = summary_dict['regression'][noise_level]['segformerb5']['images']['bg']['MAE']['mean'] * 1e-3 * 1e9 # scale MAE from [M] to [10^-9 M]
        MAE_inc = summary_dict['regression'][noise_level]['segformerb5']['images']['inclusion']['MAE']['mean'] * 1e-3 * 1e9 # scale MAE from [M] to [10^-9 M]
        segformer_bg_IoU.append(IoU_bg)
        segformer_inc_IoU.append(IoU_inc)
        segformer_bg_MAE.append(MAE_bg)
        segformer_inc_MAE.append(MAE_inc)
    # exclude outlier (unet that did not converge)
    #outlier_idx = np.argmax(np.asarray(CNNs_inc_MAE))
    #CNNs_bg_MAE.pop(outlier_idx)
    #CNNs_inc_MAE.pop(outlier_idx)
    
    print(f"Mean bg IoU of CNNs: {sum(CNNs_bg_IoU) / len(CNNs_bg_IoU):.3f}")
    print(f"Mean inclusion IoU of CNNs: {sum(CNNs_inc_IoU) / len(CNNs_inc_IoU):.3f}")
    print(f"Mean bg IoU of SegFormer-B5: {sum(segformer_bg_IoU) / len(segformer_bg_IoU):.3f}")
    print(f"Mean inclusion IoU of SegFormer-B5: {sum(segformer_inc_IoU) / len(segformer_inc_IoU):.3f}")
    
    
    print(f"Mean bg MAE of CNNs: {sum(CNNs_bg_MAE) / len(CNNs_bg_MAE):.3f}, 10^-9 M")
    print(f"Mean inclusion MAE of CNNs: {sum(CNNs_inc_MAE) / len(CNNs_inc_MAE):.3f}, 10^-9 M")
    print(f"Mean bg MAE of SegFormer-B5: {sum(segformer_bg_MAE) / len(segformer_bg_MAE):.3f}, 10^-9 M")
    print(f"Mean inclusion MAE of SegFormer-B5: {sum(segformer_inc_MAE) / len(segformer_inc_MAE):.3f}, 10^-9 M")