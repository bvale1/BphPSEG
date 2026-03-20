import numpy as np
import matplotlib.pyplot as plt
import json
import itertools
import os
from matplotlib.patches import Patch

def find_and_replace_key(d: dict, target_key: str, new_key: str) -> dict:
    if target_key in d:
        d[new_key] = d.pop(target_key)
    for k, v in d.items():
        if isinstance(v, dict):
            d = find_and_replace_key(v, target_key, new_key)
    return d


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

BINARY_METRICS = [
    'Dice',
    'IOU',
    'MCC',
    'Sensitivity',
    'Specificity',
    'Accuracy',
]

REGRESSION_METRICS = [
    'RMSE',
    'MAE',
    'R2',
]

def make_violin_plots(
        sample_metrics_dict: dict, 
        models: list[str],
        input_types: list[str],
        noise_levels: list[str], 
        gt_type: str,
        metric: str,
        save_path: str
    ) -> None:

    sample_metrics_dict = sample_metrics_dict[gt_type]
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), layout='constrained', sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    model_input_pairs = []
    x_tics = []
    for model, input_type in itertools.product(models, input_types):
        has_data = False
        for noise_level in noise_levels:
            metric_root = (
                sample_metrics_dict
                .get(noise_level, {})
                .get(model, {})
                .get(input_type, {})
            )
            inclusion_values = metric_root.get('inclusion', {}).get(metric, [])
            bg_values = metric_root.get('bg', {}).get(metric, [])
            if len(inclusion_values) > 0 or len(bg_values) > 0:
                has_data = True
                break

        if has_data:
            model_input_pairs.append((model, input_type))
            x_tics.append(f"{model}\n{input_type}")

    if not model_input_pairs:
        raise ValueError(f"No data found for gt_type={gt_type}, metric={metric}")

    x_positions = np.arange(len(model_input_pairs), dtype=float)
    offsets = np.linspace(-0.28, 0.28, len(noise_levels))
    box_width = 0.20
    jitter_width = 0.05
    rng = np.random.default_rng(42)

    for i, noise_level in enumerate(noise_levels):
        inclusion_data = []
        bg_data = []
        inclusion_positions = []
        bg_positions = []

        for j, (model, input_type) in enumerate(model_input_pairs):
            metric_root = (
                sample_metrics_dict
                .get(noise_level, {})
                .get(model, {})
                .get(input_type, {})
            )

            inclusion_values = metric_root.get('inclusion', {}).get(metric, [])
            bg_values = metric_root.get('bg', {}).get(metric, [])

            if len(inclusion_values) > 0:
                inclusion_data.append(inclusion_values)
                inclusion_positions.append(x_positions[j] + offsets[i])
            if len(bg_values) > 0:
                bg_data.append(bg_values)
                bg_positions.append(x_positions[j] + offsets[i])

        if inclusion_data:
            bp_inc = ax[0].boxplot(
                inclusion_data,
                positions=inclusion_positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={'color': 'black', 'linewidth': 1.3},
                whiskerprops={'color': 'black', 'linewidth': 1.0},
                capprops={'color': 'black', 'linewidth': 1.0},
                boxprops={'edgecolor': 'black', 'linewidth': 1.0},
            )
            for box in bp_inc['boxes']:
                box.set_facecolor(colors[i % len(colors)])
                box.set_alpha(0.45)
            for x_pos, values in zip(inclusion_positions, inclusion_data):
                values_arr = np.asarray(values)
                x_jitter = x_pos + rng.uniform(-jitter_width, jitter_width, size=values_arr.shape[0])
                ax[0].scatter(
                    x_jitter,
                    values_arr,
                    s=9,
                    color=colors[i % len(colors)],
                    alpha=0.6,
                    edgecolors='none',
                    zorder=3,
                )

        if bg_data:
            bp_bg = ax[1].boxplot(
                bg_data,
                positions=bg_positions,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                medianprops={'color': 'black', 'linewidth': 1.3},
                whiskerprops={'color': 'black', 'linewidth': 1.0},
                capprops={'color': 'black', 'linewidth': 1.0},
                boxprops={'edgecolor': 'black', 'linewidth': 1.0},
            )
            for box in bp_bg['boxes']:
                box.set_facecolor(colors[i % len(colors)])
                box.set_alpha(0.45)
            for x_pos, values in zip(bg_positions, bg_data):
                values_arr = np.asarray(values)
                x_jitter = x_pos + rng.uniform(-jitter_width, jitter_width, size=values_arr.shape[0])
                ax[1].scatter(
                    x_jitter,
                    values_arr,
                    s=9,
                    color=colors[i % len(colors)],
                    alpha=0.6,
                    edgecolors='none',
                    zorder=3,
                )

    ax[0].set_title(f"{gt_type.capitalize()} {metric}")
    ax[0].set_ylabel('Inclusion')
    ax[1].set_ylabel('Background')
    ax[1].set_xlabel('Model / Input Type')

    for axis in ax:
        axis.grid(axis='y', alpha=0.25)

    ax[1].set_xticks(x_positions)
    ax[1].set_xticklabels(x_tics)

    legend_handles = [
        Patch(facecolor=colors[i % len(colors)], edgecolor='black', label=noise_level)
        for i, noise_level in enumerate(noise_levels)
    ]
    ax[0].legend(handles=legend_handles, title='Noise Level', ncol=len(noise_levels), loc='upper right')

    fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
        
base_dir = os.path.dirname(os.path.abspath(__file__))
sample_metrics_dict = json.load(open(os.path.join(base_dir, 'per_sample_metrics.json'), 'r'))
make_violin_plots(sample_metrics_dict, MODELS, INPUT_TYPES, NOISE_LEVELS, GT_TYPES[0], 'MCC', 'binary_plots.svg')
make_violin_plots(sample_metrics_dict, MODELS, INPUT_TYPES, NOISE_LEVELS, GT_TYPES[1], 'MAE', 'regression_plots.svg')