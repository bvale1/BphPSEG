
import os
import json
import itertools
import pandas as pd
import wandb
from dotenv import load_dotenv


MODELS = [
    'RF',
    'XGB',
    'mlp',
    'Unet',
    'deeplabv3_resnet101',
    'segformer',
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

# Load wandb credentials from project-root .env if present
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path, override=True)
wandb_key = os.getenv('WANDB_API_KEY')
if wandb_key:
    wandb.login(key=wandb_key)

api = wandb.Api(timeout=6000)

# Initialise empty result structure
sample_metrics_dict: dict = {
    'binary': {
        'noise_std0' : {},
        'noise_std2' : {},
        'noise_std6' : {},
    },
    'regression': {
        'noise_std0' : {},
        'noise_std2' : {},
        'noise_std6' : {},
    },
}
for gt_type, noise_level, model in itertools.product(sample_metrics_dict.keys(), sample_metrics_dict['binary'].keys(), MODELS):
    if gt_type == 'binary':
        sample_metrics_dict[gt_type][noise_level][model] = {
            'features': {
                'bg':        {metric: [] for metric in BINARY_METRICS},
                'inclusion': {metric: [] for metric in BINARY_METRICS},
            },
            'images': {
                'bg':        {metric: [] for metric in BINARY_METRICS},
                'inclusion': {metric: [] for metric in BINARY_METRICS},
            }
        }
    else:
        sample_metrics_dict[gt_type][noise_level][model] = {
            'features': {
                'bg':        {metric: [] for metric in REGRESSION_METRICS},
                'inclusion': {metric: [] for metric in REGRESSION_METRICS},
            },
            'images': {
                'bg':        {metric: [] for metric in REGRESSION_METRICS},
                'inclusion': {metric: [] for metric in REGRESSION_METRICS},
            }
        }

# Cache runs per project to avoid repeated API calls
project_runs: dict[str, list] = {}

for model in MODELS:

    if project not in project_runs:
        project_runs[project] = list(api.runs(project))

    for gt_type in ('binary', 'regression'):
        metric_cols = BINARY_METRICS if gt_type == 'binary' else REGRESSION_METRICS
        run_name = f'{run_prefix}_{gt_type}'
        matching_runs = [r for r in project_runs[project] if r.name == run_name]

        for (run, mask_type) in zip(matching_runs, ('bg', 'inclusion')):
            table_key = f'test_per_sample_metrics_{mask_type}'
            # Tables are logged as artifacts named run-{run.id}-{table_key}
            for artifact in run.logged_artifacts():
                if table_key in artifact.name:
                    table = artifact.get(table_key)
                    if table is None:
                        break
                    df = pd.DataFrame(table.data, columns=table.columns)
                    for _, row in df.iterrows():
                        sample_name = row['sample_names']
                        for metric in metric_cols:
                            sample_metrics_dict[gt_type][model][mask_type][metric].append(
                                (sample_name, float(row[metric]))
                            )
                    break  # one table artifact per run
                
# save as json
with open(os.path.join(os.path.dirname(__file__), 'per_sample_metrics.json'), 'w') as f:
    json.dump(sample_metrics_dict, f, indent=4)
# reduces sample-level metrics to median and inter-quartile range
summary_metrics_dict = sample_metrics_dict.copy()
for gt_type, noise_level, model in itertools.product(sample_metrics_dict.keys(), sample_metrics_dict['binary'].keys(), MODELS):
    for input_type in ('features', 'images'):
        for mask_type in ('bg', 'inclusion'):
            for metric, values in sample_metrics_dict[gt_type][noise_level][model][input_type][mask_type].items():
                metric_values = [v[1] for v in values]
                if metric_values:
                    median = float(pd.Series(metric_values).median())
                    q1 = float(pd.Series(metric_values).quantile(0.25))
                    q3 = float(pd.Series(metric_values).quantile(0.75))
                else:
                    median = None
                    q1 = None
                    q3 = None
                summary_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = {
                    'median': median,
                    'q1': q1,
                    'q3': q3,
                }