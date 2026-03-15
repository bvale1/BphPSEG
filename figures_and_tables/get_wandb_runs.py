
import os
import json
import itertools
import wandb
import numpy as np
from dotenv import load_dotenv


MODELS = [
    'RF',
    'XGB',
    'mlp',
    'Unet',
    'deeplabv3_resnet101',
    'segformer',
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
for gt_type, noise_level, model, input_type in itertools.product(GT_TYPES, NOISE_LEVELS, MODELS, INPUT_TYPES):
    if model in ['mlp', 'XGB', 'RF'] and input_type == 'images':
        continue
        
    sample_metrics_dict[gt_type][noise_level][model] = {
        'features': {
            'bg':        {},
            'inclusion': {},
        },
        'images': {
            'bg':        {},
            'inclusion': {},
        }
    }

# Cache runs per project to avoid repeated API calls
project_runs: dict[str, list] = {}
# Optional: Filter by tags, state, etc. (see W&B API docs)
FILTER = {}  # You can use {"state": "finished"} to get only completed runs
runs = api.runs("aisurrey_photoacoustics/BphPSEG2", filters=FILTER)

for run in runs:
    
    noise_level = run.notes
    name = run.name
    [model, input_type, gt_type] = name.split('_')

    artifacts = [
        a for a in run.logged_artifacts()
        if a.type == "dataset" and "test_per_sample_metrics" in a.name
    ]
    if not artifacts:
        continue
    
    # if multiple versions were logged, take latest version for this run
    artifact = max(artifacts, key=lambda a: int(a.version.lstrip("v")))

    local_dir = artifact.download(root=f"/tmp/wandb_artifacts/{run.id}")
    with open(os.path.join(local_dir, "bg.json"), "r") as f:
        bg_dict = json.load(f)
    with open(os.path.join(local_dir, "inclusion.json"), "r") as f:
        inclusion_dict = json.load(f)

    sample_metrics_dict[gt_type][noise_level][model][input_type]['bg'] = bg_dict
    sample_metrics_dict[gt_type][noise_level][model][input_type]['inclusion'] = inclusion_dict

    
    
# save as json
with open(os.path.join(os.path.dirname(__file__), 'per_sample_metrics.json'), 'w') as f:
    json.dump(sample_metrics_dict, f, indent=4)
# reduces sample-level metrics to median and inter-quartile range
summary_metrics_dict = sample_metrics_dict.copy()
for gt_type, noise_level, model in itertools.product(sample_metrics_dict.keys(), sample_metrics_dict['binary'].keys(), MODELS):
    for input_type in ('features', 'images'):
        if model in ['mlp', 'XGB', 'RF'] and input_type == 'images':
            continue
        for mask_type in ('bg', 'inclusion'):
            for metric, values in sample_metrics_dict[gt_type][noise_level][model][input_type][mask_type].items():
                if metric == 'sample_names':
                    continue
                metric_values = [v[1] for v in values]
                if metric_values:
                    median = float(np.nanmedian(np.asarray(metric_values)))
                    q1 = float(np.nanquantile(np.asarray(metric_values), 0.25))
                    q3 = float(np.nanquantile(np.asarray(metric_values), 0.75))
                else:
                    median = None
                    q1 = None
                    q3 = None
                summary_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = {
                    'median': median,
                    'q1': q1,
                    'q3': q3,
                }
with open(os.path.join(os.path.dirname(__file__), 'summary_metrics.json'), 'w') as f:
    json.dump(summary_metrics_dict, f, indent=4)