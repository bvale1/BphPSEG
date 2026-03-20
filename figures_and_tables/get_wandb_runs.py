
import os
import json
import itertools
import wandb
import numpy as np
from dotenv import load_dotenv
from copy import deepcopy


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
for gt_type, noise_level, model in itertools.product(GT_TYPES, NOISE_LEVELS, MODELS):
    if model in ['mlp', 'XGB', 'RF']:
        sample_metrics_dict[gt_type][noise_level][model] = {
            'features': {
                'bg':        {},
                'inclusion': {},
            },
        }
    else:        
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
    
    # skip failed runs
    if run.state != 'finished':
        continue
    
    noise_level = run.notes
    name = run.name
    if len(name.split('_')) == 3:
        [model, input_type, gt_type] = name.split('_')
    else:
        # case model = 'deeplabv3_resnet101'
        [model1, model2, input_type, gt_type] = name.split('_')
        model = f'{model1}_{model2}'
        
    if model not in MODELS or input_type not in INPUT_TYPES or gt_type not in GT_TYPES or noise_level not in NOISE_LEVELS:
        print(f"Skipping run {run.id} with model {model}, input_type {input_type}, gt_type {gt_type}, noise_level {noise_level}")
        continue
    
    if model in ['mlp', 'XGB', 'RF'] and input_type == 'images':
        continue

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

    # need this to be a list of lists for each key in `bg_dict` and `inclusion_dict` instead of a list of dicts
    for metric in bg_dict.keys():
        if metric not in sample_metrics_dict[gt_type][noise_level][model][input_type]['bg']:
            sample_metrics_dict[gt_type][noise_level][model][input_type]['bg'][metric] = []
        sample_metrics_dict[gt_type][noise_level][model][input_type]['bg'][metric].append(bg_dict[metric])
    for metric in inclusion_dict.keys():
        if metric not in sample_metrics_dict[gt_type][noise_level][model][input_type]['inclusion']:
            sample_metrics_dict[gt_type][noise_level][model][input_type]['inclusion'][metric] = []
        sample_metrics_dict[gt_type][noise_level][model][input_type]['inclusion'][metric].append(inclusion_dict[metric])
    
    

# reduces sample-level metrics to mean and percentile-based 95% CI half-width
summary_metrics_dict = deepcopy(sample_metrics_dict)
for gt_type, noise_level, model in itertools.product(sample_metrics_dict.keys(), sample_metrics_dict['binary'].keys(), MODELS):
    
    for input_type in ('features', 'images'):
        
        if model in ['mlp', 'XGB', 'RF'] and input_type == 'images':
            continue
        
        for mask_type in ('bg', 'inclusion'):
            
            for metric, values in sample_metrics_dict[gt_type][noise_level][model][input_type][mask_type].items():
                # values should be a list of lists [5, n_samples]
                if not values:
                    print(f"no values for {model}/{input_type}/{gt_type}/{noise_level}/{mask_type}/{metric}")
                    summary_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = {
                        'mean': None,
                        'ci95': None,
                    }
                    continue
                
                if metric == 'sample_names':
                    sample_names = summary_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric][0]
                    sample_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = sample_names
                    continue
                
                if len(values) != 5:
                    print(f"Warning: explected 5 runs for {model}/{input_type}/{gt_type}/{noise_level}/{mask_type}/{metric} but got {len(values)}")
                
                # compute mean for each sample
                metric_values = np.asarray(values).mean(axis=0)
                sample_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = metric_values.tolist()
                # print top and bottom three samples for this metric
                highest_idx = np.argsort(metric_values)[-3:]
                lowest_idx = np.argsort(metric_values)[:3]
                #print(f"{model}/{input_type}/{gt_type}/{noise_level}/{mask_type}/{metric}")
                #print(f"  highest: {[(sample_names[i], metric_values[i]) for i in highest_idx]}")
                #print(f"  lowest: {[(sample_names[i], metric_values[i]) for i in lowest_idx]}")

                metric_values = np.asarray(values).flatten()
                valid_values = metric_values[~np.isnan(metric_values)]

                if valid_values.size == 0:
                    mean = None
                    ci95 = None
                else:
                    mean = float(np.mean(valid_values))
                    p2_5 = float(np.nanpercentile(valid_values, 2.5))
                    p97_5 = float(np.nanpercentile(valid_values, 97.5))
                    ci95 = float((p97_5 - p2_5) / 2.0)
                    
                summary_metrics_dict[gt_type][noise_level][model][input_type][mask_type][metric] = {
                    'mean': mean,
                    'ci95': ci95,
                }
# save as json
with open(os.path.join(os.path.dirname(__file__), 'per_sample_metrics.json'), 'w') as f:
    json.dump(sample_metrics_dict, f, indent=4)
with open(os.path.join(os.path.dirname(__file__), 'summary_metrics.json'), 'w') as f:
    json.dump(summary_metrics_dict, f, indent=4)