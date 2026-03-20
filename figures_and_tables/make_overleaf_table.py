import json
import itertools
import os

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
    #'Unet',
    #'deeplabv3_resnet101',
    #'segformerb5',
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

base_dir = os.path.dirname(os.path.abspath(__file__))
summary_dict = json.load(open(os.path.join(base_dir, 'summary_metrics.json'), 'r'))

metric1 = 'Dice'
metric2 = 'MCC'
caption = 'The performance metrics of the models applied in the binary semantic segmentation of the photoswitchable proteins for the three datasets. Values are shown as mean $\pm$ 95 \% confidence interval.'

head = f"""
\begin{{table}}[H]
\centering
\footnotesize
\caption{{The performance metrics of the models applied in the \textcolor{{red}}{{binary semantic segmentation}} of the photoswitchable proteins for the three datasets. Values are shown as mean $\pm$ 95 \% confidence interval.}}
\vspace{{0.25cm}}
%\begin{{adjustbox}}{{width=1\textwidth}}
\begin{{tabular}}{{|l|l|l|l|l|l|l|l|}}
\hline 
\multicolumn{{8}}{{|l|}}{{\textcolor{{red}}{{Binary Semantic Segmentation}} Performance Metrics}} \\
\hline
\multirow{{2}}{{*}}{{Model}} & \multirow{{2}}{{*}}{{Input}} & \multicolumn{{2}}{{|l|}}{{Dataset 1 (no noise)}} & \multicolumn{{2}}{{|l|}}{{Dataset 2 ($\textrm{{SNR}}_{{\textrm{{dB}}}}=18.4$)}} & \multicolumn{{2}}{{|l|}}{{Dataset 3 ($\textrm{{SNR}}_{{\textrm{{dB}}}}=8.8$)}} \\
\cline{{3-8}}
& & {metric1} & {metric2} & {metric1} & {metric2} & {metric1} & {metric2} \\
"""

print(head)

for model, input_type in itertools.product(MODELS, INPUT_TYPES):
    
    if model in ['mlp', 'XGB', 'RF'] and input_type == 'images':
        continue
    
    means = summary_dict['binary'][f'noise_std{NOISE_LEVELS.index("noise_std0")}'][model][input_type]
    CI2 = summary_dict['binary'][f'noise_std{NOISE_LEVELS.index("noise_std0")}'][model][input_type + '_CI2']
    
    row1 = f"\multirow{{2}}{{*}}{{{model}}} & \multirow{{2}}{{*}}{{{input_type}}} & {means[metric1][0]:.3f} $\pm$ {means[metric1][1]:.3f} & {means[metric2][0]:.3f} $\pm$ {means[metric2][1]:.3f} & {means[metric1][0]:.3f} $\pm$ {means[metric1][1]:.3f} & {means[metric2][0]:.3f} $\pm$ {means[metric2][1]:.3f} & {means[metric1][0]:.3f} $\pm$ {means[metric1][1]:.3f} & {means[metric2][0]:.3f} $\pm$ {means[metric2][1]:.3f} \\"
    
    row2 = f"& & $\pm$ {CI2[metric1][0]:.3f} & $\pm$ {CI2[metric2][0]:.3f} & $\pm$ {CI2[metric1][0]:.3f} & $\pm$ {CI2[metric2][0]:.3f} & $\pm$ {CI2[metric1][0]:.3f} & $\pm$ {CI2[metric2][0]:.3f} \\"
    print(row1)
    print(row2)
    print("\\hline")
    
foot = f"""
\hline
\end{{tabular}}
%\end{{adjustbox}}
\label{{tab:binary_performance}}
\end{{table}}
"""
print(foot)