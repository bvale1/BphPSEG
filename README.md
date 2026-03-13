# BphPSEG
Preprocceses data from python_BphP_MSOT_sim into datasets, trains classical machine learning models and Unet on the datasets

# Recommended
- Linux
- Cuda ready device to run pytorch models
- Weights and biases (wandb) account to log and access experiment results

# Instructions (linux only)
Option 1:
Install required dependacies in a uv python venv 
```
pip install uv
uv venv .venv --python 3.11
uv init
uv add -r requirements.txt
```
Run one of the scripts with the uv venv
```
uv run train_BphPSEG.py
uv run train_xgboost_ML.py
```


Option 2:
Build and run through docker container (requires docker)
```
sudo docker service start
sudo docker build . -t bphpseg:BphPSEG
```
Run scripts using container
```
sudo docker run --gpus all -v "$PWD":/app -w /app bphpseg:BphPSEG python3 train_BphPSEG.py
```