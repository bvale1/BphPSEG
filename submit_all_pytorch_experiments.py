import subprocess
import os
import itertools
import textwrap

root_dir = 'preprocessing/20240517_BphP_cylinders_no_noise/'
wandb_notes = "noise_std0"

#seeds = [1, 2, 3, 4, 5]
seeds = [1]

#input_types = ['features', 'images']
input_types = ['features']

#models = ['mlp', 'Unet', 'deeplabv3_resnet101', 'segformer']
models = ['Unet']
#gt_types = ['binary', 'regression']
gt_types = ['binary']


for seed, model, input_type, gt_type in itertools.product(seeds, models, input_types, gt_types):
    sub_file = textwrap.dedent(f"""
    #!/bin/bash

    ### Job Name ###
    #SBATCH --job-name="BphPSEG"

    ## CPU core requirements ###
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=4
    #SBATCH --ntasks-per-node=1

    ### CPU Memory (RAM) requirements ###
    #SBATCH --mem=64G

    ### GPU requirements ###
    #SBATCH --partition=3090
    #SBATCH --gpus=1

    ### Max. time requirement - DD-HH:MM:SS ###
    #SBATCH --time=00-06:00:00

    ### Job log files ###
    #SBATCH -o slurm.%j.%N.out
    #SBATCH -e slurm.%j.%N.err

    ### Apptainer execution ###
    apptainer exec docker://container-registry.surrey.ac.uk/shared-containers/billy-test-container:latest \
    python3 $PWD/clone_and_run_msot_diffusion.py \
    --cluster_id .N$SLURM_JOB_NODELIST.j$SLURM_JOB_ID \
    --root_dir {root_dir} \
    --save_dir {model} \
    --model {model} \
    --input_type {input_type} \
    --wandb_notes {wandb_notes} \
    --gt_type {gt_type} \
    --input_normalisation MinMax \
    --seed {seed}
    """).strip() + "\n"

    submit_script_path = 'submit_BphPSEG.sh'
    with open(submit_script_path, 'w', encoding='utf-8') as f:
        f.write(sub_file)
    os.chmod(submit_script_path, 0o755)
    subprocess.run(['sbatch', submit_script_path], check=True)
    