import subprocess
import os
import itertools

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
    sub_file = f"""
    cat << EOF > submit_BphPSEG.sh
    #!/bin/bash

    ### Job Name ###
    #SBATCH --job-name="BphPSEG"

    ## CPU core requirements ###
    #SBATCH --nodes=$nodes
    #SBATCH --cpus-per-task=$cpus_per_task
    #SBATCH --ntasks-per-node=$ntasks_per_node

    ### CPU Memory (RAM) requirements ###
    #SBATCH --mem=32G

    ### GPU requirements ###
    #SBATCH --partition=3090
    #SBATCH --gpus=1

    ### Max. time requirement - DD-HH:MM:SS ###
    #SBATCH --time=00-02:00:00

    ### Job log files ###
    #SBATCH -o slurm.%j.%N.out
    #SBATCH -e slurm.%j.%N.err

    ### Apptainer execution ###
    apptainer exec oras://container-registry.surrey.ac.uk/shared-containers/billy-lightning-container:latest \
    python3 \$PWD/clone_and_run_msot_diffusion.py \
    --cluster_id .N\$SLURM_JOB_NODELIST.j\$SLURM_JOB_ID \
    --root_dir {root_dir} \
    --save_dir {model} \
    --model {model} \
    --input_type {input_type} \
    --wandb_notes {wandb_notes} \
    --gt_type {gt_type} \
    --input_normalization minmax \
    --seed {seed}
    EOF
    sbatch submit_BphPSEG.sh
    done
    """
    # save and run sub file
    subprocess.run(['bash', '-c', sub_file], check=True)
    