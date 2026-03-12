# script to load a preprocessed dataset and compute masks for the inclusions.
# inclusion masks were not originally saved when the datasets were created.
# inclusion masks are used to compute metrics in inclusions

import os
from glob import glob

import h5py
import numpy as np
import matplotlib.pyplot as plt

dataset_path = "preprocessing/20240517_BphP_cylinders_no_noise/dataset.h5"
#dataset_path = "preprocessing/20240502_BphP_cylinders_noise_std2/dataset.h5"
#dataset_path = "preprocessing/20240517_BphP_cylinders_noise_std6/dataset.h5"
simulations_path = "/media/billy/Seagate Hub/cluster_MSOT_simulations/BphP_phantom"

dataset = h5py.File(dataset_path, "r+")

# group names are sample names ['c139519.p0', 'c139519.p1', ...]
# keys for each sample are ['bg_mask', 'c_mask', 'c_tot', 'features', 'images']

# 1) for each sample name in dataset go to simulation directory and find the folder with the sample name in the folder name
# 2) load mus = ['background_mua_mus'][0,1,:,:] from data.h5 in the folder
# 3) compute inclusion mask from the pixels corresponding to the unique values of mu_s excluding the bottom two unique values (as these correpond to background)
# 4) save 'inclusion_mask' in the dataset file under the corresponding sample name
# 5) make a figure with 10 subplots from 5 samples, for each sample show the inclusion mask and the corresponding mus


def find_simulation_folder(sample_name: str, base_dir: str) -> str:
    matches = [p for p in glob(os.path.join(base_dir, "*")) if sample_name in os.path.basename(p)]
    if not matches:
        raise FileNotFoundError(f"No simulation folder found for sample '{sample_name}'.")
    if len(matches) > 1:
        matches.sort()
    return matches[0]


def compute_inclusion_mask(mus: np.ndarray) -> np.ndarray:
    unique_vals = np.unique(mus)
    if unique_vals.size <= 2:
        return np.zeros_like(mus, dtype=bool)
    inclusion_vals = unique_vals[2:]
    return np.isin(mus, inclusion_vals)


sample_names = list(dataset.keys())

for i, sample_name in enumerate(sample_names):
    if i % 100 == 0:
        print(f"Processing sample {i} of {len(sample_names)}: {sample_name}")
    sim_folder = find_simulation_folder(sample_name, simulations_path)
    data_path = os.path.join(sim_folder, "data.h5")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data.h5 for sample '{sample_name}' in '{sim_folder}'.")

    with h5py.File(data_path, "r") as sim_data:
        mus = sim_data["background_mua_mus"][0, 1, :, :]

    inclusion_mask = compute_inclusion_mask(mus)

    sample_group = dataset[sample_name]
    if "inclusion_mask" in sample_group:
        del sample_group["inclusion_mask"]
    sample_group.create_dataset(
        "inclusion_mask",
        data=inclusion_mask.astype(bool),
    )


num_samples_to_plot = min(5, len(sample_names))
if num_samples_to_plot > 0:
    fig, axes = plt.subplots(num_samples_to_plot, 3, figsize=(8, 3 * num_samples_to_plot), layout='constrained')

    for idx in range(num_samples_to_plot):
        sample_name = sample_names[idx]
        sim_folder = find_simulation_folder(sample_name, simulations_path)
        data_path = os.path.join(sim_folder, "data.h5")

        with h5py.File(data_path, "r") as sim_data:
            mus = np.asarray(sim_data["background_mua_mus"][0, 1, :, :], dtype=np.float32)

        inclusion_mask = np.asarray(dataset[sample_name]["inclusion_mask"][()], dtype=bool)
        bg_mask = np.asarray(dataset[sample_name]["bg_mask"][()], dtype=bool)
        
        axes[idx, 1].imshow(mus, cmap="viridis")
        axes[idx, 0].imshow(inclusion_mask, cmap="gray")
        axes[idx, 2].imshow(bg_mask, cmap="gray")
        
    fig.savefig("inclusion_masks_overview.png", dpi=200)
    

dataset.close()