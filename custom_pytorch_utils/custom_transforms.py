import torch
import numpy as np
from typing import Literal
from torchvision import transforms
from custom_pytorch_utils.custom_datasets import BphP_MSOT_Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class MaxMinNormalise(object):
    
    def __init__(self, max_val : torch.Tensor, min_val : torch.Tensor):
        self.max_val = max_val.unsqueeze(-1).unsqueeze(-1)
        self.min_val = min_val.unsqueeze(-1).unsqueeze(-1)
        
    def __call__(self, tensor : torch.Tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)
    
    def inverse(self, tensor : torch.Tensor):
        # use to convert back to original scale
        return (tensor * (self.max_val - self.min_val)) + self.min_val

    def inverse_numpy_flat(self, tensor : np.ndarray):
        # use when the tensor is a flattened numpy array (sklearn/xgboost models)
        return (tensor * (self.max_val.squeeze().numpy() - self.min_val.squeeze().numpy())) + self.min_val.squeeze().numpy()


class MeanStdNormalise(object):
    
    def __init__(self, mean : torch.Tensor, std : torch.Tensor):
        self.mean = mean.unsqueeze(-1).unsqueeze(-1)
        self.std = std.unsqueeze(-1).unsqueeze(-1)
        
    def __call__(self, tensor : torch.Tensor):
        return (tensor - self.mean) / self.std
    
    def inverse(self, tensor : torch.Tensor):
        # use to convert back to original scale
        return (tensor * self.std) + self.mean

    def inverse_numpy_flat(self, tensor : np.ndarray):
        # use when the tensor is a flattened numpy array (sklearn/xgboost models)
        return (tensor * self.std.squeeze().numpy()) + self.mean.squeeze().numpy()


class ReplaceNaNWithZero(object):
    def __call__(self, tensor : torch.Tensor):
        tensor[torch.isnan(tensor)] = 0.0
        return tensor
    

class BinaryMaskToLabel(object):
    # converts class labels to one-hot encoding
    def __call__(self, tensor : torch.Tensor):
        return torch.stack([~tensor, tensor], dim=0).to(dtype=torch.float32)
    
    
def create_dataloaders(
        root_dir : str,
        input_type : Literal['images', 'features'],
        gt_type : Literal['binary', 'regression'],
        normalisation_type : Literal['MinMax', 'MeanStd'],
        batch_size : int,
        config : dict
    ) -> tuple:
    
    # create training dataset with no normalisation
    dataset = BphP_MSOT_Dataset(
        dataset_path=root_dir, 
        gt_type='regression', 
        input_type=input_type, 
        x_transform=ReplaceNaNWithZero(),
        y_transform=ReplaceNaNWithZero(),
    )
    
    train_dataset, _, _ = random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # train/val/test sets are always the same
    )
    
    # compute training dataset statistics for normalisation
    config = compute_normalisation_statistics(dataset, input_type, config)

    if input_type == 'images':
        if normalisation_type == 'MinMax':
            normalise_x = MaxMinNormalise(
                torch.Tensor(config['image_normalisation_params']['max']),
                torch.Tensor(config['image_normalisation_params']['min'])
            )
            x_transform = transforms.Compose([
                ReplaceNaNWithZero(), 
                normalise_x
            ])
        elif normalisation_type == 'MeanStd':
            normalise_x = MeanStdNormalise(
                torch.Tensor(config['image_normalisation_params']['mean']),
                torch.Tensor(config['image_normalisation_params']['std'])
            )
            x_transform = transforms.Compose([
                ReplaceNaNWithZero(), 
                normalise_x
            ])
        
    elif input_type == 'features':
        if normalisation_type == 'MinMax':
            normalise_x = MaxMinNormalise(
                torch.Tensor(config['feature_normalisation_params']['max']),
                torch.Tensor(config['feature_normalisation_params']['min'])
            )
            x_transform = transforms.Compose([
                ReplaceNaNWithZero(),
                normalise_x
            ])
        elif normalisation_type == 'MeanStd':
            normalise_x = MeanStdNormalise(
                torch.Tensor(config['feature_normalisation_params']['mean']),
                torch.Tensor(config['feature_normalisation_params']['std'])
            )
            x_transform = transforms.Compose([
                ReplaceNaNWithZero(),
                normalise_x
            ])
            
    if gt_type == 'binary':
        normalise_y = None
        y_transform = transforms.Compose([
            ReplaceNaNWithZero(),
            BinaryMaskToLabel()
        ])
    elif gt_type == 'regression':
        if normalisation_type == 'MinMax':
            normalise_y = MaxMinNormalise(
                torch.Tensor(config['concentration_normalisation_params']['max']),
                torch.Tensor(config['concentration_normalisation_params']['min'])
            )
        elif normalisation_type == 'MeanStd':
            normalise_y = MeanStdNormalise(
                torch.Tensor(config['concentration_normalisation_params']['mean']),
                torch.Tensor(config['concentration_normalisation_params']['std'])
            )
        y_transform = transforms.Compose([
            ReplaceNaNWithZero(),
            normalise_y
        ])
        
    Y_mean = torch.Tensor(config['concentration_normalisation_params']['mean'])
            
    dataset = BphP_MSOT_Dataset(
        dataset_path=root_dir, 
        gt_type=gt_type, 
        input_type=input_type, 
        x_transform=x_transform,
        y_transform=y_transform
    )
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # train/val/test sets are always the same
    )
    print(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: \
        {len(test_dataset)}')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return (train_loader, val_loader, test_loader, dataset, train_dataset,
             val_dataset, test_dataset, Y_mean, normalise_y, normalise_x)
    

def compute_normalisation_statistics(
        dataset: BphP_MSOT_Dataset,
        input_type : Literal['images', 'features'],
        config : dict
    ) -> dict:
    # use the first pass to compute the mean, second pass to compute the std from mean
    (X, Y, _, _, _) = dataset[0]
    X_np = X.numpy()
    Y_np = Y.numpy()
    n_samples = dataset.__len__()

    if input_type == "features":
        feature_max = np.empty((n_samples, X_np.shape[0]))
        feature_min = feature_max.copy()
        feature_mean = feature_max.copy()
    else:
        image_max = np.empty(n_samples)
        image_min = image_max.copy()
        image_mean = image_max.copy()
    
    c_max = np.empty(n_samples)
    c_min = c_max.copy()
    c_mean = c_max.copy()


    for i in range(n_samples):
        (X, Y, _, _, _) = dataset[i]
        X_np = X.numpy()
        Y_np = Y.numpy()

        if input_type == 'features':
            feature_max[i] = np.max(X_np, axis=(1, 2))
            feature_min[i] = np.min(X_np, axis=(1, 2))
            feature_mean[i] = np.mean(X_np, axis=(1, 2))
        else:
            image_max[i] = np.max(X_np)
            image_min[i] = np.min(X_np)
            image_mean[i] = np.mean(X_np)

        c_max[i] = np.max(Y_np)
        c_min[i] = np.min(Y_np)
        c_mean[i] = np.mean(Y_np)

    if input_type == 'features':
        feature_max = np.max(feature_max, axis=0)
        feature_min = np.min(feature_min, axis=0)
        feature_mean = np.mean(feature_mean, axis=0)
    else:
        image_max = [np.max(image_max)]
        image_min = [np.min(image_min)]
        image_mean = [np.mean(image_mean)]

    c_max = [np.max(c_max)]
    c_min = [np.min(c_min)]
    c_mean = [np.mean(c_mean)]

    if input_type == 'features':
        feature_ssr = np.zeros_like(feature_mean, dtype=np.float64)
        features_n = n_samples * np.prod(X_np.shape[-2:])
    else:
        image_ssr = np.float64(0.0)
        images_n = n_samples * np.prod(X_np.shape)

    c_ssr = np.float64(0.0)
    c_n = n_samples * np.prod(Y_np.shape)

    for i in range(n_samples):
        (X, Y, _, _, _) = dataset[i]
        X_np = X.numpy()
        Y_np = Y.numpy()

        if input_type == 'features':
            feature_ssr += np.sum(
                (X_np - feature_mean[:, np.newaxis, np.newaxis])**2,
                axis=(1, 2)
            )
        else:
            image_ssr += np.sum((X_np - image_mean)**2)

        c_ssr += np.sum((Y_np - c_mean)**2)

    if input_type == 'features':
        feature_std = np.sqrt(feature_ssr / (features_n - 1))
        image_std = None
    else:
        feature_std = None
        image_std = np.sqrt(image_ssr / (images_n - 1))

    c_std = np.sqrt(c_ssr / (c_n - 1))
    
    if input_type == "images":
        config['image_normalisation_params']['mean'] = image_mean
        config['image_normalisation_params']['std'] = image_std
        config['image_normalisation_params']['max'] = image_max
        config['image_normalisation_params']['min'] = image_min
    else:
        config['feature_normalisation_params']['mean'] = feature_mean
        config['feature_normalisation_params']['std'] = feature_std
        config['feature_normalisation_params']['max'] = feature_max
        config['feature_normalisation_params']['min'] = feature_min
        
    config['concentration_normalisation_params']['mean'] = c_mean
    config['concentration_normalisation_params']['std'] = c_std
    config['concentration_normalisation_params']['max'] = c_max
    config['concentration_normalisation_params']['min'] = c_min

    return config