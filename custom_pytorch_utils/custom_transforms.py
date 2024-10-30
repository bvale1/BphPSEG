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
        # use when the tensor is a flattened numpy array (sklearn models)
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
        root_dir, 
        gt_type, 
        input_type, 
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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=20
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=20
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=20
    )
    
    return (train_loader, val_loader, test_loader, dataset, train_dataset,
            test_dataset, Y_mean, normalise_y, normalise_x)