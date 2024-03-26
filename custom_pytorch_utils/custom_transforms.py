import torch
import numpy as np


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
        # use when the tensor is a flattened numpy array (sklearn models)
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