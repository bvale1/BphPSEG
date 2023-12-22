import torch
from torchvision import transforms


class Normalise(object):
    
    def __init__(self, max_val, min_val):
        self.max_val = max_val.unsqueeze(-1).unsqueeze(-1)
        self.min_val = min_val.unsqueeze(-1).unsqueeze(-1)
        
    def __call__(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)
    
    def inverse(self, tensor):
        # use to convert back to original scale
        return (tensor * (self.max_val - self.min_val)) + self.min_val


class ReplaceNaNWithZero(object):
    def __call__(self, tensor):
        tensor[torch.isnan(tensor)] = 0.0
        return tensor
    

class BinaryMaskToLabel(object):
    def __call__(self, tensor):
        return torch.stack([~tensor, tensor], dim=0).to(dtype=torch.float32)