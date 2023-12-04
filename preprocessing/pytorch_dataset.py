import numpy as np
import torch, h5py
from torch.utils.data import Dataset, DataLoader


class BphP_MSOT_Dataset(Dataset):
    def __init__(self, h5_file, gt_type, transform=None):
        self.h5_file = h5_file
        if gt_type not in ['binary', 'regression']:
            # use binary classification or value regression
            raise ValueError("gt_type must be either 'binary' or 'regression'")
        self.gt_type = gt_type
        self.transform = transform
        with h5py.File(self.h5_file, 'r') as f:
            self.samples = list(f.keys())
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> tuple:
        with h5py.File(self.h5_file, 'r') as f:
            sample = self.samples[index]
            X = torch.from_numpy(f[sample]['features'][()])
            if self.gt_type == 'binary':
                Y = torch.from_numpy(f[sample]['c_mask'][()])
            elif self.gt_type == 'regression':
                Y = torch.from_numpy(f[sample]['c_tot'][()])
        
        if self.transform:
            X = self.transform(X)
            if self.gt_type == 'regression':
                Y = self.transform(Y)
        
        return (X, Y)