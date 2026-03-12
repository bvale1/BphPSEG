import numpy as np
import torch
import h5py
import json
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class BphP_MSOT_Dataset(Dataset):
    def __init__(self,
                 dataset_path : str,
                 gt_type : str,
                 input_type : str,
                 x_transform=None, 
                 y_transform=None
                ):
        self.dataset_path = dataset_path
        self.h5_file = os.path.join(dataset_path, 'dataset.h5')
        if gt_type not in ['binary', 'regression']:
            raise ValueError("gt_type must be either 'binary' or 'regression'")
        if input_type not in ['images', 'features']:
            raise ValueError("input_type must be either 'images' or 'features'")
        
        self.input_type = input_type
        self.gt_type = gt_type
        self.x_transform = x_transform
        self.y_transform = y_transform
        
        with h5py.File(self.h5_file, 'r') as f:
            self.samples = list(f.keys())
            
        # load config file
        with open(os.path.join(os.path.dirname(self.dataset_path), 'config.json'), 'r') as f:
            self.config = json.load(f)
            
        if gt_type == 'binary':
            self.get_Y = lambda f, sample: torch.from_numpy(f[sample]['c_mask'][()])
        else: # regression
            self.get_Y = lambda f, sample: torch.from_numpy(f[sample]['c_tot'][()])
            
        if input_type == 'images':
            self.get_X = lambda f, sample:  torch.flatten(
                torch.from_numpy(f[sample]['images'][0,:]), start_dim=0, end_dim=1
            )
        else: # features
            self.get_X = lambda f, sample:  torch.from_numpy(f[sample]['features'][()])
                
            
    def __len__(self) -> int:
        return len(self.samples)
    
    
    def __getitem__(self, index : int) -> tuple:
        sample = self.samples[index]
        with h5py.File(self.h5_file, 'r') as f:
            X = self.get_X(f, sample)
            Y = self.get_Y(f, sample)
            inclusion_mask = torch.from_numpy(f[sample]['inclusion_mask'][()])
            bg_mask = torch.from_numpy(f[sample]['bg_mask'][()])
            bg_mask = bg_mask & (~inclusion_mask)
        
        if self.x_transform:
            X = self.x_transform(X)
        if self.y_transform:
            Y = self.y_transform(Y)
            
        return (X, Y, bg_mask, inclusion_mask, sample)


    def plot_sample(self, X, Y, Y_pred=None, save_name=None,
                    y_transform=None, x_transform=None):
        if y_transform: # option to undo previous invertable Y transform
            Y = y_transform.inverse(Y)
        if x_transform: # option to undo previous invertable X transform
            X = x_transform.inverse(X)
        if self.gt_type == 'binary':
            # convert logits to binary mask
            Y = np.argmax(Y, axis=0)
            
        X, Y = X.numpy(), Y.squeeze().numpy()
        shape = Y.shape
        dx = self.config['dx']*1e3 # [m] -> [mm]
        extent = [
            -dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2
        ]
        
        # option to plot the sample with the predicted mask and residual image
        if isinstance(Y_pred, torch.Tensor):
            if y_transform: # option to undo previous invertable Y transform
                Y_pred = y_transform.inverse(Y_pred)
            Y_pred = Y_pred.squeeze()
            Y_pred = Y_pred.detach().numpy()
            
            fig, ax = plt.subplots(2, 2, figsize=(8, 6))
            ax = ax.ravel()
            
            if self.gt_type == 'binary':
                Y_pred = np.argmax(Y_pred, axis=0)
                Y, Y_pred = Y.astype(bool), Y_pred.astype(bool)
                
                ax[2].imshow(
                    Y_pred, cmap='binary', extent=extent, origin='lower'
                )
                ax[2].set_title('Predicted mask')
                
                # plot a visualisation of the confusion matrix
                confusion_array = np.stack([
                    np.logical_not(Y_pred) * np.logical_not(Y), # TN
                    Y_pred * np.logical_not(Y), # FP
                    np.logical_not(Y_pred) * Y, # FN
                    Y_pred * Y], # TP
                    axis=0
                )
                labels = [  'TN'   ,  'FP'   ,  'FN'  ,   'TP'    ]
                colors = [ 'white' , 'black' ,'salmon','limegreen']
                confusion_array = np.sum(
                    confusion_array * np.arange(1,5).reshape(4, 1, 1), axis=0
                )
                residual_img = ax[3].imshow(
                    confusion_array,
                    cmap=plt.cm.colors.ListedColormap(colors),
                    vmin=1, 
                    vmax=4,
                    extent=extent, 
                    origin='lower'
                )
                fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[i],
                                    label=labels[i]) for i in range(4)])
                
                ax[3].set_title('Confusion matrix visualisation')
                
            elif self.gt_type == 'regression':
                pred_img = ax[2].imshow(
                    # [mols/m^3] = [10^3 M] -> [M]
                    Y_pred*1e-3, cmap='binary', extent=extent, origin='lower', 
                    vmin=0.0, vmax=np.max(Y)*1e-3
                )
                ax[2].set_title(r'predicted $c_{tot}$ (M)')
                plt.colorbar(pred_img, ax=ax[2])
                residual_img = ax[3].imshow(
                    # [mols/m^3] = [10^3 M] -> [M]
                    (Y_pred - Y)*1e-3, cmap='RdBu', extent=extent,
                    origin='lower', vmin=-np.max(Y)*1e-3, vmax=np.max(Y)*1e-3
                )
                plt.colorbar(residual_img, ax=ax[3])
                ax[3].set_title('predicted - ground truth (M)')
                
        else:
            fig, ax = plt.subplots(1, 2, figsize=(4, 6))
            ax = ax.ravel()

        
        if self.input_type == 'features':
            R2_img = ax[0].imshow(X[9,:,:], cmap='viridis', extent=extent, origin='lower')
            ax[0].set_title(r'$R^{2}$(770 nm)')
        else:
            R2_img = ax[0].imshow(X[-1,:,:], cmap='binary_r', extent=extent, origin='lower')
            ax[0].set_title('pulse 16 (770 nm)')
        plt.colorbar(R2_img, ax=ax[0])
        if self.gt_type == 'binary':
            gt_img = ax[1].imshow(Y, cmap='binary', extent=extent, origin='lower')
            ax[1].set_title('Ground truth')
            
        elif self.gt_type == 'regression':
            gt_img = ax[1].imshow(
                # [mols/m^3] = [10^3 M] -> [M]
                Y*1e-3, cmap='binary', extent=extent, origin='lower', vmin=0.0, 
                vmax=np.max(Y)*1e-3
            )
            ax[1].set_title(r'ground truth $c_{tot}$ (M)')
            plt.colorbar(gt_img, ax=ax[1])
        
        for a in ax:
            a.set_xlabel('x (mm)')
            a.set_ylabel('z (mm)')
        
        plt.tight_layout()
        
        if save_name:
            if '.png' not in save_name:
                save_name += '.png'
            plt.savefig(save_name, dpi=300)
        else:
            plt.show()
            
        return (fig, ax)
            
         