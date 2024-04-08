import numpy as np
import torch, h5py, json, os, logging
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from preprocessing.dataloader import load_sim


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
        else:
            self.get_X = lambda f, sample:  torch.from_numpy(f[sample]['features'][()])
                
            
    def __len__(self) -> int:
        return len(self.samples)
    
    
    def __getitem__(self, index : int) -> tuple:
        sample = self.samples[index]
        with h5py.File(self.h5_file, 'r') as f:
            X = self.get_X(f, sample)
            Y = self.get_Y(f, sample)
        
        if self.x_transform:
            X = self.x_transform(X)
        if self.y_transform:
            Y = self.y_transform(Y)
            
        return (X, Y)


    def plot_sample(self, index, Y_hat=None, save_name=None, y_transform=None):
        X, Y = self.__getitem__(index)
        if y_transform: # option to undo previous invertable Y transform
            Y = y_transform.inverse(Y)
        if self.gt_type == 'binary':
            Y = np.argmax(Y, axis=0)
            
        X, Y = X.numpy(), Y.numpy()
        shape = Y.shape
        dx = self.config['dx']
        extent = [
            -dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2
        ]
        
        # option to plot the sample with the predicted mask and residual image
        if isinstance(Y_hat, torch.Tensor):
            if y_transform: # option to undo previous invertable Y transform
                Y_hat = y_transform.inverse(Y_hat)
            Y_hat = Y_hat.detach().numpy()
            Y_hat = Y_hat.squeeze()
            fig, ax = plt.subplots(2, 2, figsize=(8, 6))
            ax = ax.ravel()
            
            if self.gt_type == 'binary':
                Y_hat = np.argmax(Y_hat, axis=0)
                Y, Y_hat = Y.astype(bool), Y_hat.astype(bool)
                
                ax[2].imshow(
                    Y_hat, cmap='binary', extent=extent, origin='lower'
                )
                ax[2].set_title('Predicted Mask')
                
                # plot a visualisation of the confusion matrix
                confusion_array = np.stack([
                    np.logical_not(Y_hat) * np.logical_not(Y), # TN
                    Y_hat * np.logical_not(Y), # FP
                    np.logical_not(Y_hat) * Y, # FN
                    Y_hat * Y], # TP
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
                
                ax[3].set_title('Confusion Map')
                
            elif self.gt_type == 'regression':
                pred_img = ax[2].imshow(
                    # [mols/m^3] = [10^3 M] -> [M]
                    Y_hat*1e3, cmap='binary', extent=extent, origin='lower', vmin=0, vmax=0.1
                )
                ax[2].set_title(r'predicted $c_{tot}$ (M)')
                plt.colorbar(pred_img, ax=ax[2])
                residual_img = ax[3].imshow(
                    # [mols/m^3] = [10^3 M] -> [M]
                    np.abs(Y_hat - Y)*1e3, cmap='OrRd', extent=extent,
                    origin='lower'#, vmin=-1e-6, vmax=1e-6
                )
                plt.colorbar(residual_img, ax=ax[3])
                ax[3].set_title('Absolute Error (M)')
                
        else:
            fig, ax = plt.subplots(1, 2, figsize=(4, 6))
            ax = ax.ravel()

        R2_img = ax[0].imshow(X[9,:,:], cmap='cool', extent=extent, origin='lower')
        ax[0].set_title(r'$R^{2}$(680nm)')
        plt.colorbar(R2_img, ax=ax[0])
        if self.gt_type == 'binary':
            gt_img = ax[1].imshow(Y, cmap='binary', extent=extent, origin='lower')
            ax[1].set_title('ground truth mask')
            
        elif self.gt_type == 'regression':
            gt_img = ax[1].imshow(
                # [mols/m^3] = [10^3 M] -> [M]
                Y*1e3, cmap='binary', extent=extent, origin='lower', vmin=0, vmax=0.1
            )
            ax[1].set_title(r'ground truth $c_{tot}$ (M)')
            plt.colorbar(gt_img, ax=ax[1])
        
        for a in ax:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
        
        if save_name:
            if '.png' not in save_name:
                save_name += '.png'
            plt.savefig(save_name, dpi=300)
        else:
            plt.show()
            
         
class BphP_integrated_MSOT_Dataset(Dataset):
    # use to train integrated models with both binary semantic segmentation 
    # and regression outputs. Highly experimental and not recommended.
    def __init__(self,
                 dataset_path : str,
                 input_type : str,
                 x_transform=None, 
                 binary_y_transform=None,
                 regression_y_transform=None
                ):
        self.dataset_path = dataset_path
        self.h5_file = os.path.join(dataset_path, 'dataset.h5')
        if input_type not in ['images', 'features']:
            raise ValueError("input_type must be either 'images' or 'features'")
        
        self.x_transform = x_transform
        self.binary_y_transform = binary_y_transform
        self.regression_y_transform = regression_y_transform
        
        with h5py.File(self.h5_file, 'r') as f:
            self.samples = list(f.keys())
            
        # load config file
        with open(os.path.join(os.path.dirname(self.dataset_path), 'config.json'), 'r') as f:
            self.config = json.load(f)
            
        if input_type == 'images':
            self.get_X = lambda f, sample:  torch.flatten(
                torch.from_numpy(f[sample]['images'][0,:]), start_dim=0, end_dim=1
            )
        else:
            self.get_X = lambda f, sample:  torch.from_numpy(f[sample]['features'][()])
                
            
    def __len__(self) -> int:
        return len(self.samples)
    
    
    def __getitem__(self, index : int) -> tuple:
        sample = self.samples[index]
        with h5py.File(self.h5_file, 'r') as f:
            X = self.get_X(f, sample)
            Y = {'binary' : torch.from_numpy(f[sample]['c_mask'][()]),
                'regression' : torch.from_numpy(f[sample]['c_tot'][()])}
        
        if self.x_transform:
            X = self.x_transform(X)
        if self.binary_y_transform:    
            Y['binary'] = self.binary_y_transform(Y['binary'])
        if self.regression_y_transform:
            Y['regression'] = self.regression_y_transform(Y['regression'])
            
        return (X, Y)
    
            

class BphP_MSOT_raw_image_Dataset(Dataset):
    # this is an old version of the dataset class that loads raw data from the
    # simulation output files, it is recommended to load from the processed h5 file
    # and use the BphP_MSOT_Dataset class instead
    def __init__(self, root_dir, gt_type, n_images=32, x_transform=None, y_transform=None):
        # make sure the all files in root_dir are valid samples to avoid errors
        self.root_dir = root_dir
        if gt_type not in ['binary', 'regression']:
            # use binary classification or value regression
            raise ValueError("gt_type must be either 'binary' or 'regression'")
        self.gt_type = gt_type
        self.n_images = n_images # number of images to load, also the number of channels
        self.x_transform = x_transform
        self.y_transform = y_transform
            
        files = os.listdir(root_dir)
        self.samples = []
        for file in files:
            if '.out' in file or '.log' in file or '.error' in file:
                pass
            else:
                self.samples.append(file)
        logging.info(f'raw data located in {root_dir}')
        logging.info(f'{len(self.samples)} samples found')
        
            
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index) -> tuple:
        self.samples[index]
        
        [data, sim_cfg] = load_sim(
            os.path.join(self.root_dir, self.samples[index]),
            args=['p0_tr', 'ReBphP_PCM_c_tot']
        ) 
        data['p0_tr'] *= 1/np.asarray(sim_cfg['LaserEnergy'])[:,:,:,np.newaxis,np.newaxis]
        
        #X = torch.from_numpy(data['p0_tr'][0,1]) # 1st cycle, 2nd wavelength
        
        
        
        X = torch.flatten(torch.from_numpy(data['p0_tr'][0,:]), start_dim=0, end_dim=1)[:self.n_images]
        if self.gt_type == 'binary':
            Y = torch.from_numpy(data['ReBphP_PCM_c_tot'] > 0.0)
        elif self.gt_type == 'regression':
            Y = torch.from_numpy(data['ReBphP_PCM_c_tot'])
        
        if self.x_transform:
            X = self.x_transform(X)
        if self.y_transform:
            Y = self.y_transform(Y)
            
        return (X, Y)
    
    
    def get_config(self, index):
        self.samples[index]
        [_, sim_cfg] = load_sim(
            os.path.join(self.root_dir, self.samples[index]),
            args=['ReBphP_PCM_c_tot']
        )
        self.config = sim_cfg
        return sim_cfg
    
    
    def plot_sample(self, index, Y_hat=None, save_name=None, y_transform=None):
        X, Y = self.__getitem__(index)
        if y_transform: # option to undo previous invertable Y transform
            Y = y_transform.inverse(Y)
        if self.gt_type == 'binary':
            Y = np.argmax(Y, axis=0)
            
        X, Y = X.numpy(), Y.numpy()
        shape = Y.shape
        dx = self.config['dx']
        extent = [
            -dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2
        ]
        
        # option to plot the sample with the predicted mask and residual image
        if isinstance(Y_hat, torch.Tensor):
            if y_transform: # option to undo previous invertable Y transform
                Y_hat = y_transform.inverse(Y_hat)
            Y_hat = Y_hat.detach().numpy()
            Y_hat = Y_hat.squeeze()
            fig, ax = plt.subplots(2, 2, figsize=(8, 6))
            ax = ax.ravel()
            
            if self.gt_type == 'binary':
                Y_hat = np.argmax(Y_hat, axis=0)
                Y, Y_hat = Y.astype(bool), Y_hat.astype(bool)
                
                ax[2].imshow(
                    Y_hat, cmap='binary', extent=extent, origin='lower'
                )
                ax[2].set_title('Predicted Mask')
                
                # plot a visualisation of the confusion matrix
                confusion_array = np.stack([
                    np.logical_not(Y_hat) * np.logical_not(Y), # TN
                    Y_hat * np.logical_not(Y), # FP
                    np.logical_not(Y_hat) * Y, # FN
                    Y_hat * Y], # TP
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
                
                ax[3].set_title('Confusion Map')
                
            elif self.gt_type == 'regression':
                pred_img = ax[2].imshow(
                    # [mols/m^3] = [10^3 M] -> [M]
                    Y_hat*1e3, cmap='binary', extent=extent, origin='lower', vmin=0, vmax=0.1
                )
                ax[2].set_title(r'predicted $c_{tot}$ (M)')
                plt.colorbar(pred_img, ax=ax[2])
                residual_img = ax[3].imshow(
                    # [mols/m^3] = [10^3 M] -> [M]
                    np.abs(Y_hat - Y)*1e3, cmap='OrRd', extent=extent,
                    origin='lower'#, vmin=-1e-6, vmax=1e-6
                )
                plt.colorbar(residual_img, ax=ax[3])
                ax[3].set_title('Absolute Error (M)')
                
        else:
            fig, ax = plt.subplots(1, 2, figsize=(4, 6))
            ax = ax.ravel()

        R2_img = ax[0].imshow(X[0,:,:], cmap='cool', extent=extent, origin='lower')
        ax[0].set_title(r'Pulse 1 (770nm)')
        plt.colorbar(R2_img, ax=ax[0])
        if self.gt_type == 'binary':
            gt_img = ax[1].imshow(Y, cmap='binary', extent=extent, origin='lower')
            ax[1].set_title('ground truth mask')
            
        elif self.gt_type == 'regression':
            gt_img = ax[1].imshow(
                # [mols/m^3] = [10^3 M] -> [M]
                Y*1e3, cmap='binary', extent=extent, origin='lower', vmin=0, vmax=0.1
            )
            ax[1].set_title(r'ground truth $c_{tot}$ (M)')
            plt.colorbar(gt_img, ax=ax[1])
        
        for a in ax:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
        
        if save_name:
            if '.png' not in save_name:
                save_name += '.png'
            plt.savefig(save_name, dpi=300)
        else:
            plt.show()
            
      