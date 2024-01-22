import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import json
from preprocessing.feature_extractor import feature_extractor
import logging

def square_centre_crop(image : np.ndarray, size : int) -> np.ndarray:
    width, height = image.shape[-2:]
    if width < size or height < size:
        print('Image is smaller than crop size, returning original image')
        return image
    else:
        x = (width - size) // 2
        y = (height - size) // 2
        image = image[..., x:x+size, y:y+size]
        return image


def heatmap(img, 
            title='', 
            cmap='binary_r', 
            vmin=None, 
            vmax=None, 
            dx=0.0001, 
            rowmax=6,
            labels=None,
            sharescale=True,
            cbar_label=None):
    # TODO: heatmap should use a list to plot images of different resolution
    logging.basicConfig(level=logging.INFO)    
    # use cmap = 'cool' for feature extraction
    # use cmap = 'binary_r' for raw data
    dx = dx * 1e3 # [m] -> [mm]
    
    frames = []
    
    # convert to numpy for plotting
    if type(img) == torch.Tensor:
        img = img.detach().numpy()
        
    shape = np.shape(img)
    if sharescale or len(shape) == 2:
        mask = np.logical_not(np.isnan(img))
        if not vmin:
            vmin = np.min(img[mask])
        if not vmax:
            vmax = np.max(img[mask])
    
    extent = [-dx*shape[-2]/2, dx*shape[-2]/2, -dx*shape[-1]/2, dx*shape[-1]/2]
    
    if len(shape) == 2: # one pulse
        nframes = 1
        fig, ax = plt.subplots(nrows=1, ncols=nframes, figsize=(6,8))
        ax = np.array([ax])
        ax[0].set_xlabel('x (mm)')
        ax[0].set_ylabel('z (mm)')
        frames.append(ax[0].imshow(
            img,
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax,
            extent=extent,
            origin='lower'
        ))
        
    else: # multiple pulses
        nframes = shape[0]
        nrows = int(np.ceil(nframes/rowmax))
        rowmax = nframes if nframes < rowmax else rowmax
        fig, ax = plt.subplots(nrows=nrows, ncols=rowmax, figsize=(16, 12))
        ax = np.asarray(ax)
        if len(np.shape(ax)) == 1:
            ax = ax.reshape(1, rowmax)
        for row in range(nrows):
            ax[row, 0].set_ylabel('z (mm)')
        for col in range(rowmax):
            ax[-1, col].set_xlabel('x (mm)')
        ax = ax.ravel()
        
        for frame in range(nframes): 
            if not sharescale:
                mask = np.logical_not(np.isnan(img[frame]))
                vmin = np.min(img[frame][mask])
                vmax = np.max(img[frame][mask])
            frames.append(ax[frame].imshow(
                img[frame],
                cmap=cmap, 
                vmin=vmin, 
                vmax=vmax,
                extent=extent,
                origin='lower'
            ))
            ax[frame].set_xlabel('x (mm)')
            if labels:
                ax[frame].set(title=labels[frame])
            elif nframes > 1:
                ax[frame].set(title='pulse '+str(frame))
            if not sharescale:
                cbar = plt.colorbar(frames[frame], ax=ax[frame])
                if cbar_label:
                    cbar.set_label=cbar_label

    fig.subplots_adjust(right=0.8)
    
    if sharescale:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(frames[0], cax=cbar_ax)
        if cbar_label:
            cbar.set_label=cbar_label
    else:
        fig.tight_layout()
            
    fig.suptitle(title, fontsize='xx-large')
    
    return (fig, ax, frames)


def line_profile(x, data, labels=False, title='', xlabel='', ylabel=''):
    '''
    x : list of np.ndarray or torch.Tensor of shape (n, pixels) [units of mm]
    data : list of np.ndarray or torch.Tensor of shape(n, pixels)
    examples for data argument:
        time series z axis line profile
         -> data = data['arg'][cycle,wavelength,start:end,nx/2,:]
        time series x axis line profile
         -> data = sim.data['arg'][cycle,wavelength,start:end,:,nz/2]
        single frame x axis line profile
         -> data = sim.data['arg'][cycle,wavelength,pulse,:,nz/2]
    '''
    logging.basicConfig(level=logging.INFO)
    
    if type(data) == torch.Tensor:
        data = data.detach.numpy()
    if type(labels) != list:
        labels = [labels]
    if type(x) == torch.Tensor:
        x = x.detach().numpy()
    if type(x) != list and len(x.shape) == 1:
        x = [x]
    if type(labels) != list:
        labels = [labels]
        
    fig, ax = plt.subplots(1, 1, figsize=(6,8))
    
    if labels:
        for i in range(len(data)):
            ax.plot(x[i], data[i], label=labels[i])
    else:
        for i in range(len(data)):
            ax.plot(x[i], data[i])
            
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)
    
    fig.suptitle(title)
    
    return (fig, ax)


def percent_of_array_is_finite(arr):
    if type(arr) == torch.Tensor:
        return 100 * torch.sum(torch.isfinite(arr)) / torch.prod(arr.shape)
    elif type(arr) == np.ndarray:
        return 100 * np.sum(np.isfinite(arr)) / np.prod(arr.shape)

def percent_of_array_is_zero(arr):
    if type(arr) == torch.Tensor:
        return 100 * torch.sum(arr==0.0) / torch.prod(arr.shape)
    elif type(arr) == np.ndarray:
        return 100 * np.sum(arr==0.0) / np.prod(arr.shape)

def visualise_fit(A, k, b, p0_n, R_sqr, x, z,
                  title=False, fig=None, ax=None, label=None):
    n = np.arange(p0_n.shape[0], dtype=np.float32)
    
    p0_n = p0_n[:,z,x]
    
    if type(p0_n) == torch.Tensor:
        p0_n = p0_n.numpy()
    if type(A) == torch.Tensor:
        A = A[z,x].numpy()
    if type(k) == torch.Tensor:
        k = k[z,x].numpy()
    if type(b) == torch.Tensor:
        b = b[z,x].numpy()
    if type(R_sqr) == torch.Tensor:
        R_sqr = R_sqr[z,x].numpy()
    
    
    print('visualise fit:')
    print('p0_n ', p0_n)
    print('A ', A)
    print('k ', k)
    print('b ', b)
    
    if not fig or not ax:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
    
    if label:
        ax.scatter(n+1, p0_n, label=r'{label} $p_{0}(n)$'.format(label=label))
        n = np.linspace(n[0], n[-1], num=1000)
        ax.plot(n+1, A * np.exp(-k * n) + b, label=label+' fit')
    else:
        ax.scatter(n+1, p0_n, label=r'$p_{0}(n)$')
        n = np.linspace(n[0], n[-1], num=1000)
        ax.plot(n+1, A * np.exp(-k * n) + b, label=r'fit')
        
    ax.legend()
    ax.set_xlabel('pulse n')
    ax.set_ylabel(r'$p_{0,recon}(n)$ (J$^{-1}$ m$^{-3})$')
    ax.grid(True)
    ax.set_axisbelow(True)
    
    if title:
        fig.suptitle(title)
    ax.set(title=r'$R^{sqr}={R_sqr}, x={x}, z={z}, k={k}$'.format(
        sqr=2, R_sqr=R_sqr, x=x, z=z, k=k)
    )
    
    return fig, ax
    

def heatmap_3D(
        arr, 
        title='', 
        color='red', 
        vmin=None, 
        vmax=None,
        dx=0.0001
    ):
    # TODO: make this work
    dx = dx * 1e3 # [m] -> [mm]
    
    # arr is 3 dimensional [x, y, z]
    # convert to numpy for plotting
    if type(arr) == torch.Tensor:
        arr = arr.detach().numpy()
    shape = arr.shape
    [x, y, z] = np.meshgrid(
        np.linspace(-dx*shape[1]/2, dx*shape[1]/2, shape[1]),
        np.linspace(-dx*shape[0]/2, dx*shape[0]/2, shape[0]),
        np.linspace(-dx*shape[2]/2, dx*shape[2]/2, shape[2])
    )
    # flat view of all arrays
    x = x.ravel(); y = y.ravel(); z = z.ravel(); arr = arr.ravel()
    # rescale arr so min=0, max=1
    if not vmin:
        vmin = np.min(arr)
    if not vmax:
        vmax = np.max(arr)
    arr = (arr - vmin) / vmax
    #print(f'max: {vmax}, min: {vmin}')
    
    # delete zeros
    zero_mask = (arr <= 0.0)
    arr = np.delete(arr, zero_mask); x = np.delete(x, zero_mask)
    y = np.delete(y, zero_mask); z = np.delete(z, zero_mask)
    arr[arr > 1.0] = 1.0
    fig, ax = plt.subplots(
        1, 1, figsize=(5, 5), subplot_kw={"projection": "3d"}
    )
    # set axis limits to a cube for 1:1:1 aspect ratio
    axmax = dx*(np.max(shape))/2; axmin = - axmax
    ax.set(xlim=[axmin, axmax], ylim=[axmin, axmax], zlim=[axmin, axmax])
    ax.scatter(x, y, z, color=color, alpha=arr, s=2)
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_zlabel('z (mm)')
    fig.suptitle(title, fontsize='xx-large')
    return (fig, ax)
    

def load_sim(path : str, args='all') -> list:
    data = {}
    with h5py.File(path+'/data.h5', 'r') as f:
        if args == 'all':
            args = f.keys()
        for arg in args:
            # include 90 deg clockwise rotation
            if arg != 'sensor_data':
                data[arg] = np.rot90(np.array(f.get(arg)), k=-1, axes=(-2,-1)).copy()
            else:
                data[arg] = np.array(f.get(arg)).copy()
            #print(percent_of_array_is_zero(data[arg]))
            #print(f'loading {arg}, shape {data[arg].shape}')
            #data[arg] = np.array(f.get(arg))
            
    with open(path+'/config.json', 'r') as f:
        cfg = json.load(f)
        
    return [data, cfg]

def load_p0_3D(path):
    with h5py.File(path+'/temp.h5', 'r') as f:
        p0_3D = np.array(f.get('p0_3D'))
    return p0_3D

    
def define_ReBphP_PCM(phantoms_path, wavelengths_interp: (list, np.ndarray)) -> dict:
    # (m^2 mol^-1) = (mm^-1 M^-1) = (mm^-1 mol^-1 dm^3) = (mm^-1 mol^-1 L^3)
    wavelengths_interp = np.asarray(wavelengths_interp) * 1e9 # [m] -> [nm]
    # ignore first line, load both columns into numpy array
    with open(phantoms_path+'/Chromophores/epsilon_a_ReBphP_PCM_Pr.txt', 'r') as f:
        data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
    wavelengths_Pr = data[:,0] # [nm]
    epsilon_a_Pr = data[:,1] * 1e4 # [1e5 M^-1 cm^-1] -> [M^-1 mm^-1]
    # sort to wavelength descending order
    sort_index = wavelengths_Pr.argsort()
    wavelengths_Pr = wavelengths_Pr[sort_index[::-1]]
    epsilon_a_Pr = epsilon_a_Pr[sort_index[::-1]]
    
    with open(phantoms_path+'/Chromophores/epsilon_a_ReBphP_PCM_Pfr.txt', 'r') as f:
        data = np.genfromtxt(f, skip_header=1, dtype=np.float32, delimiter=', ')
    wavelengths_Pfr = data[:,0] # [nm]
    epsilon_a_Pfr = data[:,1] * 1e4 # [1e5 M^-1 cm^-1] -> [M^-1 mm^-1]
    # sort to wavelength descending order
    sort_index = wavelengths_Pfr.argsort()
    wavelengths_Pfr = wavelengths_Pfr[sort_index[::-1]]
    epsilon_a_Pfr = epsilon_a_Pfr[sort_index[::-1]]
    
        
    # properties of the bacterial phytochrome
    ReBphP_PCM = {
        'Pr' : { # Red absorbing form
            'epsilon_a': np.interp(
                wavelengths_interp, wavelengths_Pr, epsilon_a_Pr
            ).tolist(), # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
            'eta' : [0.03, 0.0] # photoisomerisation quantum yield (dimensionless)
            },
        'Pfr' : { # Far-red absorbing form
            'epsilon_a': np.interp(
                wavelengths_interp, wavelengths_Pfr, epsilon_a_Pfr
            ).tolist(), # molar absorption coefficient [M^-1 cm^-1]=[m^2 mol^-1]
            'eta' : [0.0, 0.005] # photoisomerisation quantum yield (dimensionless)
        }   
    }
    return ReBphP_PCM
    
def circle_mask(arr, dx, radius):
    centre = np.asarray(arr.shape) * dx / 2
    [X, Y] = np.meshgrid(np.arange(arr.shape[0])*dx, np.arange(arr.shape[1])*dx)
    R = np.linalg.norm(np.array([X-centre[0], Y-centre[1]]), axis=0)
    return R < radius
    
if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # use this to compare features for simulations
    path = 'E:/cluster_MSOT_simulations/BphP_phantom/20231123_BphP_phantom.c139519.p0'
    #path = '\\\\wsl$\\Ubuntu-22.04\\home\\wv00017\\python_BphP_MSOT_sim\\20231123_Clara_phantom_eta0p007_eta0p0018'
    
    dataset_cfg = {
        'dataset_name' : '20231204_homogeneous_cylinders',
        'BphP_SVM_RF_XGB_git_hash' : None, # TODO: get git hash automatically
        'feature_names' : [
            'A_680nm', 'k_680nm', 'b_680nm', 'R_sqr_680nm', 'diff_680nm', 'range_680nm',
            'A_770nm', 'k_770nm', 'b_770nm', 'R_sqr_770nm', 'diff_770nm', 'range_770nm',
            'radial_dist'
        ]
    }
    
    labels = [r'$A(680$nm$)$', r'$k(680$nm$)$', r'$b(680$nm$)$', r'$R^{2}(680$nm$)$', r'$diff(680$nm$)$', r'$range(680$nm$)$',
              r'$A(770$nm$)$', r'$k(770$nm$)$', r'$b(770$nm$)$', r'$R^{2}(770$nm$)$', r'$diff(770$nm$)$', r'$range(770$nm$)$',
              'r (mm)']
    
    [data, cfg] = load_sim(path, args='all')
    
    data['p0_tr'] *= 1/np.asarray(cfg['LaserEnergy'])[:,:,:,np.newaxis,np.newaxis]
    
    fe = feature_extractor(data['p0_tr'][0,0], mask=data['bg_mask'])
    fe.NLS_scipy(display_progress=True)
    fe.threshold_features()
    fe.filter_features()
    fe.R_squared()
    fe.differetial_image()
    fe.range_image()
    
    features_680nm, keys_680nm = fe.get_features(asTensor=True)
    
    fe = feature_extractor(data['p0_tr'][0,1], mask=data['bg_mask'])
    fe.NLS_scipy(display_progress=True)
    fe.threshold_features()
    fe.filter_features()
    fe.R_squared()
    fe.differetial_image()
    fe.range_image()
    fe.radial_distance(cfg['dx'])
    
    features_770nm, keys_770nm = fe.get_features(asTensor=True)
    
    features = torch.cat([features_680nm, features_770nm], dim=0)
    
    # uncomment to plot features
    
    heatmap(
        features, 
        labels=labels,
        dx=cfg['dx'],
        sharescale=False,
        cmap='cool',
        rowmax=4
    )
    heatmap(
        data['p0_tr'][0,1][0::7],
        dx=cfg['dx'],
        labels=['pulse 1', 'pulse 7', 'pulse 14'],
        title='Pressure reconstructions 770nm (Pa J$^{-1}$)',
        sharescale=True
    )
    heatmap(
        data['ReBphP_PCM_c_tot']*1e5, # [1e3 M] -> [1e-2 M]
        dx=cfg['dx'],
        title='protein concentration ($10^{-2}$M)'
    )
    heatmap(
        data['background_mua_mus'][:,0],
        title=r'absorption coefficient $mu_{a}$ (m$^{-1}$)',
        labels=['680nm', '770nm'],
        sharescale=True
    )
    heatmap(
        data['background_mua_mus'][:,1],
        title=r'scattering coefficient $mu_{s}$ (m$^{-1}$)',
        labels=['680nm', '770nm'],
        sharescale=True
    )
         
    
    visualise_fit(
        features_770nm[0], 
        features_770nm[1], 
        features_770nm[2],
        data['p0_tr'][0,1],
        features_770nm[3],
        190, 
        90
    )
    visualise_fit(
        features_770nm[0], 
        features_770nm[1], 
        features_770nm[2],
        data['p0_tr'][0,1],
        features_770nm[3],
        90, 
        90
    )
    visualise_fit(
        features_770nm[0], 
        features_770nm[1], 
        features_770nm[2],
        data['p0_tr'][0,1],
        features_770nm[3],
        175, 
        185
    )