import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import json


def heatmap(
        img, 
        title='', 
        cmap='binary_r', 
        vmin=None, 
        vmax=None, 
        dx=0.0001, 
        rowmax=6,
        labels=None
    ):
    
    # use cmap = 'cool' for feature extraction
    # use cmap = 'binary_r' for raw data
    dx = dx * 1e3 # [m] -> [mm]
    
    # convert to numpy for plotting
    if type(img) == torch.Tensor:
        img = img.detach().numpy()
        
    shape = np.shape(img)
    mask = np.logical_not(np.isnan(img))
    if not vmin:
        vmin = np.min(img[mask])
    if not vmax:
        vmax = np.max(img[mask])
    
    extent = [-dx*shape[-2], dx*shape[-2], -dx*shape[-1], dx*shape[-1]]
    #extent=None
    
    if len(shape) == 2: # one pulse
        nframes = 1
        fig, ax = plt.subplots(nrows=1, ncols=nframes, figsize=(6,8))
        ax = np.array([ax])
        img = np.reshape(img, (1, shape[0], shape[1]))
        ax[0].set_xlabel('x (mm)')
        ax[0].set_ylabel('z (mm)')
        
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
        
    frames = []
                     
    
    for frame in range(nframes): 
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

    fig.subplots_adjust(right=0.8)
    #fig.tight_layout()
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(frames[0], cax=cbar_ax)
    
    fig.suptitle(title, fontsize='xx-large')
    
    return (fig, ax, frames)


def line_profile(x, data, labels=False, title='', xlabel='', ylabel=''):
    '''
    x : np.ndarray of shape (pixels) [mm]
    data : np.ndarray of shape(n, pixels)
    examples for data argument:
        time series z axis line profile
         -> data = data['arg'][cycle,wavelength,start:end,nx/2,:]
        time series x axis line profile
         -> data = sim.data['arg'][cycle,wavelength,start:end,:,nz/2]
        single frame x axis line profile
         -> data = sim.data['arg'][cycle,wavelength,pulse,:,nz/2]
    '''
    if type(data) == torch.Tensor:
        data = data.detach.numpy()
    if type(labels) != list:
        labels = [labels]
        
    fig, ax = plt.subplots(1, 1, figsize=(6,8))
    
    if len(data.shape) == 1:
        if labels:
            ax.plot(x, data, label=labels[0])
        else:
            ax.plot(x, data)
    elif len(data.shape) == 2:
        if labels:
            for i in range(data.shape[0]):
                ax.plot(x, data[i], label=labels[i])
        else:
            for i in range(data.shape[0]):
                ax.plot(x, data[i])
    else:
        print('error, data should be 1 or 2 dimensional')
                    
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
            print(f'loading {arg}')
            data[arg] = np.rot90(np.array(f.get(arg)), k=-1, axes=(-2,-1))
            
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
    print(X.shape)
    R = np.linalg.norm(np.array([X-centre[0], Y-centre[1]]), axis=0)
    print(R.shape)
    return R < radius
    
if __name__ == '__main__':

    args = ['Phi', 'p0', 'ReBphP_PCM_Pfr_c', 'ReBphP_PCM_Pr_c', 'ReBphP_PCM_c_tot', 'p0_recon']
    args = 'all'
    #path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20230929_cyclinder_phantom_QA_test'
    #path = 'E:/cluster_MSOT_simulations/20231002_simple_tomour_phantom.c133476.p0'
    #path = 'E:/cluster_MSOT_simulations/20231003_simple_tomour_phantom_mu_a1.c133488.p0'
    #path = 'E:/cluster_MSOT_simulations/20231003_simple_tomour_phantom_mu_a1.c133526.p0'
    #path = 'E:/cluster_MSOT_simulations/20231003_simple_tomour_phantom_mu_a1.c133531.p0'
    #path = 'E:/cluster_MSOT_simulations/20231003_simple_tomour_phantom_mu_a1.c133535.p0'
    path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20231011_bp_test'
    #path = 'E:/cluster_MSOT_simulations/20231010_simple_tomour_phantom_mu_a1.c133970.p0'
    #path = 'E:/cluster_MSOT_simulations/20231011_simple_tomour_phantom_mu_a1_itralpha1.c134019.p0'
    path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20231011_interp512_nearest_test'
    path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20231011_interp1024_linear_test'
    path = 'E:/cluster_MSOT_simulations/20231011_ppw2_transducer_array_model.c134129.p0'
    [data, cfg] = load_sim(path, args)
    [nx, nz] = data['p0'][0,0,0].shape
    
    heatmap(
        data['Phi'][0,0,0],
        dx=cfg['dx'], 
        title=r'$\Phi(680$nm)   (J m$^{2}$)',
        vmin=0.0, 
        vmax=1000.0
    )
    labels = [r'$p_{0}$', r'$p_{0,tr,1}$', '$p_{0,tr,2}$', r'$p_{0,tr,3}$',
              r'$p_{0,tr,4}$', r'$p_{0,tr,5}$', r'$p_{0,tr,6}$',
              r'$p_{0,tr,7}$',r'$p_{0,tr,8}$', r'$p_{0,tr,9}$', r'$p_{0,tr,10}$']#, r'$p_{0,bp}$'
    
    lines = np.concatenate(
        (
            np.diag(np.fliplr(data['p0'][0,0,0,:,:]))[np.newaxis], 
            #np.diag(np.fliplr(data['p0_bp'][0,0,0,:,:]))[np.newaxis],
            np.diag(np.fliplr(data['p0_tr_1']))[np.newaxis],
            np.diag(np.fliplr(data['p0_tr_2']))[np.newaxis],
            np.diag(np.fliplr(data['p0_tr_3']))[np.newaxis],
            np.diag(np.fliplr(data['p0_tr_4']))[np.newaxis],
            np.diag(np.fliplr(data['p0_tr_5']))[np.newaxis],
            np.diag(np.fliplr(data['p0_tr_6']))[np.newaxis],
            np.diag(np.fliplr(data['p0_tr_7']))[np.newaxis]
        ), axis=0
    )
    for i, arg in enumerate(['p0_tr_1', 'p0_tr_2', 'p0_tr_3', 'p0_tr_4', 'p0_tr_5',
                             'p0_tr_6', 'p0_tr_7']):

        RMSE = round(np.sqrt(np.mean((data['p0'][0,0,0] - data[arg])**2)), 2)
        print(f'{arg} RMSE = {RMSE}')
        labels[i+1] += f' RMSE = {RMSE}'
    
    
    #heatmap(data['p0'][0,0,0], dx=cfg['dx'], title=r'$p_{0}(680$nm)   (Pa)')
    #heatmap(data['p0_tr'][0,0,0], dx=cfg['dx'], title=r'time reversal $p_{0,recon}(680$nm)   (Pa)')
    #heatmap(data['background_mua_mus'][0,0], dx=cfg['dx'], title=r'$mu_{a}(680$nm)   (m$^{-1}$)')
    
    #heatmap(data['p0_bp'][0,0,0], dx=cfg['dx'], title=r'backprojection $p_{0,recon}(680$nm)   (Pa)')
    recons = np.concatenate(
        (
            data['p0'][0,0,:],
            #data['p0_bp'][0,0,:],
            data['p0_tr_1'][np.newaxis],
            data['p0_tr_2'][np.newaxis],
            data['p0_tr_3'][np.newaxis],
            data['p0_tr_4'][np.newaxis],
            data['p0_tr_5'][np.newaxis],
            data['p0_tr_6'][np.newaxis],
            data['p0_tr_7'][np.newaxis],
            #data['p0_tr_8'][np.newaxis],
            #data['p0_tr_9'][np.newaxis],
            #data['p0_tr_10'][np.newaxis]
        ), axis=0
    )
    heatmap(recons, dx=cfg['dx'], title=r'$p_{0}(680$nm)   (Pa)', rowmax=4, labels=labels)


    
    RMSE = np.sqrt(np.mean((data['p0'][0,0,0] - data['p0_tr'])**2))
    print(f'p0_tr RMSE = {RMSE}')
        
    line_profile(
        np.arange(cfg['crop_size']) * cfg['dx'] * np.sqrt(2) * 1e3,
        lines, 
        labels=labels, 
        title='diagonal line profile, top left to bottom right', 
        xlabel='mm',
        ylabel='Pa'
    )
    

