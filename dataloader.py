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
        
        if nframes > 1:
            ax[frame].set(title='pulse '+str(frame))

    fig.subplots_adjust(right=0.8)
    #fig.tight_layout()
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(frames[0], cax=cbar_ax)
    
    fig.suptitle(title, fontsize='xx-large')
    
    return (fig, ax, frames)


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
    
    #path = 'home/wv00017/python_BphP_MSOT_sim/core/202307020_python_Clara_phantom_ReBphP_0p001'
    #path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\core\\202307020_python_Clara_phantom_ReBphP_0p001'
    #path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20230912_Clara_phantom_ReBphP_0p001'
    #path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20230912_Clara_phantom_reduced'
    #path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\20230915_Clara_phantom_1pulse_nonnegativity'
    #path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20230918_Clara_phantom_reduced_1itr'
    #path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20230918_Clara_phantom_reduced_8pulses_kernal_restart'
    path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20230926_Clara_phantom_itr1_fixed_mcx'

    args = ['Phi', 'p0', 'ReBphP_PCM_Pfr_c', 'ReBphP_PCM_Pr_c', 'ReBphP_PCM_c_tot']
    [data, cfg] = load_sim(path, args)
    
    path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20230928_random_cylinder_1itr_reduced'
    [data_reduced, cfg_reduced] = load_sim(path, args)
    
    #path = '\\\\wsl.localhost\\Ubuntu\\home\\wv00017\\python_BphP_MSOT_sim\\20230928_random_cylinder_1itr_reduced_doubleMCXy'
    #[data_reduced_doubleMCXy, cfg_reduced_doubleMCXy] = load_sim(path, args)
    
    
    heatmap(
        data['Phi'][0,0,0],#*circle_mask(data['Phi'][0,0,0], cfg['dx'], 0.01), 
        dx=cfg['dx'], 
        title=r'$\Phi(680$nm)   (J m$^{2}$)',
        vmin=0.0, 
        vmax=1000.0
    )
    heatmap(
        data_reduced['Phi'][0,0,0],#*circle_mask(data_reduced['Phi'][0,0,0], cfg_reduced['dx'], 0.01),
        dx=cfg_reduced['dx'],
        title=r'$\Phi(680$nm)   (J m$^{2}$)', 
        vmin=0.0, 
        vmax=1000.0
    )
    #heatmap(data_reduced_doubleMCXy['Phi'][0,0,0], dx=cfg_reduced_doubleMCXy['dx'], title=r'$\Phi(680$nm)   (J m$^{2}$)')
    
    
    #data['p0_3D'] = load_p0_3D(path)
    #heatmap_3D(data['p0_3D'][0,0,0], title='p0', dx=cfg['dx'])

