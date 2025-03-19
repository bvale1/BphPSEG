import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import sys

# add parent folder to path
sys.path.append(os.path.abspath(
    '/'.join(os.path.dirname(__file__).split('/')[:-1])
))
from preprocessing.dataloader import heatmap, load_sim

files = os.listdir('/mnt/f/cluster_MSOT_simulations/BphP_phantom_more_noise/')
for file in files:
    if '.out' in file or '.log' in file or '.error' in file:
        pass
    elif 'c143423.p31' in file:
        [data, sim_cfg] = load_sim(
            os.path.join(
                '/mnt/f/cluster_MSOT_simulations/BphP_phantom_more_noise/',
                file
            ), args=['p0_tr', 'noisy_p0_tr', 'noise_std6_p0_tr', 'ReBphP_PCM_c_tot']
        )
        break

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
x1, z1 = 195, 105
x2, z2 = 90, 50
x3, z3 = 180, 165 #175, #175
for i, noise_level in enumerate(['p0_tr', 'noisy_p0_tr', 'noise_std6_p0_tr']):
    # laser energy correction, divide by total energy delivered [Pa] -> [Pa J^-1]
    data[noise_level] *= 1/np.asarray(sim_cfg['LaserEnergy'])[:,:,:,np.newaxis,np.newaxis]
    # get reconstructed pressure at x1, z1, x2, z2, x3, z3 for lambda1 and lambda2
    p0_1_lambda1 = data[noise_level][0, 0, :, z1, x1]
    p0_1_lambda2 = data[noise_level][0, 1, :, z1, x1]
    p0_2_lambda1 = data[noise_level][0, 0, :, z2, x2]
    p0_2_lambda2 = data[noise_level][0, 1, :, z2, x2]
    p0_3_lambda1 = data[noise_level][0, 0, :, z3, x3]
    p0_3_lambda2 = data[noise_level][0, 1, :, z3, x3]
    # fit exponential decay with bassline to the data
    A1_lambda1, k1_lambda1, b1_lambda1 = 1.0, 1.0, 1.0
    A1_lambda2, k1_lambda2, b1_lambda2 = 1.0, 1.0, 1.0
    A2_lambda1, k2_lambda1, b2_lambda1 = 1.0, 1.0, 1.0
    A2_lambda2, k2_lambda2, b2_lambda2 = 1.0, 1.0, 1.0
    A3_lambda1, k3_lambda1, b3_lambda1 = 1.0, 1.0, 1.0
    A3_lambda2, k3_lambda2, b3_lambda2 = 1.0, 1.0, 1.0

    residuals = lambda x, y, n: y - x[0] * np.exp(-x[1] * n) - x[2]
    lm_scipy_fit = lambda x0, p0, n: least_squares(
        residuals, x0=x0, args=(p0, n), method='lm', ftol=1e-9, xtol=1e-9, gtol=1e-9
    ).x

    n = np.arange(len(p0_1_lambda1))
    n_continuous = np.linspace(0, len(p0_1_lambda1), 1000)
    (A1_lambda1, k1_lambda1, b1_lambda1) = lm_scipy_fit(
        [A1_lambda1, k1_lambda1, b1_lambda1], p0_1_lambda1, n
    )
    (A1_lambda2, k1_lambda2, b1_lambda2) = lm_scipy_fit(
        [A1_lambda2, k1_lambda2, b1_lambda2], p0_1_lambda2, n
    )
    (A2_lambda1, k2_lambda1, b2_lambda1) = lm_scipy_fit(
        [A2_lambda1, k2_lambda1, b2_lambda1], p0_2_lambda1, n
    )
    (A2_lambda2, k2_lambda2, b2_lambda2) = lm_scipy_fit(
        [A2_lambda2, k2_lambda2, b2_lambda2], p0_2_lambda2, n
    )
    (A3_lambda1, k3_lambda1, b3_lambda1) = lm_scipy_fit(
        [A3_lambda1, k3_lambda1, b3_lambda1], p0_3_lambda1, n
    )
    (A3_lambda2, k3_lambda2, b3_lambda2) = lm_scipy_fit(
        [A3_lambda2, k3_lambda2, b3_lambda2], p0_3_lambda2, n
    )

    ax[0, i].scatter(n, p0_1_lambda1, color='red')
    ax[0, i].plot(
        n_continuous, A1_lambda1 * np.exp(-k1_lambda1 * n_continuous) + b1_lambda1, color='red'
    )
    ax[0, i].scatter(n+16, p0_1_lambda2, color='red', marker='x')
    ax[0, i].plot(
        n_continuous+16,
        A1_lambda2 * np.exp(-k1_lambda2 * n_continuous) + b1_lambda2,
        color='red', linestyle='--'
    )
    ax[1, i].scatter(n, p0_2_lambda1, color='green')
    ax[1, i].plot(
        n_continuous, A2_lambda1 * np.exp(-k2_lambda1 * n_continuous) + b2_lambda1, color='green'
    )
    ax[1, i].scatter(n+16, p0_2_lambda2, color='green', marker='x')
    ax[1, i].plot(
        n_continuous+16, 
        A2_lambda2 * np.exp(-k2_lambda2 * n_continuous) + b2_lambda2, 
        color='green', linestyle='--'
    )
    ax[2, i].scatter(n, p0_3_lambda1, color='blue')
    ax[2, i].plot(
        n_continuous, A3_lambda1 * np.exp(-k3_lambda1 * n_continuous) + b3_lambda1, color='blue'
    )
    ax[2, i].scatter(n+16, p0_3_lambda2, color='blue', marker='x')
    ax[2, i].plot(
        n_continuous+16,
        A3_lambda2 * np.exp(-k3_lambda2 * n_continuous) + b3_lambda2,
        color='blue', linestyle='--'
    )

ax[0, 0].set_title('Dataset 1')
ax[0, 1].set_title('Dataset 2')
ax[0, 2].set_title('Dataset 3')
ax[0, 0].set_ylabel(r'Amplitude (Pa J${^-1}$)')
ax[1, 0].set_ylabel(r'Amplitude (Pa J${^-1}$)')
ax[2, 0].set_ylabel(r'Amplitude (Pa J${^-1}$)')
ax[2, 0].set_xlabel('pulse number')
ax[2, 1].set_xlabel('pulse number')
ax[2, 2].set_xlabel('pulse number')
for x in ax.flatten():
    x.grid(True)
    x.set_axisbelow(True)
fig.tight_layout()
fig.savefig('decay_curves.png')

# plot x, z positions on example images
images = np.array([data['p0_tr'][0, 0, 0],
                   data['noisy_p0_tr'][0, 0, 0],
                   data['noise_std6_p0_tr'][0, 0, 0],
                   data['p0_tr'][0, 0, -1],
                   data['noisy_p0_tr'][0, 0, -1], 
                   data['noise_std6_p0_tr'][0, 0, -1],
                   data['p0_tr'][0, 1, 0],
                   data['noisy_p0_tr'][0, 1, 0],
                   data['noise_std6_p0_tr'][0, 1, 0],
                   data['p0_tr'][0, 1, -1],
                   data['noisy_p0_tr'][0, 1, -1],
                   data['noise_std6_p0_tr'][0, 1, -1]])  

titles = [r'$\lambda_{1}$, n=1', r'$\lambda_{1}$, n=1', r'$\lambda_{1}$, n=1',
          r'$\lambda_{1}$, n=16', r'$\lambda_{1}$, n=16', r'$\lambda_{1}$, n=16',
          r'$\lambda_{2}$, n=1', r'$\lambda_{2}$, n=1', r'$\lambda_{2}$, n=1',
          r'$\lambda_{2}$, n=16', r'$\lambda_{2}$, n=16', r'$\lambda_{2}$, n=16']

dx = sim_cfg['dx']
plt.rcParams.update({'font.size': 12})
fig, ax, frames = heatmap(
    images,
    dx=dx,
    rowmax=3,
    labels=titles,
    sharescale=False,
    cbar_label=''#'PA signal amplitude (Pa J$^{-1}$)'
)
dx *= 1e3 # convert to mm
for i, image in enumerate(ax):
    image.plot((x1-128)*dx, (z1-128)*dx, 'rx')
    image.plot((x2-128)*dx, (z2-128)*dx, 'gx')
    image.plot((x3-128)*dx, (z3-128)*dx, 'bx')
fig.savefig('reconstructed_images_example.png')

fig, ax, frames = heatmap( # convert to mol m^-3 to M=Mol L^-1
    data['ReBphP_PCM_c_tot']*1e-3*1e7, dx=sim_cfg['dx'], cbar_label=r'$10^{-7}$ M'
)
ax[0].plot((x1-128)*dx, (z1-128)*dx, 'rx', markersize=20, markeredgewidth=4)
ax[0].plot((x2-128)*dx, (z2-128)*dx, 'gx', markersize=20, markeredgewidth=4)
ax[0].plot((x3-128)*dx, (z3-128)*dx, 'bx', markersize=20, markeredgewidth=4)
plt.savefig('total_protein_concentration.png')

fig, ax, frames = heatmap( # convert to mol m^-3 to M=Mol L^-1
    data['p0_tr'][0,0,0], dx=sim_cfg['dx'], cbar_label=r'Pa J$^{-1}$'
)
ax[0].plot((x1-128)*dx, (z1-128)*dx, 'rx', markersize=20, markeredgewidth=4)
ax[0].plot((x2-128)*dx, (z2-128)*dx, 'gx', markersize=20, markeredgewidth=4)
ax[0].plot((x3-128)*dx, (z3-128)*dx, 'bx', markersize=20, markeredgewidth=4)
plt.savefig('roi_reconstruction.png')

print(f'x1={(x1-128)*dx:.2f} mm, z1={(z1-128)*dx:.2f} mm')
print(f'x2={(x2-128)*dx:.2f} mm, z2={(z2-128)*dx:.2f} mm')
print(f'x3={(x3-128)*dx:.2f} mm, z3={(z3-128)*dx:.2f} mm')