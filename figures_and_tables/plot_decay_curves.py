import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import sys

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "cm"

# add parent folder to path
sys.path.append(os.path.abspath(
    '/'.join(os.path.dirname(__file__).split('/')[:-1])
))
from preprocessing.dataloader import heatmap, load_sim

sim_root = "/media/billy/Seagate Hub/cluster_MSOT_simulations/BphP_phantom_more_noise/"
files = os.listdir(sim_root)
for file in files:
    if '.out' in file or '.log' in file or '.error' in file:
        pass
    elif 'c143423.p31' in file:
        [data, sim_cfg] = load_sim(
            os.path.join(
                sim_root,
                file
            ), args=['p0_tr', 'noisy_p0_tr', 'noise_std6_p0_tr', 'ReBphP_PCM_c_tot']
        )
        break
    
dx = sim_cfg['dx'] * 1e3   # convert from m to mm

plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(3, 3, figsize=(18, 18), gridspec_kw={"wspace": 0.3, "hspace": 0.2})
x1, z1 = 195, 105
x2, z2 = 90, 50
#x3, z3 = 180, 165
x3, z3 = 175, 175
#tick_positions = np.arange(1, 32, 2)
#tick_labels = np.tile(np.arange(0, 18, 2), 2)
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

    n = np.arange(len(p0_1_lambda1)) + 1
    n_continuous = np.linspace(0, len(p0_1_lambda1), 1000) + 1
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
    # x1, z1
    ax[0, i].scatter(
        n, p0_1_lambda1, color='red', label=r'$\lambda_{1}$ data'
        )
    ax[0, i].plot(
        n_continuous, A1_lambda1 * np.exp(-k1_lambda1 * n_continuous) + b1_lambda1, 
        color='red', label=r'$\lambda_{1}$ fit'
    )
    ax[0, i].scatter(
        n, p0_1_lambda2, color='red', marker='x', label=r'$\lambda_{2}$ data'
    )
    ax[0, i].plot(
        n_continuous,
        A1_lambda2 * np.exp(-k1_lambda2 * n_continuous) + b1_lambda2,
        color='red', linestyle='--', label=r'$\lambda_{2}$ fit'
    )
    
    # x2, z2
    ax[1, i].scatter(
        n, p0_2_lambda1, color='green', label=r'$\lambda_{1}$ data'
    )
    ax[1, i].plot(
        n_continuous, A2_lambda1 * np.exp(-k2_lambda1 * n_continuous) + b2_lambda1, 
        color='green', label=r'$\lambda_{1}$ fit'
    )
    ax[1, i].scatter(
        n, p0_2_lambda2, color='green', marker='x', label=r'$\lambda_{2}$ data'
    )
    ax[1, i].plot(
        n_continuous, 
        A2_lambda2 * np.exp(-k2_lambda2 * n_continuous) + b2_lambda2, 
        color='green', linestyle='--', label=r'$\lambda_{2}$ fit'
    )
    # x3, z3
    ax[2, i].scatter(n, p0_3_lambda1, color='blue', label=r'$\lambda_{1}$ data')
    ax[2, i].plot(
        n_continuous, A3_lambda1 * np.exp(-k3_lambda1 * n_continuous) + b3_lambda1,
        color='blue', label=r'$\lambda_{1}$ fit'
    )
    ax[2, i].scatter(
        n, p0_3_lambda2, color='blue', marker='x', label=r'$\lambda_{2}$ data'
    )
    ax[2, i].plot(
        n_continuous,
        A3_lambda2 * np.exp(-k3_lambda2 * n_continuous) + b3_lambda2,
        color='blue', linestyle='--', label=r'$\lambda_{2}$ fit'
    )
    ax[0, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[1, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax[2, 0].set_ylim(484, 590)
# text above row 0
fig.text(0.5, 0.91, f'x={(x1-128)*dx:.2f} mm, z={(z1-128)*dx:.2f} mm', ha='center', va='top', fontsize=25)
# place text above row 1
fig.text(0.5, 0.64, f'x={(x2-128)*dx:.2f} mm, z={(z2-128)*dx:.2f} mm', ha='center', va='top', fontsize=25)
# place text above row 2
fig.text(0.5, 0.37, f'x={(x3-128)*dx:.2f} mm, z={(z3-128)*dx:.2f} mm', ha='center', va='top', fontsize=25)

ax[0, 0].legend()
ax[1, 0].legend()
ax[2, 0].legend()

ax[0, 0].set_title(r'Dataset 1 (no noise)', y=1.17, fontsize=25)
ax[0, 1].set_title(r'Dataset 2 (SNR$_{\mathrm{dB}}=18.4)$', y=1.17, fontsize=25)
ax[0, 2].set_title(r'Dataset 3 (SNR$_{\mathrm{dB}}=8.8)$', y=1.17, fontsize=25)
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
fig.savefig("fig_9_decay_curves.pdf", format="pdf", dpi=600, bbox_inches="tight")
plt.close(fig)

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
    
# save as pdf with high dpi
fig.savefig('decay_curve_images.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close(fig)

fig, ax, frames = heatmap( # convert to mol m^-3 to M=Mol L^-1
    data['ReBphP_PCM_c_tot']*1e-3*1e7, dx=sim_cfg['dx'], cbar_label=r'$10^{-7}$ M'
)
ax[0].plot((x1-128)*dx, (z1-128)*dx, 'rx', markersize=20, markeredgewidth=4)
ax[0].plot((x2-128)*dx, (z2-128)*dx, 'gx', markersize=20, markeredgewidth=4)
ax[0].plot((x3-128)*dx, (z3-128)*dx, 'bx', markersize=20, markeredgewidth=4)
fig.savefig('roi_concentration.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close(fig)

fig, ax, frames = heatmap( # convert to mol m^-3 to M=Mol L^-1
    data['p0_tr'][0,0,0], dx=sim_cfg['dx'], cbar_label=r'Pa J$^{-1}$'
)
ax[0].plot((x1-128)*dx, (z1-128)*dx, 'rx', markersize=20, markeredgewidth=4)
ax[0].plot((x2-128)*dx, (z2-128)*dx, 'gx', markersize=20, markeredgewidth=4)
ax[0].plot((x3-128)*dx, (z3-128)*dx, 'bx', markersize=20, markeredgewidth=4)
fig.savefig('roi_reconstruction.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.close(fig)

print(f'x1={(x1-128)*dx:.2f} mm, z1={(z1-128)*dx:.2f} mm')
print(f'x2={(x2-128)*dx:.2f} mm, z2={(z2-128)*dx:.2f} mm')
print(f'x3={(x3-128)*dx:.2f} mm, z3={(z3-128)*dx:.2f} mm')