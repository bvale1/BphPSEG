from dataloader import heatmap, load_sim
import numpy as np

path = '\\\\wsl$\\Ubuntu-22.04\\home\\wv00017\\python_BphP_MSOT_sim\\20231123_BphP_phantom_test'
#path = '\\\\wsl$\\Ubuntu-22.04\\home\\wv00017\\python_BphP_MSOT_sim\\20231123_Clara_phantom_eta0p006_eta0p0018'


[data, sim_cfg] = load_sim(path, args='all')

heatmap(
    data['p0_tr'][0,:,0],
    title='initial pressure reconstructions',
    dx=sim_cfg['dx'],
    labels=['pulse 1 (680nm)', 'pulse 1 (770nm)'],
    cbar_label=r'Pa J$^{-1}$'
)
heatmap(
    data['background_mua_mus'][:,0],
    title='absorption coefficient',
    dx=sim_cfg['dx'],
    labels=[r'$\mu_a$(680nm)', r'$\mu_a$(770nm)'],
    cbar_label=r'm$^{-1}$'
)
heatmap(
    data['background_mua_mus'][:,1],
    title='scattering coefficient',
    dx=sim_cfg['dx'],
    labels=[r'$\mu_s$(680nm)', r'$\mu_s$(770nm)'],
    cbar_label=r'm$^{-1}$'
)
heatmap(
    data['ReBphP_PCM_c_tot'],
    title='total protein concentration',
    dx=sim_cfg['dx'],
    cbar_label=r'$10^{3}$ M'
)