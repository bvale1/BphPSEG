import numpy as np
import h5py


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


if __name__ == '__main__':
    paths = ['']
    dataset_name = 'homogeneous_cylinders'
    
    # process one sample at a time,
    # the option to run in parallel may be implemented later
    for path in paths:
        [data, cfg] = load_sim(path, args=['p0_tr', 'ReBphP_PCM_c_tot'])
        
        