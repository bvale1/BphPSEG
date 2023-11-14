import numpy as np
import UNet
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
import xdgboost
import h5py, logging, json, torch, os

logging.basicConfig(level=logging.Debug)

def load_dataset(path):

    with open(os.path.join(path, 'config.json'), 'r') as f:
        data_cfg = json.load(f)
    with h5py.File(os.path.join(path, 'dataset.h5'), 'r') as f:
        samples = list(f.keys())
        X = np.array([])
        Y = np.array([])
        bg_masks = []
        for sample in samples:
            bg_masks.append(f[sample]['bg_mask'][()])
            X.append(f[sample]['features'][()][bg_masks[-1]])
            Y.append(f[sample]['C_mask'][()][bg_masks[-1]])
            
    

if __name__ == '__main__':
    path = ''
    
    with h5py.File(os.path.join(path, 'dataset.h5'), 'r') as f:
        data = f['data'][()]
    with open(os.path.join(path, 'config.json'), 'r') as f:
        data_cfg = json.load(f)
    
    
        
    clf = svm.SVC() # kernal='linear', 'rbf', 'poly'
    clf.fit(X, y)
    clf.predict(X)