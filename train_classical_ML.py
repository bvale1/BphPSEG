import numpy as np
import matplotlib.pyplot as plt
from preprocessing.dataloader import heatmap
import argparse

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import h5py, logging, json, os

logging.basicConfig(level=logging.DEBUG)

def plot_features(path):
    with h5py.File(os.path.join(path, 'dataset.h5'), 'r') as f:
        samples = list(f.keys())
        X = []
        Y = []
        bg_masks = []
        for sample in samples:
            bg_masks.append(f[sample]['bg_mask'][()])
            import matplotlib.pyplot as plt
            X.append(f[sample]['features'][()])
            Y.append(f[sample]['c_mask'][()])

    with open(os.path.join(path, 'config.json'), 'r') as f:
        data_cfg = json.load(f)

    heatmap(
        X[-1], 
        labels=data_cfg['feature_names'],
        title=sample,
        sharescale=False,
        cmap='cool'
    )
    plt.savefig(os.path.join(path, 'features.png'))
    plt.close()
    plt.imshow(Y[-1])
    plt.savefig(os.path.join(path, 'c_mask.png'))
    plt.close()
    plt.imshow(bg_masks[-1])
    plt.savefig(os.path.join(path, 'bg_mask.png'))
    plt.close()


def load_dataset(path):

    with open(os.path.join(path, 'config.json'), 'r') as f:
        data_cfg = json.load(f)
    with h5py.File(os.path.join(path, 'dataset.h5'), 'r') as f:
        samples = list(f.keys())
        X = []
        Y = []
        bg_masks = []
        for sample in samples:
            bg_masks.append(f[sample]['bg_mask'][()])
            import matplotlib.pyplot as plt
            X.append(f[sample]['features'][()][:,bg_masks[-1]])
            Y.append(f[sample]['c_mask'][()][bg_masks[-1]])
    
    X = np.transpose(np.asarray(X), (0,2,1))
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    
    # discard nan values
    not_nan_idx = ~np.isnan(X).any(axis=1)
    X = X[not_nan_idx]
    Y = Y[not_nan_idx]
            
    return X, Y, data_cfg, np.asarray(bg_masks)


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='20231127_homogeneous_cylinders')
    argparser.add_argument('--git_hash', type=str, default='None')
    
    path = argparser.parse_args().path
    cfg = {}
    cfg['data_path'] = path
    cfg['git_hash'] = argparser.parse_args().git_hash
    
    KNN_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA(n_components=0.95)),
        ('clf', KNeighborsClassifier())
    ])
    SVM_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA(n_components=0.95)),
        ('clf', LinearSVC())
    ])
    RF_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA(n_components=0.95)),
        ('clf', RandomForestClassifier())
    ])
    XGB_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA(n_components=0.95)),
        ('clf', XGBClassifier())
    ])
    
    
    X, Y, data_cfg, _ = load_dataset(path)
    
    logging.info(f'dataset loaded, X shape: {X.shape}, Y shape: {Y.shape}')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        logging.info(f'Fold {i+1}/5')
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        for pipeline in [KNN_pipeline, SVM_pipeline, RF_pipeline, XGB_pipeline]:
            logging.info(f'Training {pipeline}')
            pipeline.fit(X_train, Y_train)
            Y_pred = pipeline.predict(X_test)
            logging.info(f'Accuracy: {accuracy_score(Y_test, Y_pred)}')
            logging.info(f'Precision: {precision_score(Y_test, Y_pred)}')
            logging.info(f'Recall: {recall_score(Y_test, Y_pred)}')
            logging.info(f'F1: {f1_score(Y_test, Y_pred)}')
            logging.info(f'Confusion matrix: {confusion_matrix(Y_test, Y_pred)}')
            logging.info(f'Classification report: {classification_report(Y_test, Y_pred)}')