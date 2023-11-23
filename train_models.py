import numpy as np
#import UNet_model

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
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
            
    return X, Y, data_cfg, bg_masks

if __name__ == '__main__':
    path = ''
    
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
        ('clf', xgboost())
    ])
    
    X, Y, data_cfg, _ = load_dataset(path)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        logging.info(f'Fold {i+1}/5')
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        for pipeline in [KNN_pipeline, SVM_pipeline, RF_pipeline, XGB_pipeline]:
            pipeline.fit(X_train, Y_train)
            Y_pred = pipeline.predict(X_test)
            logging.info(f'Accuracy: {accuracy_score(Y_test, Y_pred)}')
            logging.info(f'Precision: {precision_score(Y_test, Y_pred)}')
            logging.info(f'Recall: {recall_score(Y_test, Y_pred)}')
            logging.info(f'F1: {f1_score(Y_test, Y_pred)}')
            logging.info(f'Confusion matrix: {confusion_matrix(Y_test, Y_pred)}')
            logging.info(f'Classification report: {classification_report(Y_test, Y_pred)}')