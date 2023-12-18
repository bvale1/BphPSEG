import numpy as np
import matplotlib.pyplot as plt
from preprocessing.dataloader import heatmap
import argparse

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, matthews_corrcoef, \
    jaccard_score, mean_squared_error, mean_absolute_error, r2_score, \
    explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import NuSVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
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


def load_dataset(path, gt_type, max_samples=100000):
    
    if gt_type not in ['binary', 'regression']:
        # use binary classification or value regression
        raise ValueError("gt_type must be either 'binary' or 'regression'")

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
            if gt_type == 'binary':
                Y.append(f[sample]['c_mask'][()][bg_masks[-1]])
            elif gt_type == 'regression':
                Y.append(f[sample]['c_tot'][()][bg_masks[-1]])
    
    X = np.transpose(np.asarray(X), (0,2,1))
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    
    # discard nan values
    not_nan_idx = ~np.isnan(X).any(axis=1)
    X = X[not_nan_idx]
    Y = Y[not_nan_idx]
    
    if len(Y) < max_samples:
        max_samples = len(Y)
    
    if gt_type == 'binary':
        X, Y = class_equalize(X, Y, max_samples)
    elif gt_type == 'regression':
        X, Y = positive_samples(X, Y, max_samples)
            
    return X, Y, data_cfg, np.asarray(bg_masks)

def class_equalize(X, Y, max_samples):
    # randomly choose max_samples/2 positive and max_samples/2 negative samples
    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]
    logging.info(f'positive samples: {len(pos_idx)}, negative samples: {len(neg_idx)}')
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx[:int(max_samples/2)]
    neg_idx = neg_idx[:int(max_samples/2)]
    idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    return X, Y

def positive_samples(X, Y, max_samples):
    # randomly choose max_samples positive samples
    pos_idx = np.where(Y > 0)[0]
    np.random.seed(42)
    np.random.shuffle(pos_idx)
    pos_idx = pos_idx[:max_samples]
    X = X[pos_idx]
    Y = Y[pos_idx]
    return X, Y

def plot_PCA(X, Y):
    pca = Pipeline([('scaler', StandardScaler()), ('PCA', PCA())])
    X_pca = pca.fit_transform(X)
    print(f'explained variance ratio: {pca.named_steps["PCA"].explained_variance_ratio_}')
    print(f'explained variance: {pca.named_steps["PCA"].explained_variance_}')
    print(f'principal components: {pca.named_steps["PCA"].components_}')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), subplot_kw={"projection": "3d"})
    ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=Y, s=1, alpha=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.savefig(os.path.join(path, 'PCA.png'))
    
    return pca, fig, ax


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='20231127_homogeneous_cylinders')
    argparser.add_argument('--git_hash', type=str, default='None')
    
    args = argparser.parse_args()
    path = args.data_path
    cfg = {}
    cfg['data_path'] = path
    cfg['git_hash'] = args.git_hash
    
    # ===========================BINARY CLASSIFICATION==========================
    
    KNN_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', KNeighborsClassifier())
    ])
    SVM_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', NuSVC(kernel='rbf', probability=False))
    ])
    RF_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', RandomForestClassifier())
    ])
    XGB_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', XGBClassifier())
    ])
    ANN_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', MLPClassifier(max_iter=500))
    ])
    
    
    X, Y, data_cfg, _ = load_dataset(path, gt_type='binary')
    logging.info(f'binary classification dataset loaded, X shape: {X.shape}, Y shape: {Y.shape}')
    
    plot_PCA(X, Y)
    
    # 5 folds, 5 classifiers
    F1 = np.zeros((5,5), dtype=np.float32) # F1 score
    Accuracy = np.copy(F1) # also known as overall accuracy, in clustering also referred to as Rand index
    Precision = np.copy(F1) # also known as positive predictive value
    Recall = np.copy(F1) # also known as sensitivity, true positive rate
    MCC = np.copy(F1) # Matthews Correlation Coefficient or Phi Coefficient
    IOU = np.copy(F1) # Intersection Over Union, also referred to as Jaccard index
    
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        
        logging.info(f'Fold {i+1}/5')
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        for j, pipeline in enumerate([KNN_pipeline, SVM_pipeline, RF_pipeline, XGB_pipeline, ANN_pipeline]):
            
            logging.info(f'Training {pipeline}')
            pipeline.fit(X_train, Y_train)
            Y_pred = pipeline.predict(X_test)
            
            F1[i,j] = f1_score(Y_test, Y_pred)
            Accuracy[i,j] = accuracy_score(Y_test, Y_pred)
            Precision[i,j] = precision_score(Y_test, Y_pred)
            Recall[i,j] = recall_score(Y_test, Y_pred)
            MCC[i,j] = matthews_corrcoef(Y_test, Y_pred)
            IOU[i,j] = jaccard_score(Y_test, Y_pred)
            
    i = np.argmax(F1, axis=0)
    logging.info('best F1 scores:')
    logging.info(f'KNN: F1={F1[i[0],0]}, Accuracy={Accuracy[i[0],0]}, Precision={Precision[i[0],0]}, Recall={Recall[i[0],0]}, MCC={MCC[i[0],0]}, IOU={IOU[i[0],0]}')
    logging.info(f'SVM: F1={F1[i[1],1]}, Accuracy={Accuracy[i[1],1]}, Precision={Precision[i[1],1]}, Recall={Recall[i[1],1]}, MCC={MCC[i[1],1]}, IOU={IOU[i[1],1]}')
    logging.info(f'RF: F1={F1[i[2],2]}, Accuracy={Accuracy[i[2],2]}, Precision={Precision[i[2],2]}, Recall={Recall[i[2],2]}, MCC={MCC[i[2],2]}, IOU={IOU[i[2],2]}')
    logging.info(f'XGB: F1={F1[i[3],3]}, Accuracy={Accuracy[i[3],3]}, Precision={Precision[i[3],3]}, Recall={Recall[i[3],3]}, MCC={MCC[i[3],3]}, IOU={IOU[i[3],3]}')
    logging.info(f'ANN: F1={F1[i[4],4]}, Accuracy={Accuracy[i[4],4]}, Precision={Precision[i[4],4]}, Recall={Recall[i[4],4]}, MCC={MCC[i[4],4]}, IOU={IOU[i[4],4]}')
    
    # ================================REGRESSION================================
    
    KNR_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', KNeighborsRegressor())
    ])
    SVR_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', SVR(kernel='rbf'))
    ])
    RF_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', RandomForestRegressor())
    ])
    XGB_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', XGBRegressor())
    ])
    ANN_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', MLPRegressor(max_iter=500))
    ])
    
    X, Y, data_cfg, _ = load_dataset(path, gt_type='regression')
    logging.info('regression dataset loaded, X shape: {X.shape}, Y shape: {Y.shape}')
    
    y_scaler = StandardScaler()
    Y_scaled = y_scaler.fit_transform(Y.reshape(-1,1)).reshape(-1)
    
    MSE = np.zeros((5,5), dtype=np.float32) # Mean Squared Error
    MAE = np.copy(MSE) # Mean Absolute Error
    MAE_Percent = np.copy(MSE) # Mean Absolute Error in percent
    STD_AE_Percent = np.copy(MSE) # STD of absolute error in percent
    R2 = np.copy(MSE) # R^2 score
    EVS = np.copy(MSE) # Explained Variance Score    
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
            
            logging.info(f'Fold {i+1}/5')
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y_scaled[train_index], Y[test_index]
            
            for j, pipeline in enumerate([KNR_pipeline, SVR_pipeline, RF_pipeline, XGB_pipeline, ANN_pipeline]):
                
                logging.info(f'Training {pipeline}')
                pipeline.fit(X_train, Y_train)
                Y_pred_scaled = pipeline.predict(X_test)
                Y_pred = y_scaler.inverse_transform(Y_pred_scaled.reshape(-1,1)).reshape(-1)
                
                MSE[i,j] = mean_squared_error(Y_test, Y_pred)
                MAE[i,j] = mean_absolute_error(Y_test, Y_pred)
                AE = np.abs(Y_test - Y_pred)
                MAE_Percent[i,j] = np.mean(100 * AE / Y_test) # Mean Absolute Error in percent
                STD_AE_Percent[i,j] = np.std(100 * AE / Y_test) # STD of absolute error in percent
                R2[i,j] = r2_score(Y_test, Y_pred)
                EVS[i,j] = explained_variance_score(Y_test, Y_pred)
    
    i = np.argmin(MSE, axis=0)
    logging.info('best MSE scores:')
    logging.info(f'KNR: MSE={MSE[i[0],0]}, MAE={MAE[i[0],0]}, R2={R2[i[0],0]}, EVS={EVS[i[0],0]}, MAE_Percent={MAE_Percent[i[0],0]}, STD_AE_Percent={STD_AE_Percent[i[0],0]}')
    logging.info(f'SVR: MSE={MSE[i[1],1]}, MAE={MAE[i[1],1]}, R2={R2[i[1],1]}, EVS={EVS[i[1],1]}, MAE_Percent={MAE_Percent[i[1],1]}, STD_AE_Percent={STD_AE_Percent[i[1],1]}')
    logging.info(f'RF: MSE={MSE[i[2],2]}, MAE={MAE[i[2],2]}, R2={R2[i[2],2]}, EVS={EVS[i[2],2]}, MAE_Percent={MAE_Percent[i[2],2]}, STD_AE_Percent={STD_AE_Percent[i[2],2]}')
    logging.info(f'XGB: MSE={MSE[i[3],3]}, MAE={MAE[i[3],3]}, R2={R2[i[3],3]}, EVS={EVS[i[3],3]}, MAE_Percent={MAE_Percent[i[3],3]}, STD_AE_Percent={STD_AE_Percent[i[3],3]}')
    logging.info(f'ANN: MSE={MSE[i[4],4]}, MAE={MAE[i[4],4]}, R2={R2[i[4],4]}, EVS={EVS[i[4],4]}, MAE_Percent={MAE_Percent[i[4],4]}, STD_AE_Percent={STD_AE_Percent[i[4],4]}')
    