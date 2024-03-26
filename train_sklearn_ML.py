import numpy as np
import matplotlib.pyplot as plt
from preprocessing.dataloader import heatmap
import argparse
from preprocessing.sample_train_val_test_sets import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score, matthews_corrcoef, \
    jaccard_score, mean_squared_error, mean_absolute_error, r2_score, \
    explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
#from sklearn.svm import NuSVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import h5py, logging, json, os, timeit
import torch
import torchvision.transforms as transforms
from custom_pytorch_utils.custom_transforms import ReplaceNaNWithZero, \
    MaxMinNormalise, BinaryMaskToLabel
from custom_pytorch_utils.custom_datasets import BphP_MSOT_Dataset
from torch.utils.data import random_split


logging.basicConfig(level=logging.INFO)


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
    plt.savefig('PCA.png')
    
    return pca, fig, ax


def specificity_score(y_true, y_pred):
    # true negative rate
    print(y_pred.shape, y_true.shape)
    print(y_pred)
    print(y_true)
    print(y_pred.dtype, y_true.dtype)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)


# sanity check for NaNs and Infs
def percent_of_array_is_finite(arr):
    if type(arr) == torch.Tensor:
        return 100 * torch.sum(torch.isfinite(arr)) / torch.prod(torch.tensor(arr.shape))
    elif type(arr) == np.ndarray:
        return 100 * np.sum(np.isfinite(arr)) / np.prod(arr.shape)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='preprocessing/20240305_homogeneous_cylinders/')
    argparser.add_argument('--git_hash', type=str, default='None')
    
    args = argparser.parse_args()
    path = args.data_path
    cfg = {}
    cfg['data_path'] = path
    cfg['git_hash'] = args.git_hash
    
    # ===========================BINARY CLASSIFICATION==========================
    
    '''
    KNN_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', KNeighborsClassifier())
    ])
    SVM_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', NuSVC(kernel='rbf', probability=False))
    ])
    '''
    RF_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', RandomForestClassifier())
    ])
    XGB_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', XGBClassifier())
    ])
    ANN_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('clf', MLPClassifier(max_iter=500))
    ])
    '''
    (_, _, _, _, _, train_dataset, test_dataset, _) = get_torch_train_val_test_sets(
        path, 
        gt_type='binary',
    )
    '''
    
    with open(os.path.join(os.path.dirname(args.data_path), 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    dataset = BphP_MSOT_Dataset(
        args.data_path, 
        'binary', 
        'features', 
        x_transform=transforms.Compose([
            ReplaceNaNWithZero(), 
            MaxMinNormalise(
                torch.Tensor(config['image_normalisation_params']['max']),
                torch.Tensor(config['image_normalisation_params']['min'])
            )
        ]),
        y_transform=transforms.Compose([
            ReplaceNaNWithZero(),
            BinaryMaskToLabel()
        ])
    )
    train_dataset, _, test_dataset = random_split(
        dataset, 
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # reproducible results
    )
    logging.info(f'train: {len(train_dataset)}, test: {len(test_dataset)}')
    
    X_train, Y_train, X_test, Y_test = get_sklearn_train_test_sets(
        train_dataset, 
        test_dataset,
        subsample_train=None # 4e5
    )
    
    logging.info(f'binary classification dataset loaded \n \
        X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape} \n \
        X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')
    
    # sanity check features
    #for i in range(X_train.shape[1]):
    #    logging.info(f'percent of X_train[:,{i}] is finite: {percent_of_array_is_finite(X_train[:,i])}')
    
    plot_PCA(X_test, Y_test)
    
    # classifiers
    F1 = np.zeros(3, dtype=np.float32) # F1 score
    Accuracy = np.copy(F1) # also known as overall accuracy, in clustering also referred to as Rand index
    Precision = np.copy(F1) # also known as positive predictive value
    Recall = np.copy(F1) # also known as sensitivity, true positive rate
    Specificity = np.copy(F1) # also known as selectivity, true negative rate
    MCC = np.copy(F1) # Matthews Correlation Coefficient or Phi Coefficient
    IOU = np.copy(F1) # Intersection Over Union, also referred to as Jaccard index
    
    for i, pipeline in enumerate([RF_pipeline, XGB_pipeline, ANN_pipeline]):
        start = timeit.default_timer()
        logging.info(f'Training {pipeline}')
        pipeline.fit(X_train, Y_train)
        Y_pred = pipeline.predict(X_test)
        
        F1[i] = f1_score(Y_test, Y_pred)
        Accuracy[i] = accuracy_score(Y_test, Y_pred)
        Precision[i] = precision_score(Y_test, Y_pred)
        Recall[i] = recall_score(Y_test, Y_pred)
        Specificity[i] = specificity_score(Y_test, Y_pred)
        MCC[i] = matthews_corrcoef(Y_test, Y_pred)
        IOU[i] = jaccard_score(Y_test, Y_pred)
        logging.info(f'F1={F1[i]}, Accuracy={Accuracy[i]}, Precision={Precision[i]}, Recall={Recall[i]}, Specificity={Specificity[i]}, MCC={MCC[i]}, IOU={IOU[i]}')
        logging.info(f'time taken: {timeit.default_timer() - start}')
        
    #logging.info(f'KNN: F1={F1[0]}, Accuracy={Accuracy[0]}, Precision={Precision[0]}, Recall={Recall[0]}, Specificity={Specificity[0]}, MCC={MCC[0]}, IOU={IOU[0]}')
    #logging.info(f'SVM: F1={F1[1]}, Accuracy={Accuracy[1]}, Precision={Precision[1]}, Recall={Recall[1]}, Specificity={Specificity[1]}, MCC={MCC[1]}, IOU={IOU[1]}')
    logging.info(f'RF: F1={F1[0]}, Accuracy={Accuracy[0]}, Precision={Precision[0]}, Recall={Recall[0]}, Specificity={Specificity[0]}, MCC={MCC[0]}, IOU={IOU[0]}')
    logging.info(f'XGB: F1={F1[1]}, Accuracy={Accuracy[1]}, Precision={Precision[1]}, Recall={Recall[1]}, Specificity={Specificity[1]}, MCC={MCC[1]}, IOU={IOU[1]}')
    logging.info(f'ANN: F1={F1[2]}, Accuracy={Accuracy[2]}, Precision={Precision[2]}, Recall={Recall[2]}, Specificity={Specificity[2]}, MCC={MCC[2]}, IOU={IOU[2]}')
    
    # ================================REGRESSION================================
    
    '''
    KNR_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', KNeighborsRegressor())
    ])
    SVR_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', SVR(kernel='rbf'))
    ])
    '''
    RF_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', RandomForestRegressor())
    ])
    XGB_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', XGBRegressor())
    ])
    ANN_pipeline = Pipeline([
        #('scaler', StandardScaler()),
        #('reduce_dim', PCA(n_components=0.95)),
        ('reg', MLPRegressor(max_iter=500))
    ])
    '''
    _, _, _, _, normalise_y, train_dataset, test_dataset, _ = get_torch_train_val_test_sets(
        path, 
        gt_type='regression'
    )
    '''
    with open(os.path.join(os.path.dirname(args.data_path), 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
        
    normalise_y = MaxMinNormalise(
        torch.Tensor(config['concentration_normalisation_params']['max']),
        torch.Tensor(config['concentration_normalisation_params']['min'])
    )
    dataset = BphP_MSOT_Dataset(
        args.data_path, 
        'regression', 
        'features', 
        x_transform=transforms.Compose([
            ReplaceNaNWithZero(), 
            MaxMinNormalise(
                torch.Tensor(config['image_normalisation_params']['max']),
                torch.Tensor(config['image_normalisation_params']['min'])
            )
        ]),
        y_transform=transforms.Compose([
            ReplaceNaNWithZero(),
            normalise_y
        ])
    )
    train_dataset, _, test_dataset = random_split(
        dataset, 
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # reproducible results
    )
    logging.info(f'train: {len(train_dataset)}, test: {len(test_dataset)}')
    
    X_train, Y_train, X_test, Y_test = get_sklearn_train_test_sets(
        train_dataset, 
        test_dataset,
        subsample_train=None # 4e5
    )
    
    logging.info(f'regression dataset loaded \n \
        X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape} \n \
        X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')
    
    MSE = np.zeros(3, dtype=np.float32) # Mean Squared Error
    MAE = np.copy(MSE) # Mean Absolute Error
    R2 = np.copy(MSE) # R^2 score
    EVS = np.copy(MSE) # Explained Variance Score
    
    Y_test = normalise_y.inverse_numpy_flat(Y_test)
    
    for i, pipeline in enumerate([RF_pipeline, XGB_pipeline, ANN_pipeline]):
        start = timeit.default_timer()
        
        logging.info(f'Training {pipeline}')
        pipeline.fit(X_train, Y_train)
        Y_pred = pipeline.predict(X_test)
        Y_pred = normalise_y.inverse_numpy_flat(Y_pred)
        
        MSE[i] = mean_squared_error(Y_test, Y_pred)
        MAE[i] = mean_absolute_error(Y_test, Y_pred)
        AE = np.abs(Y_test - Y_pred)
        R2[i] = r2_score(Y_test, Y_pred)
        EVS[i] = explained_variance_score(Y_test, Y_pred)
        logging.info(f'MSE={MSE[i]}, MAE={MAE[i]}, R2={R2[i]}, EVS={EVS[i]}')
        logging.info(f'time taken: {timeit.default_timer() - start}')

    # Note percent error is not a reliable metric as it is undefined for Y=0
    #logging.info(f'KNR: MSE={MSE[0]}, MAE={MAE[0]}, R2={R2[0]}, EVS={EVS[0]}')
    #logging.info(f'SVR: MSE={MSE[1]}, MAE={MAE[1]}, R2={R2[1]}, EVS={EVS[1]}')
    logging.info(f'RF: MSE={MSE[0]}, MAE={MAE[0]}, R2={R2[0]}, EVS={EVS[0]}')
    logging.info(f'XGB: MSE={MSE[1]}, MAE={MAE[1]}, R2={R2[1]}, EVS={EVS[1]}')
    logging.info(f'ANN: MSE={MSE[2]}, MAE={MAE[2]}, R2={R2[2]}, EVS={EVS[2]}')
    