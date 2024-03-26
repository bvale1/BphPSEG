import numpy as np
import torch


def get_sklearn_train_test_sets(train_dataset, test_dataset, subsample_train=None):
    # Get a subset of the training dataset to train the sklearn models
    # if subsample_train is None, the entire training dataset is used
    
    # currently loads the entire dataset into memory to sample a subset
    # which cant be scalable for large datasets
    (X, Y) = test_dataset[0]
    
    if subsample_train is None: # use the entire training dataset
        subsample_train = train_dataset.__len__() * np.prod(Y.shape[-2:])
    
    X_train = torch.empty(
        (train_dataset.__len__(), ) + X.shape,
        dtype=torch.float32, 
        requires_grad=False
    )
    Y_train = torch.empty(
        (train_dataset.__len__(),) + Y.shape[-2:],
        dtype=torch.float32,
        requires_grad=False
    )
    for i in range(train_dataset.__len__()):
        (X, Y) = train_dataset[i]
        if Y.shape[0] == 2:
            # need to convert one-hot encoding to logit class labels
            Y = torch.argmax(Y, dim=0)
        X_train[i,:,:,:] = X
        Y_train[i,:,:] = Y
    
    # transform dimensions for sklearn models
    # X shape: (sample, feature, x, z) -> (sample*x*z, feature)
    # Y shape: (sample, x, z) -> (sample*x*z)
    X_train = torch.flatten(X_train, start_dim=2, end_dim=3)
    X_train = torch.transpose(X_train, 1, 2)
    X_train = torch.flatten(X_train, start_dim=0, end_dim=1)
    Y_train = torch.flatten(Y_train)
    
    # np arrays used for sklearn models
    X_train = X_train.numpy()
    Y_train = Y_train.numpy()
    true_idx = np.where(Y_train > 0)[0]
    false_idx = np.where(Y_train == 0)[0]
    
    np.random.seed(42)
    np.random.shuffle(true_idx)
    np.random.shuffle(false_idx)
    true_idx = true_idx[:int(subsample_train/2)]
    false_idx = false_idx[:int(subsample_train/2)]
    idx = np.concatenate((true_idx, false_idx))
    X_train = X_train[idx,:]
    Y_train = Y_train[idx]
    
    X_test = torch.empty(
        (test_dataset.__len__(), X.shape[0], X.shape[1], X.shape[2]),
        dtype=torch.float32,
        requires_grad=False
    )
    Y_test = torch.empty(
        (test_dataset.__len__(), X.shape[1], X.shape[2]),
        dtype=torch.float32,
        requires_grad=False
    )
    for i in range(test_dataset.__len__()):
        (X, Y) = test_dataset[i]
        if Y.shape[0] == 2:
            # need to convert one-hot encoding to logit class labels
            Y = torch.argmax(Y, dim=0)
        X_test[i,:,:,:] = X
        Y_test[i,:,:] = Y
    
    # transform dimensions for sklearn models
    # X shape: (sample, feature, x, z) -> (sample*x*z, feature)
    X_test = torch.flatten(X_test, start_dim=2, end_dim=3)
    X_test = torch.transpose(X_test, 1, 2)
    X_test = torch.flatten(X_test, start_dim=0, end_dim=1)
    Y_test = torch.flatten(Y_test)
    X_test = X_test.numpy()
    Y_test = Y_test.numpy()
    
    return(X_train, Y_train, X_test, Y_test)
