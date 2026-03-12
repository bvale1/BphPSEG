import numpy as np
import torch


def get_sklearn_train_test_sets(train_dataset, val_dataset, test_dataset):
    (X, Y, bg_mask, inclusion_mask, _) = test_dataset[0]
    
    X_train = torch.empty(
        (train_dataset.__len__(),) + X.shape[-3:],
        dtype=torch.float32, 
        requires_grad=False
    )
    Y_train = torch.empty(
        (train_dataset.__len__(),) + Y.shape[-2:],
        dtype=torch.float32,
        requires_grad=False
    )
    bg_mask_train = torch.empty(
        (train_dataset.__len__(),) + bg_mask.shape,
        dtype=torch.bool,
        requires_grad=False
    )
    inclusion_mask_train = torch.empty(
        (train_dataset.__len__(),) + inclusion_mask.shape,
        dtype=torch.bool,
        requires_grad=False
    )
    sample_names_train = []
    for i in range(train_dataset.__len__()):
        (X, Y, bg_mask, inclusion_mask, sample_name) = train_dataset[i]
        sample_names_train.append(sample_name)
        if Y.shape[0] == 2:
            # need to convert one-hot encoding to logit class labels
            Y = torch.argmax(Y, dim=0)
        X_train[i,:,:,:] = X
        Y_train[i,:,:] = Y
        bg_mask_train[i,:,:] = bg_mask
        inclusion_mask_train[i,:,:] = inclusion_mask
    
    # transform dimensions for sklearn models
    # X shape: (sample, feature, x, z) -> (sample*x*z, feature)
    # Y shape: (sample, x, z) -> (sample*x*z)
    X_train = torch.flatten(X_train, start_dim=2, end_dim=3)
    X_train = torch.transpose(X_train, 1, 2)
    X_train = torch.flatten(X_train, start_dim=0, end_dim=1)
    Y_train = torch.flatten(Y_train)
    bg_mask_train = torch.flatten(bg_mask_train)
    inclusion_mask_train = torch.flatten(inclusion_mask_train)
    
    # np arrays used for sklearn models
    X_train = X_train.numpy()
    Y_train = Y_train.numpy()
    bg_mask_train = bg_mask_train.numpy()
    inclusion_mask_train = inclusion_mask_train.numpy()
    
    X_val = torch.empty(
        (val_dataset.__len__(),) + X.shape[-3:],
        dtype=torch.float32,
        requires_grad=False
    )
    Y_val = torch.empty(
        (val_dataset.__len__(),) + Y.shape[-2:],
        dtype=torch.float32,
        requires_grad=False
    )
    bg_mask_val = torch.empty(
        (val_dataset.__len__(),) + bg_mask.shape,
        dtype=torch.bool,
        requires_grad=False
    )
    inclusion_mask_val = torch.empty(
        (val_dataset.__len__(),) + inclusion_mask.shape,
        dtype=torch.bool,
        requires_grad=False
    )
    sample_names_val = []
    for i in range(val_dataset.__len__()):
        (X, Y, bg_mask, inclusion_mask, sample_name) = val_dataset[i]
        sample_names_val.append(sample_name)
        if Y.shape[0] == 2:
            # need to convert one-hot encoding to logit class labels
            Y = torch.argmax(Y, dim=0)
        X_val[i,:,:,:] = X
        Y_val[i,:,:] = Y
        bg_mask_val[i] = bg_mask
        inclusion_mask_val[i] = inclusion_mask

    # transform dimensions for sklearn models
    # X shape: (sample, feature, x, z) -> (sample*x*z, feature)
    X_val = torch.flatten(X_val, start_dim=2, end_dim=3)
    X_val = torch.transpose(X_val, 1, 2)
    X_val = torch.flatten(X_val, start_dim=0, end_dim=1)
    Y_val = torch.flatten(Y_val)
    bg_mask_val = torch.flatten(bg_mask_val)
    inclusion_mask_val = torch.flatten(inclusion_mask_val)
    X_val = X_val.numpy()
    Y_val = Y_val.numpy()
    bg_mask_val = bg_mask_val.numpy()
    inclusion_mask_val = inclusion_mask_val.numpy()

    X_test = torch.empty(
        (test_dataset.__len__(),) + X.shape[-3:],
        dtype=torch.float32,
        requires_grad=False
    )
    Y_test = torch.empty(
        (test_dataset.__len__(),) + Y.shape[-2:],
        dtype=torch.float32,
        requires_grad=False
    )
    bg_mask_test = torch.empty(
        (test_dataset.__len__(),) + bg_mask.shape,
        dtype=torch.bool,
        requires_grad=False
    )
    inclusion_mask_test = torch.empty(
        (test_dataset.__len__(),) + inclusion_mask.shape,
        dtype=torch.bool,
        requires_grad=False
    )
    sample_names_test = []
    for i in range(test_dataset.__len__()):
        (X, Y, bg_mask, inclusion_mask, sample_name) = test_dataset[i]
        sample_names_test.append(sample_name)
        if Y.shape[0] == 2:
            # need to convert one-hot encoding to logit class labels
            Y = torch.argmax(Y, dim=0)
        X_test[i,:,:,:] = X
        Y_test[i,:,:] = Y
        bg_mask_test[i] = bg_mask
        inclusion_mask_test[i] = inclusion_mask
    
    # transform dimensions for sklearn models
    # X shape: (sample, feature, x, z) -> (sample*x*z, feature)
    X_test = torch.flatten(X_test, start_dim=2, end_dim=3)
    X_test = torch.transpose(X_test, 1, 2)
    X_test = torch.flatten(X_test, start_dim=0, end_dim=1)
    Y_test = torch.flatten(Y_test)
    bg_mask_test = torch.flatten(bg_mask_test)
    inclusion_mask_test = torch.flatten(inclusion_mask_test)
    X_test = X_test.numpy()
    Y_test = Y_test.numpy()
    bg_mask_test = bg_mask_test.numpy()
    inclusion_mask_test = inclusion_mask_test.numpy()
    
    return (X_train, Y_train, bg_mask_train, inclusion_mask_train, sample_names_train,
            X_val, Y_val, bg_mask_val, inclusion_mask_val, sample_names_val,
            X_test, Y_test, bg_mask_test, inclusion_mask_test, sample_names_test)
