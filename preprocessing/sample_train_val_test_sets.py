import numpy as np
import torch, logging
from torch.utils.data import DataLoader, random_split
from custom_pytorch_utils.custom_datasets import BphP_MSOT_Dataset, BphP_MSOT_raw_image_Dataset
from custom_pytorch_utils.custom_transforms import Normalise, ReplaceNaNWithZero, \
    BinaryMaskToLabel
from torchvision import transforms

def get_dataset_mins_maxs(dataset):
    if dataset.gt_type != 'regression':
        logging.info('Warning: regression ground truth type should be used \
            calculate the correct dataset min, max and mean values')
    
    # Used to normalise the features of the dataset, also computes the class
    # weights for the binary classification task
    (X, Y) = dataset[0]
    X_max = torch.empty(
        (
            dataset.__len__(), # sample number
            X.shape[0] # feature number
        ), 
        dtype=torch.float32,
        requires_grad=False
    )
    X_min = X_max.clone()
    Y_max = torch.empty(
        (dataset.__len__()),
        dtype=torch.float32,
        requires_grad=False
    )
    Y_min = Y_max.clone()
    Y_mean = Y_max.clone() # used to calculate the R2 score
    # Need to apply the same tranformations to the entire dataset to get 
    # to predict the correct quantities
    for i in range(len(dataset)):
        # calculate mean, min and max in a streaming fashion
        (X, Y) = dataset[i]
        X_max[i,:] = torch.max(
            torch.flatten(X, start_dim=1, end_dim=2), dim=1
        ).values
        X_min[i,:] = torch.min(
            torch.flatten(X, start_dim=1, end_dim=2), dim=1
        ).values
        Y_max[i] = torch.max(Y)
        Y_min[i] = torch.min(Y)
        Y_mean[i] = torch.mean(Y)
    
    X_max = X_max.max(dim=0).values
    X_min = X_min.min(dim=0).values
    Y_max = Y_max.max()
    Y_min = Y_min.min()    
    Y_mean = Y_mean.mean()
    
    return (X_max, X_min, Y_max, Y_min, Y_mean)


def get_torch_train_val_test_sets(data_path,
                                  gt_type,
                                  train_val_test_split=[0.8, 0.1, 0.1],
                                  batch_size=16):
    # Split the dataset into train, validation and test sets
    # A subset of train is used for the less scalable classical ML models
    
    if gt_type not in ['binary', 'regression']:
        raise ValueError("gt_type must be either 'binary' or 'regression'")
    
    logging.info('Computing overall dataset min and max values for normalising')
    (X_max, X_min, Y_max, Y_min, Y_mean) = get_dataset_mins_maxs(
        BphP_MSOT_Dataset(
            data_path, 
            'regression',
            x_transform=transforms.Compose([ReplaceNaNWithZero()]),
            y_transform=transforms.Compose([ReplaceNaNWithZero()])
        )
    )
    
    normalise_y = Normalise(Y_max, Y_min)
    if gt_type == 'binary':
        dataset = BphP_MSOT_Dataset(
            data_path,
            'binary',
            x_transform=transforms.Compose([
                ReplaceNaNWithZero(),
                Normalise(X_max, X_min)
            ]),
            y_transform=transforms.Compose([
                    ReplaceNaNWithZero(), 
                    BinaryMaskToLabel()
            ])
        )
    elif gt_type == 'regression':
        dataset = BphP_MSOT_Dataset(
            data_path, 
            'regression',
            x_transform=transforms.Compose([
                ReplaceNaNWithZero(),
                Normalise(X_max, X_min)
            ]),
            y_transform=transforms.Compose([
                ReplaceNaNWithZero(),
                normalise_y
            ])
        )
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        train_val_test_split,
        generator=torch.Generator().manual_seed(42) # reproducible results
    )
    logging.info(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: \
        {len(test_dataset)}')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=20
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=20
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=20
    )
    
    return (train_loader, val_loader, test_loader, Y_mean, normalise_y,
            train_dataset, test_dataset, dataset)


def get_sklearn_train_test_sets(train_dataset, test_dataset, sample_train=4e5):
    # Get a subset of the training dataset to train the sklearn models
    # because they are not scalable enough for the entire test set
    
    # currently loads the entire test dataset into memory to sample a subset
    # not scalable for large datasets
    (X, Y) = test_dataset[0]
    train_subset_X = torch.empty(
        (train_dataset.__len__(), X.shape[0], X.shape[1], X.shape[2]),
        dtype=torch.float32, 
        requires_grad=False
    )
    train_subset_Y = torch.empty(
        (train_dataset.__len__(), X.shape[1], X.shape[2]),
        dtype=torch.float32,
        requires_grad=False
    )
    for i in range(train_dataset.__len__()):
        (X, Y) = train_dataset[i]
        if Y.shape[0] == 2:
            # need to convert one-hot encoding to logit class labels
            Y = torch.argmax(Y, dim=0)
        train_subset_X[i,:,:,:] = X
        train_subset_Y[i,:,:] = Y
    
    # transform dimensions for sklearn models
    # X shape: (sample, feature, x, z) -> (sample*x*z, feature)
    # Y shape: (sample, x, z) -> (sample*x*z)
    train_subset_X = torch.flatten(train_subset_X, start_dim=2, end_dim=3)
    train_subset_X = torch.transpose(train_subset_X, 1, 2)
    train_subset_X = torch.flatten(train_subset_X, start_dim=0, end_dim=1)
    train_subset_Y = torch.flatten(train_subset_Y)
    
    # np arrays used for sklearn models
    train_subset_X = train_subset_X.numpy()
    train_subset_Y = train_subset_Y.numpy()
    true_idx = np.where(train_subset_Y > 0)[0]
    false_idx = np.where(train_subset_Y == 0)[0]
    
    np.random.seed(42)
    np.random.shuffle(true_idx)
    np.random.shuffle(false_idx)
    true_idx = true_idx[:int(sample_train/2)]
    false_idx = false_idx[:int(sample_train/2)]
    idx = np.concatenate((true_idx, false_idx))
    train_subset_X = train_subset_X[idx,:]
    train_subset_Y = train_subset_Y[idx]
    
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
    
    return(train_subset_X, train_subset_Y, X_test, Y_test)


def get_raw_image_torch_train_val_test_sets(root_dir,
                                            gt_type,
                                            train_val_test_split=[0.8, 0.1, 0.1],
                                            batch_size=16):
    # Split the dataset into train, validation and test sets
    # used for the raw image classification task
    
    if gt_type not in ['binary', 'regression']:
        raise ValueError("gt_type must be either 'binary' or 'regression'")
    
    logging.info('Computing overall dataset min and max values for normalising')
    (X_max, X_min, Y_max, Y_min, Y_mean) = get_dataset_mins_maxs(
        BphP_MSOT_raw_image_Dataset(
            root_dir, 
            'regression',
            x_transform=transforms.Compose([ReplaceNaNWithZero()]),
            y_transform=transforms.Compose([ReplaceNaNWithZero()])
        )
    )
    
    normalise_y = Normalise(Y_max, Y_min)
    if gt_type == 'binary':
        dataset = BphP_MSOT_raw_image_Dataset(
            root_dir,
            'binary',
            x_transform=transforms.Compose([
                ReplaceNaNWithZero(),
                Normalise(X_max, X_min)
            ]),
            y_transform=transforms.Compose([
                    ReplaceNaNWithZero(), 
                    BinaryMaskToLabel()
            ])
        )
    elif gt_type == 'regression':
        dataset = BphP_MSOT_raw_image_Dataset(
            root_dir, 
            'regression',
            x_transform=transforms.Compose([
                ReplaceNaNWithZero(),
                Normalise(X_max, X_min)
            ]),
            y_transform=transforms.Compose([
                ReplaceNaNWithZero(),
                normalise_y
            ])
        )
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        train_val_test_split,
        generator=torch.Generator().manual_seed(42) # reproducible results
    )
    logging.info(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: \
        {len(test_dataset)}')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=20
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=20
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=20
    )
    
    return (train_loader, val_loader, test_loader, Y_mean, normalise_y,
            dataset)