import torch
import logging
import pytorch_lightning as pl
from argparse import ArgumentParser
from preprocessing.sample_train_val_test_sets import *

from train_Unet_DL import *
        

def train_UNet_raw_images_main():
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--root_dir', type=str, default='/mnt/f/cluster_MSOT_simulations/Bphp_phantom/')
    parser.add_argument('--git_hash', type=str, default='None')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = UNet.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    trainer = pl.Trainer.from_argparse_args(
        args, check_val_every_n_epoch=1, accelerator='gpu', devices=1, max_epochs=args.epochs
    )
    
    # weighting the true class more heavily improves the True Positive Rate
    # but tends to decrease all other performance metrics
    weight_true = 1.0
    weight_false = 1.0    
    
    # BINARY CLASSIFICATION / SEMANTIC SEGMENTATION
    
    (train_loader, val_loader, test_loader, Y_mean, normalise_y, dataset) = get_raw_image_torch_train_val_test_sets(
        args.root_dir,
        'binary',
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=args.batch_size
    )
    
    model = UNet(
        32, 
        args.out_channels, 
        'binary',
        weight_false=weight_false, 
        weight_true=weight_true,
        y_mean=Y_mean
    )
    
    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(model, test_loader)
    
    print(result)
    # visualise the results
    dataset.get_config(0)
    dataset.plot_sample(0, model(dataset[0][0].unsqueeze(0)), save_name='c139519.p0_semantic_segmentation_epoch100.png')
    
    # REGRESSION / QUANTITATIVE SEGMENTATION    
    weight_true = 1.0
    weight_false = 1.0 
    
    trainer = pl.Trainer.from_argparse_args(
        args, check_val_every_n_epoch=1, accelerator='gpu', devices=1, max_epochs=args.epochs
    )
    
    # save instance to invert transform for testing
    
    (train_loader, val_loader, test_loader, Y_mean, normalise_y, dataset) = get_raw_image_torch_train_val_test_sets(
        args.root_dir,
        'regression',
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=args.batch_size
    )
    
    model = UNet(
        32, 
        1, 
        'regression',
        y_transform=normalise_y, 
        weight_false=weight_false,
        weight_true=weight_true
    )

    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(model, test_loader)
    
    # visualise the results
    print(result)    
    dataset.get_config(0)
    dataset.plot_sample(
        0, 
        model(dataset[0][0].unsqueeze(0)), 
        save_name='c139519.p0_quantitative_segmentation_epoch100.png',
        y_transform=normalise_y
    )
    
    
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('highest')
    train_UNet_raw_images_main()