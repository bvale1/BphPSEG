#!/usr/bin/env python3

import argparse, wandb, logging, torch, os, json, random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from custom_pytorch_utils.custom_transforms import *
from pytorch_models.BphPSEG import BphPSEG
from pytorch_models.BphPQUANT import BphPQUANT
from pytorch_models.MLP import inherit_mlp_class_from_parent
from pytorch_models.Unet import inherit_unet_pretrained_class_from_parent
from pytorch_models.BphP_deeplabv3 import inherit_deeplabv3_smp_resnet101_class_from_parent
from pytorch_models.BphP_segformer import inherit_segformer_class_from_parent
from custom_pytorch_utils.custom_transforms import *
from custom_pytorch_utils.custom_datasets import *
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
import custom_pytorch_utils.augment_models_func as amf

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='preprocessing/20240502_BphP_cylinders/', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--model', choices=['mlp', 'Unet', 'UnetPlusPlus', 'deeplabv3_resnet101', 'segformer'], default='Unet', help='choose from [Unet, UnetPlusPlus, deeplabv3_resnet101, segformer]')
    parser.add_argument('--wandb_log', help='disable log to wandb', action='store_false')
    parser.add_argument('--input_type', choices=['images', 'features'], default='images', help='type of input data')
    parser.add_argument('--gt_type', choices=['binary', 'regression'], default='binary', help='type of ground truth data')
    parser.add_argument('--input_normalisation', choices=['MinMax', 'MeanStd'], default='MinMax', help='normalisation method for input data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout', help='enable dropout layers', action='store_true')
    parser.add_argument('--batchnorm', help='enable batchnorm layers', action='store_true')
    parser.add_argument('--save_test_example', help='disable save test examples to wandb', action='store_false')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    print(f'args dict: {vars(args)}')
    
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('high')
    
    # force cuDNN to deterministically select algorithms, 
    # improves reproducability but reduces performance
    
    # cannot set the deterministic flag unfortunately, as DeepLabV3 and Segformer 
    # use some non-deterministic algorithms
    torch.use_deterministic_algorithms(False)
    print(f'cuDNN deterministic: {torch.torch.backends.cudnn.deterministic}')
    print(f'cuDNN benchmark: {torch.torch.backends.cudnn.benchmark}')
    
    if args.seed: # I can still set the seed manually
        seed = args.seed
    else:
        seed = np.random.randint(0, 2**32 - 1)
    print(f'seed: {seed}')
    logging.info(f'seed: {seed}')
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
    random.seed(seed) 
    pl.seed_everything(seed, workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    
    with open(os.path.join(os.path.dirname(args.root_dir), 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    
    if args.gt_type == 'binary':
        MLP = inherit_mlp_class_from_parent(BphPSEG)
        Unet = inherit_unet_pretrained_class_from_parent(BphPSEG)
        UnetPlusPlus = inherit_unet_pretrained_class_from_parent(BphPSEG)
        BphP_deeplabv3_resnet101 = inherit_deeplabv3_smp_resnet101_class_from_parent(BphPSEG)
        BphP_segformer = inherit_segformer_class_from_parent(BphPSEG)
    elif args.gt_type == 'regression':
        MLP = inherit_mlp_class_from_parent(BphPQUANT)
        Unet = inherit_unet_pretrained_class_from_parent(BphPQUANT)
        UnetPlusPlus = inherit_unet_pretrained_class_from_parent(BphPQUANT)
        BphP_deeplabv3_resnet101 = inherit_deeplabv3_smp_resnet101_class_from_parent(BphPQUANT)
        BphP_segformer = inherit_segformer_class_from_parent(BphPQUANT)
        
    if args.input_type == 'images':
        in_channels = 32
    elif args.input_type == 'features':
        in_channels = 12
    if args.gt_type == 'binary':
        out_channels = 2
    elif args.gt_type == 'regression':
        out_channels = 1

    (train_loader, val_loader, test_loader, dataset, train_dataset, test_dataset, Y_mean, normalise_y, normalise_x) = create_dataloaders(
        args.root_dir, args.input_type, args.gt_type, args.input_normalisation, 
        args.batch_size, config
    )
    
    wandb.login()
    # some boilderplate code used by all models, written as lambdas for brevity
    init_wabdb = lambda arg, model : WandbLogger(
        project='BphPSEG', name=model, save_code=True, reinit=True
    ) if arg else None
    
    get_trainer = lambda args : pl.Trainer.from_argparse_args(
        args, log_every_n_steps=1, check_val_every_n_epoch=1, accelerator='gpu',
        devices=1, max_epochs=args.epochs, deterministic=True, logger=wandb_log
    )

    if args.model == 'mlp':
        wandb_log = init_wabdb(args.wandb_log, 'mlp_'+args.input_type+'_'+args.gt_type)
        trainer = get_trainer(args)
        model = MLP(
            in_channels=in_channels, out_channels=out_channels, 
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash, seed=seed
        )
        if not args.dropout:
            amf.remove_dropout(model)
        if not args.batchnorm:
            amf.remove_batchnorm(model)
        print(model)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
    
    if args.model == 'Unet':
        wandb_log = init_wabdb(args.wandb_log, 'Unet_'+args.input_type+'_'+args.gt_type)
        trainer = get_trainer(args)
        model = Unet(
            smp.Unet(
                encoder_name='resnet101', encoder_weights='imagenet',
                in_channels=in_channels, classes=out_channels,
            ),
            in_channels=in_channels, out_channels=out_channels, 
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash, seed=seed
        )
        if not args.dropout:
            amf.remove_dropout(model.net)
        if not args.batchnorm:
            amf.remove_batchnorm(model.net)
        print(model.net)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        
    elif args.model == 'UnetPlusPlus':
        wandb_log = init_wabdb(args.wandb_log, 'Unet_pluplus_'+args.input_type+'_'+args.gt_type)
        trainer = get_trainer(args)
        model = UnetPlusPlus(
            smp.UnetPlusPlus(
                encoder_name='resnet101', encoder_depth=5, encoder_weights='imagenet',
                in_channels=in_channels, classes=out_channels,
                decoder_use_batchnorm=True
            ),
            in_channels=in_channels, out_channels=out_channels,
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash, seed=seed
        )
        if not args.dropout:
            amf.remove_dropout(model.net)
        if not args.batchnorm:
            amf.remove_batchnorm(model.net)
        print(model.net)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        
    elif args.model == 'deeplabv3_resnet101':
        wandb_log = init_wabdb(args.wandb_log, 'deeplabv3_resnet101_'+args.input_type+'_'+args.gt_type)
        trainer = get_trainer(args)
        model = BphP_deeplabv3_resnet101(
            smp.DeepLabV3(
                encoder_name='resnet101', encoder_weights='imagenet',
                in_channels=in_channels, classes=out_channels
            ),
            in_channels=in_channels, out_channels=out_channels,
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash, seed=seed
        )
        if not args.dropout:
            amf.remove_dropout(model.net)
        if not args.batchnorm:
            amf.remove_batchnorm(model.net)
        print(model.net)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        
    elif args.model == 'segformer':
        wandb_log = init_wabdb(args.wandb_log, 'segformerb5_'+args.input_type+'_'+args.gt_type)
        trainer = get_trainer(args)
        model = BphP_segformer(
            SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640'),
            in_channels=in_channels, out_channels=out_channels,
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash, seed=seed
        )
        if not args.dropout:
            amf.remove_dropout(model.net)
        if not args.batchnorm:
            amf.remove_batchnorm(model.net)
        print(model.net)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        
    if args.save_test_example:
        model.eval()
        # test example idx. 6 is 'c143423.p31' when 42 is sampler seed
        (X, Y) = test_dataset[6]
        Y_hat = model.forward(X.unsqueeze(0)).squeeze()
        (fig, ax) = dataset.plot_sample(X, Y, Y_hat, y_transform=normalise_y, x_transform=normalise_x)
        if args.wandb_log:
            wandb.log({'test_example': wandb.Image(fig)})