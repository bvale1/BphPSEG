import torch, logging, wandb, argparse, json, os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_models.BphPQUANT import BphPQUANT
from pytorch_models.Unet import inherit_unet_class_from_parent
from pytorch_models.Unet import inherit_unet_pretrained_class_from_parent
from pytorch_models.BphP_deeplabv3 import inherit_deeplabv3_resnet101_class_from_parent
from pytorch_models.BphP_segformer import inherit_segformer_class_from_parent
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
from torchvision.models.segmentation import deeplabv3_resnet101
from torch.utils.data import DataLoader, random_split
from custom_pytorch_utils.custom_transforms import *
from custom_pytorch_utils.custom_datasets import BphP_MSOT_Dataset
import torchvision.transforms as transforms
import custom_pytorch_utils.augment_models_func as amf


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('high')
    
    # force cuDNN to deterministically select algorithms, 
    # improves reproducability but reduces performance
    
    # setting the deterministic flag broke the models on my machine so is
    # currently set to false
    torch.use_deterministic_algorithms(False)
    print(f'cuDNN deterministic: {torch.torch.backends.cudnn.deterministic}')
    print(f'cuDNN benchmark: {torch.torch.backends.cudnn.benchmark}')
    pl.seed_everything(42, workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--root_dir', type=str, default='preprocessing/20240305_homogeneous_cylinders/', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--model', type=str, default='all', help='choose from [Unet, UnetPlusPlus, deeplabv3_resnet101, segformer, all]')
    parser.add_argument('--wandb_log', type=bool, default=True, help='log to wandb')
    parser.add_argument('--input_type', type=str, default='images', help='choose from [images, features], input channels must be 12 if "images", and 32 if "features"')
    
    parser = pl.Trainer.add_argparse_args(parser)
    
    Unet = inherit_unet_class_from_parent(BphPQUANT)
    UnetPlusPlus = inherit_unet_pretrained_class_from_parent(BphPQUANT)
    BphP_deeplabv3_resnet101 = inherit_deeplabv3_resnet101_class_from_parent(BphPQUANT)
    BphP_segformer = inherit_segformer_class_from_parent(BphPQUANT)
    
    # cannot add args from all models at once, as they share the same arguments
    parser = BphP_deeplabv3_resnet101.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    # BINARY CLASSIFICATION / SEMANTIC SEGMENTATION
    with open(os.path.join(os.path.dirname(args.root_dir), 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    
    if args.input_type == 'images':
        in_channels = 32
        x_transform = transforms.Compose([
            ReplaceNaNWithZero(), 
            MaxMinNormalise(
                torch.Tensor(config['image_normalisation_params']['max']),
                torch.Tensor(config['image_normalisation_params']['min'])
            )
        ])
    elif args.input_type == 'features':
        in_channels = 12
        x_transform = transforms.Compose([
            ReplaceNaNWithZero(),
            MaxMinNormalise(
                torch.Tensor(config['feature_normalisation_params']['max']),
                torch.Tensor(config['feature_normalisation_params']['min'])
            )
        ])
    
    Y_mean = torch.Tensor(config['concentration_normalisation_params']['mean'])
    normalise_y = MaxMinNormalise(
        torch.Tensor(config['concentration_normalisation_params']['max']),
        torch.Tensor(config['concentration_normalisation_params']['min'])
    )
    dataset = BphP_MSOT_Dataset(
        args.root_dir, 
        'regression', 
        args.input_type, 
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
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # reproducible results
    )
    logging.info(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: \
        {len(test_dataset)}')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20
    )
    
    wandb.login()
    init_wabdb = lambda arg, model : WandbLogger(
        project='BphPQUANT', name=model, save_code=True, reinit=True
    ) if arg else None
    
    if args.model not in ['Unet', 'UnetPlusPlus', 'deeplabv3_resnet101', 'segformer', 'all']:
        raise ValueError(f'unknown model: {args.model}, choose from \
            [Unet, deeplabv3_resnet101, segformer, all]')
    
        
    if args.model == 'Unet' or args.model == 'all':
        wandb_log = init_wabdb(args.wandb_log, 'Unet_'+args.input_type+'_QUANT')
        trainer = pl.Trainer.from_argparse_args(
            args, log_every_n_steps=1, check_val_every_n_epoch=1, 
            accelerator='gpu', devices=1, max_epochs=args.epochs, logger=wandb_log
        )
        model = Unet(
            in_channels, 1,
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash
        )
        amf.remove_dropout(model.net)
        amf.remove_batchnorm(model.net)
        print(model.net)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        
    if args.model == 'UnetPlusPlus' or args.model == 'all':
        wandb_log = init_wabdb(args.wandb_log, 'Unet_pluplus_'+args.input_type+'_QUANT')
        trainer = pl.Trainer.from_argparse_args(
            args, log_every_n_steps=1, check_val_every_n_epoch=1, accelerator='gpu',
            devices=1, max_epochs=args.epochs, deterministic=True, logger=wandb_log
        )
        model = UnetPlusPlus(
            smp.UnetPlusPlus(
                encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet',
                in_channels=in_channels, classes=1,
                decoder_use_batchnorm=False, activation=None
            ),
            in_channels, 1, 
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash
        )
        amf.remove_dropout(model.net)
        amf.remove_batchnorm(model.net)
        print(model.net)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        
        
    if args.model == 'deeplabv3_resnet101' or args.model == 'all':
        wandb_log = init_wabdb(args.wandb_log, 'deeplabv3_resnet101_'+args.input_type+'_QUANT')
        trainer = pl.Trainer.from_argparse_args(
            args, log_every_n_steps=1, check_val_every_n_epoch=1,
            accelerator='gpu', devices=1, max_epochs=args.epochs, logger=wandb_log
        )        
        model = BphP_deeplabv3_resnet101(
            deeplabv3_resnet101(weights='DEFAULT'),
            in_channels, 1, aux_loss_weight=args.aux_loss_weight,
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash
        )
        amf.remove_dropout(model.net)
        amf.remove_batchnorm(model.net)
        print(model.net)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
    
    if args.model == 'segformer' or args.model == 'all':
        wandb_log = init_wabdb(args.wandb_log, 'segformer_'+args.input_type+'_QUANT')
        trainer = pl.Trainer.from_argparse_args(
            args, log_every_n_steps=1, check_val_every_n_epoch=1,
            accelerator='gpu', devices=1, max_epochs=args.epochs, logger=wandb_log
        )
        model = BphP_segformer(
            SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512'),
            in_channels, 1, 
            y_transform=normalise_y, y_mean=Y_mean,
            wandb_log=wandb_log, git_hash=args.git_hash
        )
        amf.remove_dropout(model.net)
        amf.remove_batchnorm(model.net)
        print(model.net)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
    
    
    # TODO: fix inference 'model(dataset[0][0].unsqueeze(0))'
    # visualise the results
    #dataset.get_config(0)
    #dataset.plot_sample(0, model(dataset[0][0].unsqueeze(0))['out'], save_name=f'c139519.p0_{args.model}_semantic_segmentation_epoch100.png')
    