import torch, logging, wandb, argparse, json
import pytorch_lightning as pl
from preprocessing.sample_train_val_test_sets import *
from custom_pytorch_utils.custom_focal_loss import CrossEntropyLoss
from pytorch_models.Unet import inherit_unet_class_from_parent
from pytorch_models.BphP_deeplabv3 import inherit_deeplabv3_resnet101_class_from_parent
from pytorch_models.BphP_segformer import inherit_segformer_class_from_parent

from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, \
    BinaryPrecision, BinaryRecall, MatthewsCorrCoef, JaccardIndex, Dice, \
    BinarySpecificity, BinaryConfusionMatrix

from abc import abstractmethod
from typing import Optional, Union


def reset_weights(m):
    '''
    Reset all the parameters of a model no matter how deeply nested each
    layer is.
    '''
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
    else:
        for layer in m.children():
            reset_weights(layer)


class BphPSEG(pl.LightningModule): 
    '''
    BphPSEG is a parent module for training and testing semantic segmentation models.
    Each model inherits from this class and implements the architecture and forward pass.
    Methods for training, validation and testing are implemented in this class, 
    but can be overridden by child classes.
    '''    
    def __init__(
            self, 
            wandb_log : Optional[wandb.sdk.wandb_run.Run] = None, # wandb logger
            git_hash : Optional[str] = None, # git hash of the current commit
            lr : Optional[float] = 1e-3 # learning rate
        ):
        
        super().__init__()
        
        self.loss = CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device='cuda'))
        self.accuracy = BinaryAccuracy().to(device='cuda')
        self.f1 = BinaryF1Score().to(device='cuda')
        self.recall = BinaryRecall().to(device='cuda')
        self.PPV = BinaryPrecision().to(device='cuda') # naming it precision introduces a bug with pytorch lightning
        self.Specificity = BinarySpecificity().to(device='cuda')
        self.MCC = MatthewsCorrCoef(task='binary').to(device='cuda')
        self.IoU = JaccardIndex(task='binary').to(device='cuda')
        self.dice = Dice(average='micro').to(device='cuda')
        self.confusion_matrix = BinaryConfusionMatrix().to(device='cuda')
        self.accumalate_confusion = [] # manually accumulate confusion matrix over batches
        self.metrics = [
            ('Accuracy', self.accuracy), 
            ('F1', self.f1), 
            ('Recall', self.recall), # sensitivity, true positive rate
            ('Precision', self.PPV), # positive predictive value
            ('Specificity', self.Specificity), # true negative rate
            ('MCC', self.MCC), # Matthews correlation coefficient
            ('IOU', self.IoU), # Jaccard index
            ('Dice', self.dice)
        ]
        
        self.wandb_log = wandb_log
        self.git_hash = git_hash
        self.lr = lr
        
    
    @abstractmethod
    def forward(self, x):
        pass
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        
        loss = self.loss(y_hat, y)
        y = y.to(dtype=torch.long) # dice metric only accepts long type
        
        self.log('train_loss', loss)
        if self.wandb_log:
            self.wandb_log.log({'train_loss': loss})
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        y = y.to(dtype=torch.long) # dice metric only accepts long type
        self.log('val_loss', loss)
        
        y_hat = torch.argmax(y_hat, dim=-3) # <- convert logits to class labels
        y = torch.argmax(y, dim=-3)
        
        for metric_name, metric in self.metrics:
            self.log(f'val_{metric_name}', metric(y_hat, y))
        if self.wandb_log:
            self.wandb_log.log({f'val_loss': loss})
            for metric_name, metric in self.metrics:
                self.wandb_log.log({f'val_{metric_name}': metric(y_hat, y)})
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        
        y = y.to(dtype=torch.long) # dice metric has a hissy fit if target is float
        self.log('test_loss', loss)
        
        y_hat = torch.argmax(y_hat, dim=-3) # <- convert logits to class labels
        y = torch.argmax(y, dim=-3)
        
        for metric_name, metric in self.metrics:
            self.log(f'test_{metric_name}', metric(y_hat, y))
        if self.wandb_log:
            self.wandb_log.log({f'test_loss': loss})
            for metric_name, metric in self.metrics:
                self.wandb_log.log({f'test_{metric_name}': metric(y_hat, y)})
        # accumulate confusion matrix over batches
        self.accumalate_confusion.append(self.confusion_matrix(y_hat, y))
        return loss
    
    
    def on_test_epoch_end(self):
        # manually accumulate confusion matrix over batches
        self.accumalate_confusion = torch.stack(self.accumalate_confusion, dim=0)
        self.accumalate_confusion = torch.sum(self.accumalate_confusion, dim=0)
        print(f'[[TN, FP],[FN, TP]] = {self.accumalate_confusion}')
        self.accumalate_confusion = []
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=32)
        parser.add_argument('--out_channels', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        return parser
    
            

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('high')
    # causes cuDNN to deterministically select an algorithm, 
    # possibly at the cost of reduced performance
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--root_dir', type=str, default='/mnt/f/cluster_MSOT_simulations/Bphp_phantom/', help='path to the root directory of the raw data')
    parser.add_argument('--git_hash', type=str, default='None', help='git hash of the current commit')
    parser.add_argument('--model', type=str, default='segformer', help='choose from [Unet, deeplabv3_resnet101, segformer, all]')
    parser.add_argument('--wandb_log', type=bool, default=True, help='log to wandb')
    parser.add_argument('--results_json', type=str, help='path to file for logging experirment results', default='/home/wv00017/20240229_lr1em4_fine_time_results.json')
    
    parser = pl.Trainer.add_argparse_args(parser)
    
    Unet = inherit_unet_class_from_parent(BphPSEG)
    BphP_deeplabv3_resnet101 = inherit_deeplabv3_resnet101_class_from_parent(BphPSEG)
    BphP_segformer = inherit_segformer_class_from_parent(BphPSEG)
    
    # cannot add args from all models at once, as they have the same names
    parser = BphP_deeplabv3_resnet101.add_model_specific_args(parser)
    
    args = parser.parse_args()
    results = {'args' : vars(args)}
    
    # BINARY CLASSIFICATION / SEMANTIC SEGMENTATION
    
    (train_loader, val_loader, test_loader, _, _, dataset) = get_raw_image_torch_train_val_test_sets(
        args.root_dir,
        'binary',
        n_images=32,
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=args.batch_size
    )
    
    wandb.login()
    init_wabdb = lambda arg, model : wandb.init(project='BphPSEG', name=model) if arg else None
    
    if args.model == 'Unet':
        models = {'Unet' : True, 'deeplabv3_resnet101' : False, 'segformer' : False}
        
    elif args.model == 'deeplabv3_resnet101':
        models = {'Unet' : False, 'deeplabv3_resnet101' : True, 'segformer' : False}
        
    elif args.model == 'segformer':
        models = {'Unet' : False, 'deeplabv3_resnet101' : False, 'segformer' : True}
        
    elif args.model == 'all':
        models = {'Unet' : True, 'deeplabv3_resnet101' : True, 'segformer' : True}
        
    
    else:
        raise ValueError(f'unknown model: {args.model}, choose from [Unet, deeplabv3_resnet101, segformer, all]')
    
    
        
    if models['Unet']:
        trainer = pl.Trainer.from_argparse_args(
            args, log_every_n_steps=1, check_val_every_n_epoch=1, accelerator='gpu',
            devices=1, max_epochs=args.epochs
        )
        from train_Unet_DL import UNet
        wandb_log = init_wabdb(args.wandb_log, 'Unet')
        model = UNet(
            args.in_channels, 
            args.out_channels, 
            'binary'
        )
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        results['Unet'] = result
        
    if models['deeplabv3_resnet101']:
        trainer = pl.Trainer.from_argparse_args(
            args, log_every_n_steps=1, check_val_every_n_epoch=1, accelerator='gpu',
            devices=1, max_epochs=args.epochs
        )
        from torchvision.models.segmentation import deeplabv3_resnet101
        wandb_log = init_wabdb(args.wandb_log, 'deeplabv3_resnet101')
        model = BphP_deeplabv3_resnet101(
            deeplabv3_resnet101(weights='DEFAULT'),
            args.in_channels, args.out_channels,
            wandb_log=wandb_log, git_hash=args.git_hash
        )
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        results['deeplabv3_resnet101'] = result
    
    if models['segformer']:
        trainer = pl.Trainer.from_argparse_args(
            args, log_every_n_steps=1, check_val_every_n_epoch=1, accelerator='gpu',
            devices=1, max_epochs=args.epochs
        )
        from transformers import SegformerForSemanticSegmentation
        wandb_log = init_wabdb(args.wandb_log, 'segformer')
        model = BphP_segformer(
            SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b0-finetuned-ade-512-512'),
            args.in_channels, args.out_channels, wandb_log, args.git_hash
        )
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        results['segformer'] = result
        
    
    print(results)
    if args.wandb_log:
        wandb.summary.update(results)
        wandb_log.finish()
        
    if args.results_json:
        try:
            with open(args.results_json, 'w') as f:
                json.dump(results, f, indent=4)
        except:
            logging.error(f'failed to write results to {args.results_json}, invalid path?')
    
    
    # TODO: fix inference 'model(dataset[0][0].unsqueeze(0))'
    # visualise the results
    #dataset.get_config(0)
    #dataset.plot_sample(0, model(dataset[0][0].unsqueeze(0))['out'], save_name=f'c139519.p0_{args.model}_semantic_segmentation_epoch100.png')
    