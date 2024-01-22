import torch
import logging
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from sklearn.model_selection import KFold
from custom_pytorch_utils.custom_transforms import Normalise, ReplaceNaNWithZero, \
    BinaryMaskToLabel
from preprocessing.sample_train_val_test_sets import *
from torchvision import transforms
from custom_pytorch_utils.custom_focal_loss import CrossEntropyLoss, FocalLoss
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, \
    BinaryPrecision, BinaryRecall, MatthewsCorrCoef, JaccardIndex, Dice, \
    BinarySpecificity, BinaryConfusionMatrix
    
from torchmetrics.regression import ExplainedVariance, R2Score, \
    MeanAbsolutePercentageError, MeanSquaredError


class DoubleConv(pl.LightningModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(pl.LightningModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(pl.LightningModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(pl.LightningModule):
    def __init__(self, in_channels : int, out_channels :int, gt_type : str,
                 y_transform=None, bilinear=False, scale=1,
                 weight_true=1.0, weight_false=1.0, y_mean=0.0):
        # in_channels, out_channels are input and output channels respectively
        # bilinear: whether to use bilinear interpolation or transposed convolutions
        # scale downsizes the number of channels in the network by a factor of 'scale'
        # scale is useful for reducing the number of parameters in the network during debugging
        
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.y_transform = y_transform
        
        if gt_type not in ['binary', 'regression']:
            # use binary classification or value regression
            raise ValueError("gt_type must be either 'binary' or 'regression'")
        self.gt_type = gt_type
        
        if gt_type == 'binary':
            self.loss = CrossEntropyLoss(
                weight=torch.tensor([weight_false, weight_true]).to(device='cuda')
            )
            #self.loss = FocalLoss(
                #weight=torch.tensor([weight_false, weight_true]).to(device='cuda')
            #)
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
                ('Recall', self.recall), 
                ('Precision', self.PPV),
                ('Specificity', self.Specificity),
                ('MCC', self.MCC),
                ('IOU', self.IoU),
                ('Dice', self.dice)
            ]
        else:
            self.loss = F.mse_loss
            self.EVS = ExplainedVariance().to(device='cuda')
            self.MSE = MeanSquaredError().to(device='cuda')
            self.percent_error = MeanAbsolutePercentageError().to(device='cuda')
            # For R_sqr score
            self.SSres, self.SStot, self.y_mean = 0.0, 0.0, y_mean
            self.metrics = [
                ('EVS', self.EVS),
                ('MSE', self.MSE),
                ('percent_error', self.percent_error)
            ]
        self.bilinear = bilinear  
        
        self.inc = DoubleConv(in_channels, 64//scale)
        self.down1 = Down(64//scale, 128//scale)
        self.down2 = Down(128//scale, 256//scale)
        self.down3 = Down(256//scale, 512//scale)
        factor = 2 if bilinear else 1
        self.down4 = Down(512//scale, (1024//scale) // factor)
        self.up1 = Up(1024//scale, (512//scale) // factor, bilinear)
        self.up2 = Up(512//scale, (256//scale) // factor, bilinear)
        self.up3 = Up(256//scale, (128//scale) // factor, bilinear)
        self.up4 = Up(128//scale, (64//scale), bilinear)
        self.outc = OutConv(64//scale, out_channels)
        
        self.save_hyperparameters()
    
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        if self.gt_type == 'binary':
            loss = self.loss(y_hat, y)
            y = y.to(dtype=torch.long) # dice metric only accepts long type
        else:            
            y_hat = y_hat.squeeze() # remove singleton channel dimension
            loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.gt_type == 'binary':
            loss = self.loss(y_hat, y)
            y = y.to(dtype=torch.long) # dice metric only accepts long type
        else:
            y_hat = y_hat.squeeze() # remove singleton channel dimension
            loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        if self.y_transform:
            # invert the linear transform to get the original values (cpu only)
            y = y.to(device='cpu')
            y_hat = y_hat.to(device='cpu')
            y = self.y_transform.inverse(y)
            y_hat = self.y_transform.inverse(y_hat)
            y = y.to(device='cuda')
            y_hat = y_hat.to(device='cuda')
        if self.gt_type == 'regression':
            y_hat, y = y_hat.view(-1), y.view(-1) # <- regression metrics require 1D tensors
        else:
            y_hat = torch.argmax(y_hat, dim=-3) # <- convert logits to class labels
            y = torch.argmax(y, dim=-3)
        for metric_name, metric in self.metrics:
            self.log(f'val_{metric_name}', metric(y_hat, y))
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.gt_type == 'binary':
            loss = self.loss(y_hat, y)
            y = y.to(dtype=torch.long) # dice metric has a hissy fit if target is float
        else:
            y_hat = y_hat.squeeze() # remove singleton channel dimension
            loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        if self.y_transform:
            # invert the linear transform to get the original values (cpu only)
            y = y.to(device='cpu')
            y_hat = y_hat.to(device='cpu')
            y = self.y_transform.inverse(y)
            y_hat = self.y_transform.inverse(y_hat)
            y = y.to(device='cuda')
            y_hat = y_hat.to(device='cuda')
        if self.gt_type == 'regression':
            y_hat, y = y_hat.view(-1), y.view(-1) # <- regression metrics require 1D tensors
        else:
            y_hat = torch.argmax(y_hat, dim=-3) # <- convert logits to class labels
            y = torch.argmax(y, dim=-3)
        for metric_name, metric in self.metrics:
            self.log(f'test_{metric_name}', metric(y_hat, y))
        if self.gt_type == 'binary':
            # accumulate confusion matrix over batches
            self.accumalate_confusion.append(self.confusion_matrix(y_hat, y))
        else: # regression
            # accumulate residuals over batches
            self.SSres += torch.sum((y - y_hat)**2)
            self.SStot += torch.sum((y - self.y_mean)**2)
        return loss
    
    def on_test_epoch_end(self):
        # manually accumulate confusion matrix over batches
        if self.gt_type == 'binary':
            self.accumalate_confusion = torch.stack(self.accumalate_confusion, dim=0)
            self.accumalate_confusion = torch.sum(self.accumalate_confusion, dim=0)
            print(f'[[TN, FP],[FN, TP]] = {self.accumalate_confusion}')
            self.accumalate_confusion = []
        else: # regression / pixel-level prediction
            R2Score = 1 - (self.SSres / self.SStot)
            print(f'R2Score={R2Score}')
            self.log('test_R2Score', R2Score)
            self.SSres, self.SStot = 0.0, 0.0
        
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=13)
        parser.add_argument('--out_channels', type=int, default=2)
        return parser
        
    

def train_UNet_main():
    pl.seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data_path', type=str, default='preprocessing/20231212_homogeneous_cylinders/dataset.h5')
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

    
    (train_loader, val_loader, test_loader, Y_mean, normalise_y, _, _, dataset) = get_torch_train_val_test_sets(
        args.data_path,
        'binary',
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=args.batch_size
    )
    
    model = UNet(
        args.in_channels, 
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
    dataset.plot_sample(0, model(dataset[0][0].unsqueeze(0)), save_name='c139519.p0_semantic_segmentation_epoch100.png')
    
    # REGRESSION / QUANTITATIVE SEGMENTATION    
    weight_true = 1.0
    weight_false = 1.0 
    
    trainer = pl.Trainer.from_argparse_args(
        args, check_val_every_n_epoch=1, accelerator='gpu', devices=1, max_epochs=args.epochs
    )
    
    # save instance to invert transform for testing
    
    (train_loader, val_loader, test_loader, Y_mean, normalise_y, _, _, dataset) = get_torch_train_val_test_sets(
        args.data_path,
        'regression',
        train_val_test_split=[0.8, 0.1, 0.1],
        batch_size=args.batch_size
    )
    
    model = UNet(
        args.in_channels, 
        1, 
        'regression',
        y_transform=normalise_y, 
        weight_false=weight_false,
        weight_true=weight_true
    )

    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(model, test_loader)
    
    # visualise the results
    dataset.plot_sample(
        0, 
        model(dataset[0][0].unsqueeze(0)), 
        save_name='c139519.p0_quantitative_segmentation_epoch100.png',
        y_transform=normalise_y
    )
    
    print(result)
    
    dataset.plot_sample(0, model(dataset[0][0].unsqueeze(0)), save_name='c139519.p0_semantic_segmentation_epoch100.png')
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('highest')
    train_UNet_main()