import torch
import logging
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from sklearn.model_selection import KFold
from custom_pytorch_utils.custom_datasets import BphP_MSOT_Dataset
from custom_pytorch_utils.custom_transforms import Normalise, ReplaceNaNWithZero, \
    BinaryMaskToLabel
from torchvision import transforms
from custom_pytorch_utils.custom_focal_loss import CrossEntropyLoss, FocalLoss
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, \
    BinaryPrecision, BinaryRecall, MatthewsCorrCoef, JaccardIndex, Dice
from torchmetrics.regression import ExplainedVariance, R2Score, \
    MeanAbsolutePercentageError


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
                 y_transform=None, bilinear=False, scale=1, weight=torch.tensor([1.0, 1.0])):
        # in_channels, out_channels are input and output channels respectively
        # bilinear: whether to use bilinear interpolation or transposed convolutions
        # scale downsizes the number of channels in the network by a factor of 'scale'
        # scale is useful for reducing the number of parameters in the network during debugging
        
        super(UNet, self).__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.y_transform = y_transform
        if gt_type not in ['binary', 'regression']:
            # use binary classification or value regression
            raise ValueError("gt_type must be either 'binary' or 'regression'")
        self.gt_type = gt_type
        self.weight = weight.to(device='cuda')
        if gt_type == 'binary':
            self.loss = CrossEntropyLoss(weight=self.weight)
            self.accuracy = BinaryAccuracy()
            self.f1 = BinaryF1Score()
            self.recall = BinaryRecall()
            self.PPV = BinaryPrecision() # naming it precision introduces a bug with pytorch lightning
            self.MCC = MatthewsCorrCoef(task='binary') 
            self.IoU = JaccardIndex(task='binary')
            self.dice = Dice(average='micro')
            self.metrics = [
                ('Accuracy', self.accuracy), 
                ('F1', self.f1), 
                ('Recall', self.recall), 
                ('Precision', self.PPV),
                ('MCC', self.MCC),
                ('IOU', self.IoU),
                ('Dice', self.dice)
            ]
        else:
            self.loss = F.mse_loss
            self.EVS = ExplainedVariance()
            self.R2 = R2Score()
            self.percent_error = MeanAbsolutePercentageError()
            self.metrics = [
                ('EVS', self.EVS),
                ('R2', self.R2),
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
            y = y.to(dtype=torch.long) # dice metric has a hissy fit if target is float
        else:
            loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.gt_type == 'binary':
            loss = self.loss(y_hat, y)
            y = y.to(dtype=torch.long) # dice metric has a hissy fit if target is float
        else:
            loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        if self.y_transform:
            y = self.y_transform.inverse(y)
            y_hat = self.y_transform.inverse(y_hat)
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
            loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        if self.y_transform:
            y = self.y_transform.inverse(y)
            y_hat = self.y_transform.inverse(y_hat)
        for metric_name, metric in self.metrics:
            self.log(f'test_{metric_name}', metric(y_hat, y))
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=12)
        parser.add_argument('--out_channels', type=int, default=2)
        return parser
        

def get_dataset_mins_maxs(dataset):
    # Used to normalise the features of the dataset, also computes the class
    # weights for the binary classification task
    (X1, Y1) = dataset[0]
    X_max = torch.empty(
        (
            dataset.__len__(),
            X1.shape[0],
            X1.shape[1],
            X1.shape[2]
        ), 
        dtype=torch.float32,
        requires_grad=False
    )
    Y_max = torch.empty(
        (dataset.__len__(), Y1.shape[0], Y1.shape[1]),
        dtype=torch.float32,
        requires_grad=False
    )
    
    for i in range(len(dataset)): # load entire dataset into memory, not ideal
        # TODO : calculate mean and std of each channel with rolling mean and
        #        std to save memory
        (X_max[i], Y_max[i]) = dataset[i]
        
    X_max = torch.transpose(X_max, 0, 1)
    X_max = torch.flatten(X_max, start_dim=1, end_dim=-1)
    X_min = X_max.min(dim=1).values
    X_max = X_max.max(dim=1).values
    
    N_true = torch.sum(Y_max)
    N_false = torch.sum(torch.logical_not(Y_max))
    weights = torch.tensor([N_true, N_false], dtype=torch.float32) / (N_false + N_true)
    Y_max = torch.flatten(Y_max)
    Y_min = Y_max.min()
    Y_max = Y_max.max()    
    
    
    return (X_max, X_min, Y_max, Y_min, weights)
    

def train_UNet_main():
    pl.seed_everything(42)
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data_path', type=str, default='20231212_homogeneous_cylinders/dataset.h5')
    parser.add_argument('--git_hash', type=str, default='None')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = UNet.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    trainer = pl.Trainer.from_argparse_args(
        args, check_val_every_n_epoch=1, accelerator='gpu', devices=1, max_epochs=args.epochs
    )
    
    (X_max, X_min, Y_max, Y_min, weights) = get_dataset_mins_maxs(
        BphP_MSOT_Dataset(
            args.data_path, 
            'regression',
            x_transform=transforms.Compose([ReplaceNaNWithZero()]),
            y_transform=transforms.Compose([ReplaceNaNWithZero()])
        )
    )
    dataset = BphP_MSOT_Dataset(
        args.data_path, 
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
    
    # BINARY CLASSIFICATION / SEMANTIC SEGMENTATION
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [0.8, 0.1, 0.1], # 10% validation set
        generator=torch.Generator().manual_seed(42) # reproducible results
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = UNet(args.in_channels, args.out_channels, 'binary', weight=weights)
    
    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(model, test_loader)
    
    print(result)
    
    # REGRESSION / QUANTITATIVE SEGMENTATION
    
    # save instance to invert transform for testing
    normalise_y = Normalise(Y_max, Y_min) 
    dataset = BphP_MSOT_Dataset(
        args.data_path, 
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
        train_dataset, 
        [0.8, 0.1, 0.1] # 10% validation set
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = UNet(args.in_channels, 1, 'regression', y_transform=normalise_y)

    trainer.fit(model, train_loader, val_loader, max_epochs=args.epochs)
    result = trainer.test(model, test_loader)
    
    print(result)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('medium')
    train_UNet_main()