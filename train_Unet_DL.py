import torch
import logging
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from sklearn.model_selection import KFold
from preprocessing.pytorch_dataset import BphP_MSOT_Dataset


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
    def __init__(self, in_channels, out_channels, bilinear=False, scale=1):
        # in_channels, out_channels are input and output channels respectively
        # bilinear: whether to use bilinear interpolation or transposed convolutions
        # scale downsizes the number of channels in the network by a factor of 'scale'
        # scale is useful for reducing the number of parameters in the network during debugging
        
        super(UNet, self).__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.out_channels = out_channels
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
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=13)
        parser.add_argument('--out_channels', type=int, default=2)
        

def train_UNet_main():
    pl.seed_everything(42)
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='20231127_homogeneous_cylinders')
    parser.add_argument('--git_hash', type=str, default='None')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = UNet.add_model_specific_args(parser)
    
    args = parser.parse_args()
    
    dataset = BphP_MSOT_Dataset(args.data_path, 'binary')
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        logging.info(f'Fold {i+1}/5')
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        train_dataset, val_dataset = random_split(train_dataset, [int(0.8*len(train_dataset)), int(0.2*len(train_dataset))])
        test_dataset = torch.utils.data.Subset(dataset, test_index)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = UNet(args.in_channels, args.out_channels)
        
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model, train_loader, val_loader)
        result = trainer.test(model, test_loader)
        
        print(result)
    
    
if __name__ == '__main__':
    train_UNet_main()