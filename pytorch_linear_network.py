import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser


class LinearNetwork(pl.LightningModule):
    def __init__(self, n_channels, n_classes, gt_type):
        # n_channels, n_classes are input and output channels respectively
        # gt_type: either 'binary' or 'regression'
        
        super(UNet, self).__init__()
        self.save_hyperparameters()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.gt_type = gt_type
        
        ANN = nn.Sequential(
            nn.Linear(n_channels, 12),
            nn.Sigmoid(),
            nn.Linear(12, 12),
            nn.Sigmoid(),
            nn.Linear(12, n_classes)
        )
    def forward(self, x):
        return self.ANN(x)
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)
        if self.gt_type == 'binary':
            loss = F.cross_entropy(y_hat, y)
        elif self.gt_type == 'regression':
            loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        if self.gt_type == 'binary':
            loss = F.cross_entropy(y_hat, y)
        elif self.gt_type == 'regression':
            loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        if self.gt_type == 'binary':
            loss = F.cross_entropy(y_hat, y)
        elif self.gt_type == 'regression':
            loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=13)
        parser.add_argument('--out_channels', type=int, default=2)