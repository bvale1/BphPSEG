import torch, wandb, argparse
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchmetrics.regression import ExplainedVariance, \
    MeanSquaredError, MeanAbsoluteError
from typing import Optional
from abc import abstractmethod


class BphPQUANT(pl.LightningModule): 
    '''
    BphPQUANT is a parent module for training and testing pixel level prediction models.
    Each model inherits from this class and implements the architecture and forward pass.
    Methods for training, validation and testing are implemented in this class, 
    but can be overridden by child classes.
    '''    
    def __init__(
            self,
            y_transform : nn.Module, # required to rescale the output to the ground truth
            y_mean : float, # mean of the ground truth
            wandb_log : Optional[wandb.sdk.wandb_run.Run] = None, # wandb logger
            git_hash : Optional[str] = None, # git hash of the current commit
            lr : Optional[float] = 1e-3 # learning rate
        ):
        
        super().__init__()
        
        self.loss = F.mse_loss
        self.EVS = ExplainedVariance().to(device='cuda')
        self.MSE = MeanSquaredError().to(device='cuda')
        self.MAE = MeanAbsoluteError().to(device='cuda')
        # For R_sqr score, pytorch R^2 does not accumulate correctly over batches
        self.SSres, self.SStot, self.y_mean = 0.0, 0.0, y_mean.to(device='cuda')
        self.metrics = [
            ('EVS', self.EVS),
            ('MSE', self.MSE),
            ('MAE', self.MAE)
        ]
        
        self.y_transform = y_transform
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
        
        if self.wandb_log:
            self.logger.experiment.log({'train_loss': loss}, step=self.trainer.global_step)
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        
        loss = self.loss(y_hat, y)
        
        # inverse transform the output and ground truth for properly scaled 
        # regression metrics
        y = y.to(device='cpu')
        y_hat = y_hat.to(device='cpu')
        y = self.y_transform.inverse(y)
        y_hat = self.y_transform.inverse(y_hat)
        y = y.to(device='cuda')
        y_hat = y_hat.to(device='cuda')
        # transfering between cpu and gpu slows down inference, but avoids
        # an error when calling the transform.inverse method. plz fix
        
        y_hat = y_hat.contiguous().view(-1)  # <- regression metrics require 1D tensors
        y = y.contiguous().view(-1)
        
        metrics_eval = {'val_loss' : loss}
        for metric_name, metric in self.metrics:
            metrics_eval[f'val_{metric_name}'] = metric(y_hat, y)
        if self.wandb_log:
            self.logger.experiment.log(metrics_eval, step=self.trainer.global_step)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        
        # inverse transform the output and ground truth for properly scaled 
        # regression metrics
        y = y.to(device='cpu')
        y_hat = y_hat.to(device='cpu')
        y = self.y_transform.inverse(y)
        y_hat = self.y_transform.inverse(y_hat)
        y = y.to(device='cuda')
        y_hat = y_hat.to(device='cuda')
        # transfering between cpu and gpu slows down inference, but avoids
        # an error when calling the transform.inverse method. plz fix
        
        y_hat = y_hat.contiguous().view(-1)  # <- regression metrics require 1D tensors
        y = y.contiguous().view(-1)
        
        metrics_eval = {'test_loss' : loss}
        for metric_name, metric in self.metrics:
            metrics_eval[f'test_{metric_name}'] = metric(y_hat, y)
        if self.wandb_log:
            self.logger.experiment.log(metrics_eval, step=self.trainer.global_step)
        # accumulate confusion matrix over batches
        self.SSres += torch.sum((y - y_hat)**2)
        self.SStot += torch.sum((y - self.y_mean)**2)
        return metrics_eval
    
    
    def test_epoch_end(self, outputs):
        
        # manually accumulate coefficient of determination over batches
        R2Score = 1 - (self.SSres / self.SStot)
        aggregate_metrics = {'test_R2Score' : R2Score}
        self.SSres, self.SStot = 0.0, 0.0
        for metric_name, _ in self.metrics:
            aggregate_metrics[f'average_test_{metric_name}'] = torch.stack(
                [x[f'test_{metric_name}'] for x in outputs]
            ).mean()
        if self.wandb_log:
            self.logger.experiment.log(aggregate_metrics, step=self.trainer.global_step)
        print(f'average_test_metrics: {aggregate_metrics}')
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)   
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=32)
        parser.add_argument('--out_channels', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        return parser

