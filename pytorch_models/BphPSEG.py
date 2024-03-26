import torch, argparse, wandb
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, \
    BinaryPrecision, BinaryRecall, MatthewsCorrCoef, JaccardIndex, Dice, \
        BinarySpecificity, BinaryConfusionMatrix
from custom_pytorch_utils.custom_focal_loss import CrossEntropyLoss
from abc import abstractmethod
from typing import Optional


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
        
        self.save_hyperparameters()
        
    
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
        y = y.to(dtype=torch.long) # dice metric only accepts long type
        
        y_hat = torch.argmax(y_hat, dim=-3) # <- convert logits to class labels
        y = torch.argmax(y, dim=-3)
        
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
        
        y = y.to(dtype=torch.long) # dice metric has a hissy fit if target is float
        
        y_hat = torch.argmax(y_hat, dim=-3) # <- convert logits to class labels
        y = torch.argmax(y, dim=-3)
        
        metrics_eval = {'test_loss' : loss}
        for metric_name, metric in self.metrics:
            metrics_eval[f'test_{metric_name}'] = metric(y_hat, y)
        if self.wandb_log:
            self.logger.experiment.log(metrics_eval, step=self.trainer.global_step)
        self.accumalate_confusion.append(self.confusion_matrix(y_hat, y))
        return metrics_eval

    
    def test_epoch_end(self, outputs):
        aggregate_metrics = {}
        # manually accumulate confusion matrix over batches
        self.accumalate_confusion = torch.stack(self.accumalate_confusion, dim=0)
        self.accumalate_confusion = torch.sum(self.accumalate_confusion, dim=0)
        print(f'confusion_matrix: {self.accumalate_confusion}')
        aggregate_metrics = {
            'confusion_matrix' : f'[[TN, FP],[FN, TP]] = {self.accumalate_confusion}'
        }
        self.accumalate_confusion = []
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
    
            