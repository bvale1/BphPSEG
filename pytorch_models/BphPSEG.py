import torch, argparse, wandb, json
import pytorch_lightning as pl
from custom_pytorch_utils.custom_focal_loss import CrossEntropyLoss
from custom_pytorch_utils.peformance_metrics import BinaryTestMetricCalculator
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
            lr : Optional[float] = 1e-3, # learning rate
            y_transform = None, # unused in semantic segmentation
            seed : int = None # seed for reproducibility
        ):
        
        super().__init__()
        
        self.loss = CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device='cuda'))
        self.test_metric_calculator_bg = BinaryTestMetricCalculator()
        self.test_metric_calculator_inclusion = BinaryTestMetricCalculator()
        
        self.wandb_log = wandb_log
        self.git_hash = git_hash
        self.lr = lr
        self.seed = seed
        self.save_hyperparameters(ignore=['net'])            
        
    
    @abstractmethod
    def forward(self, x):
        pass
    
    
    def training_step(self, batch, batch_idx):
        x, y, *_ = batch
        y_pred = self.forward(x)
        
        loss = self.loss(y_pred, y)
        
        if self.wandb_log:
            self.logger.experiment.log({'train_loss': loss}, step=self.trainer.global_step)
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        if self.wandb_log:
            self.logger.experiment.log({'val_loss': loss}, step=self.trainer.global_step)
        return loss
    
    
    def test_step(self, batch, batch_idx):
        x, y, bg_mask, inclusion_mask, sample_names = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)

        y_pred_cls = torch.argmax(y_pred, dim=-3)  # [b, 2, h, w] -> [b, h, w]
        y_gt = torch.argmax(y, dim=-3)             # [b, 2, h, w] -> [b, h, w]

        self.test_metric_calculator_bg(y_gt, y_pred_cls, sample_names, Y_mask=bg_mask)
        self.test_metric_calculator_inclusion(y_gt, y_pred_cls, sample_names, Y_mask=inclusion_mask)

        return {'test_loss': loss}

    
    def test_epoch_end(self, outputs):
        median_metrics_bg = self.test_metric_calculator_bg.get_median_metrics()
        all_metrics_bg = self.test_metric_calculator_bg.get_all_metrics()
        median_metrics_inclusion = self.test_metric_calculator_inclusion.get_median_metrics()
        all_metrics_inclusion = self.test_metric_calculator_inclusion.get_all_metrics()
        mean_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        print(f'test bg median metrics: {median_metrics_bg}')
        print(f'test inclusion median metrics: {median_metrics_inclusion}')

        if self.wandb_log:
            bg_log = {f'bg_{k}': v for k, v in median_metrics_bg.items()}
            inclusion_log = {f'inclusion_{k}': v for k, v in median_metrics_inclusion.items()}
            self.logger.experiment.log(
                {**bg_log, **inclusion_log, 'test_loss': mean_loss.item(),
                 'git_hash': self.git_hash, 'seed': self.seed},
                step=self.trainer.global_step
            )
            artifact = wandb.Artifact('test_per_sample_metrics', type='dataset')
            with artifact.new_file('bg.json', mode='w') as f:
                json.dump(all_metrics_bg, f)
            with artifact.new_file('inclusion.json', mode='w') as f:
                json.dump(all_metrics_inclusion, f)
            wandb.log_artifact(artifact)
        # reset for potential re-use
        self.test_metric_calculator_bg = BinaryTestMetricCalculator()
        self.test_metric_calculator_inclusion = BinaryTestMetricCalculator()
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
    
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--in_channels', type=int, default=32)
        parser.add_argument('--out_channels', type=int, default=2)
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        return parser
    
            