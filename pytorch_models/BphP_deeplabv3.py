import torch
import torch.nn as nn
from typing import Optional
from pytorch_models.BphPQUANT import BphPQUANT
from pytorch_models.BphPSEG import BphPSEG


def inherit_deeplabv3_resnet101_class_from_parent(parent_class):

    class BphP_deeplabv3_resnet101(parent_class):
        
        def __init__(
                self, 
                net : nn.Module, # pretrained deeplabv3_resnet101 model
                in_channels : int, # : number of input channels
                out_channels : int, # : number of output channels (segmentation classes)
                aux_loss_weight : Optional[float] = 0.3, # : weight of the auxillary loss set 0.0 to disable
                *args, **kwargs
            ):
            super(BphP_deeplabv3_resnet101, self).__init__(*args, **kwargs)
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            
            # Rethinking Atrous Convolution for Semantic Image Segmentation
            # https://arxiv.org/abs/1706.05587
            
            self.net = net
            self.aux_loss_weight = aux_loss_weight
            
            with torch.no_grad():
                # augment the input channels of the first layer
                conv1_weight = self.net.backbone.conv1.weight.data.clone()
                conv1_weight = torch.sum(conv1_weight, dim=1, keepdim=True)
                conv1_weight = conv1_weight.repeat(1, in_channels, 1, 1)
                
                self.net.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
                self.net.backbone.conv1.weight = nn.Parameter(conv1_weight)
                #nn.init.kaiming_normal_(
                #    self.net.backbone.conv1.weight, mode='fan_out', nonlinearity='relu'
                #)
        
                # augment output channels of the last layer and re initialise the weights
                self.net.classifier[-1] = nn.Conv2d(
                    256, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True
                )
                nn.init.kaiming_normal_(
                    self.net.classifier[-1].weight, mode='fan_out', nonlinearity='relu'
                )
                # do the same for the auxillary classifier
                self.net.aux_classifier[-1] = nn.Conv2d(
                    256, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True
                )
                nn.init.kaiming_normal_(
                    self.net.aux_classifier[-1].weight, mode='fan_out', nonlinearity='relu'
                )
                

        def forward(self, x):
            # forward pass returns output and auxillery logits as a dictionary
            # see for details on auxillery loss: 
            # Going Deeper With Convolutions
            # Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
            # Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich;
            # Proceedings of the IEEE Conference on Computer Vision and Pattern
            # Recognition (CVPR), 2015, pp. 1-9
            return self.net.forward(x)
        
        
        # override the training step to include the auxillery loss
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.forward(x)
            
            loss = self.loss(y_hat['out'], y) + self.aux_loss_weight*self.loss(y_hat['aux'], y)
            
            if self.wandb_log:
                self.logger.experiment.log({'train_loss': loss}, step=self.trainer.global_step)
            return loss
        
        
        # override the validation step to include the auxillery loss
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.forward(x)['out'] # auxillary logits are not used in inference
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
        
        
        # override the test step to include the auxillery loss
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.forward(x)['out'] # auxillary logits are not used in inference
            loss = self.loss(y_hat, y)
            
            y = y.to(dtype=torch.long) # dice metric has a hissy fit if target is float
            
            y_hat = torch.argmax(y_hat, dim=-3) # <- convert logits to class labels
            y = torch.argmax(y, dim=-3)
            
            metrics_eval = {'test_loss' : loss}
            for metric_name, metric in self.metrics:
                metrics_eval[f'test_{metric_name}'] = metric(y_hat, y)
            if self.wandb_log:
                self.logger.experiment.log(metrics_eval, step=self.trainer.global_step)
            
            if issubclass(self.__class__, BphPSEG):
                # accumulate confusion matrix over batches
                self.accumalate_confusion.append(self.confusion_matrix(y_hat, y))
            
            # only used if parent is a regression model
            if issubclass(self.__class__, BphPQUANT):
                self.SSres += torch.sum((y - y_hat)**2)
                self.SStot += torch.sum((y - self.y_mean)**2)
            
            return metrics_eval
    
        
    return BphP_deeplabv3_resnet101