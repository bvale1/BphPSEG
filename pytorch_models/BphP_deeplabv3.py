import torch, argparse, wandb
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional
from pytorch_models.BphPQUANT import BphPQUANT
from pytorch_models.BphPSEG import BphPSEG
from custom_pytorch_utils.custom_focal_loss import CrossEntropyLoss
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, \
    BinaryPrecision, BinaryRecall, MatthewsCorrCoef, JaccardIndex, Dice, \
        BinarySpecificity, BinaryConfusionMatrix
from torchmetrics.regression import ExplainedVariance, \
    MeanSquaredError, MeanAbsoluteError

def inherit_deeplabv3_smp_resnet101_class_from_parent(parent_class):
    
    class BphP_deeplabv3_resnet101(parent_class):
        # the parent_class argument specifies which class to inherit from.
        # inheret from BphPSEG for semantic segmentation models and BphPQUANT for
        # quantification models of the Bacterial Phytochrome.
        def __init__(
                self, net, # : nn.Module : pretrained segformer model
                in_channels : int, # : number of input channels
                out_channels : int, # : number of output channels (segmentation classes)
                *args, **kwargs
            ):
            # scale is useful for reducing the number of parameters in the network during debugging
            
            super(BphP_deeplabv3_resnet101, self).__init__(*args, **kwargs)
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.net = net
        
            if issubclass(self.__class__, BphPQUANT):
                self.net.segmentation_head = nn.Sequential(
                    nn.Conv2d(256, 16, kernel_size=1, stride=1),
                    nn.UpsamplingBilinear2d(scale_factor=8),
                    nn.Conv2d(16, out_channels, kernel_size=1, stride=1)
                )
                
                #self.net.segmentation_head = nn.Sequential(
                #    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                #    nn.ReLU(inplace=True),
                #    nn.UpsamplingBilinear2d(scale_factor=2),
                #    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                #    nn.ReLU(inplace=True),
                #    nn.UpsamplingBilinear2d(scale_factor=2),
                #    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                #    nn.ReLU(inplace=True),
                #    nn.UpsamplingBilinear2d(scale_factor=2),
                #    nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
                #)
                
        def forward(self, x):
            return self.net.forward(x)
        
    return BphP_deeplabv3_resnet101


# deprecated method
def inherit_deeplabv3_resnet101_class_from_parent(parent_class):
    # this used the torchvision.models.segmentation version of deeplabv3_resnet101
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
            
            if issubclass(self.__class__, BphPQUANT):
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
                
            if issubclass(self.__class__, BphPSEG):
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
            
            if issubclass(self.__class__, BphPQUANT):
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
            
            if issubclass(self.__class__, BphPSEG):
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
    

        @staticmethod
        def add_model_specific_args(parent_parser):
            parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
            parser.add_argument('--in_channels', type=int, default=32)
            parser.add_argument('--out_channels', type=int, default=2)
            parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
            parser.add_argument('--aux_loss_weight', type=float, default=0.3, help='weight of the auxillary loss set 0.0 to disable')
            return parser
        
    return BphP_deeplabv3_resnet101
                


class BphP_integrated_deeplabv3_resnet101(pl.LightningModule):
    # this implementation integrated the two models BphPSEG and BphPQUANT
    # into the same network. It is highly experimental and not recommended.
    # It uses the auxillery classifier for binary semantic 
    # segmentation and the main decoder output for pixel level prediction

    def __init__(
            self, 
            net : nn.Module, # pretrained deeplabv3_resnet101 model
            in_channels : int, # : number of input channels
            out_channels : int, # : number of output channels (segmentation classes)
            y_transform : nn.Module, # required to rescale the output to the ground truth
            y_mean : float, # mean of the ground truth
            wandb_log : Optional[wandb.sdk.wandb_run.Run] = None, # wandb logger
            git_hash : Optional[str] = None, # git hash of the current commit
            lr : Optional[float] = 1e-3, # learning rate
            aux_loss_weight : Optional[float] = 0.3, # : weight of the auxillary loss set 0.0 to disable
        ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Rethinking Atrous Convolution for Semantic Image Segmentation
        # https://arxiv.org/abs/1706.05587
        
        # main pixel level prediction metrics and loss
        self.main_loss = F.mse_loss
        self.EVS = ExplainedVariance().to(device='cuda')
        self.MSE = MeanSquaredError().to(device='cuda')
        self.MAE = MeanAbsoluteError().to(device='cuda')
        # For R_sqr score, pytorch R^2 does not accumulate correctly over batches
        self.SSres, self.SStot, self.y_mean = 0.0, 0.0, y_mean.to(device='cuda')
        self.main_metrics = [
            ('EVS', self.EVS),
            ('MSE', self.MSE),
            ('MAE', self.MAE)
        ]
        self.y_transform = y_transform
        
        # binary semantic segmentation metrics and loss
        
        self.aux_loss = CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(device='cuda'))
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
        self.aux_metrics = [
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
                256, 2, kernel_size=1, stride=1, padding=0, dilation=1, bias=True
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


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        
        loss = (self.main_loss(y_hat['out'], y['regression']) +
                self.aux_loss_weight*self.aux_loss(y_hat['aux'], y['binary']))
        
        if self.wandb_log:
            self.logger.experiment.log({'train_loss': loss}, step=self.trainer.global_step)
        return loss


    # override the validation step to include the auxillery loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        main_y_hat = y_hat['out']
        main_loss = self.main_loss(main_y_hat, y['regression'])
        
        # pixel level prediction:        
        # inverse transform the output and ground truth for properly scaled 
        # regression metrics
        main_y = y['regression'].to(device='cpu')
        main_y_hat = main_y_hat.to(device='cpu')
        main_y = self.y_transform.inverse(main_y)
        main_y_hat = self.y_transform.inverse(main_y_hat)
        main_y = main_y.to(device='cuda')
        main_y_hat = main_y_hat.to(device='cuda')
        # transfering between cpu and gpu slows down inference, but avoids
        # an error when calling the transform.inverse method. plz fix

        main_y_hat = main_y_hat.contiguous().view(-1)  # <- regression metrics require 1D tensors
        main_y = main_y.contiguous().view(-1)
        metrics_eval = {'main_val_loss' : main_loss}
        for metric_name, metric in self.main_metrics:
            metrics_eval[f'val_{metric_name}'] = metric(main_y_hat, main_y)
            
        # binary semantic segmentation:
        aux_y_hat = y_hat['aux']
        aux_loss = self.aux_loss_weight*self.aux_loss(aux_y_hat, y['binary'])
        aux_y = y['binary'].to(dtype=torch.long) # dice metric only accepts long type
        aux_y_hat = torch.argmax(aux_y_hat, dim=-3) # <- convert logits to class labels
        aux_y = torch.argmax(aux_y, dim=-3)
        
        metrics_eval['aux_val_loss'] = aux_loss
        for metric_name, metric in self.aux_metrics:
            metrics_eval[f'val_{metric_name}'] = metric(aux_y_hat, aux_y)
        
        if self.wandb_log:
            self.logger.experiment.log(metrics_eval, step=self.trainer.global_step)
            
        return main_loss + aux_loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        main_y_hat = y_hat['out']
        main_loss = self.main_loss(main_y_hat, y['regression'])
        
        # pixel level prediction:        
        # inverse transform the output and ground truth for properly scaled 
        # regression metrics
        main_y = y['regression'].to(device='cpu')
        main_y_hat = main_y_hat.to(device='cpu')
        main_y = self.y_transform.inverse(main_y)
        main_y_hat = self.y_transform.inverse(main_y_hat)
        main_y = main_y.to(device='cuda')
        main_y_hat = main_y_hat.to(device='cuda')
        # transfering between cpu and gpu slows down inference, but avoids
        # an error when calling the transform.inverse method. plz fix

        main_y_hat = main_y_hat.contiguous().view(-1)  # <- regression metrics require 1D tensors
        main_y = main_y.contiguous().view(-1)
        metrics_eval = {'main_test_loss' : main_loss}
        for metric_name, metric in self.main_metrics:
            metrics_eval[f'test_{metric_name}'] = metric(main_y_hat, main_y)
            
        # binary semantic segmentation:
        aux_y_hat = y_hat['aux']
        aux_loss = self.aux_loss_weight*self.aux_loss(aux_y_hat, y['binary'])
        aux_y = y['binary'].to(dtype=torch.long) # dice metric only accepts long type
        aux_y_hat = torch.argmax(aux_y_hat, dim=-3) # <- convert logits to class labels
        aux_y = torch.argmax(aux_y, dim=-3)
        
        metrics_eval['aux_test_loss'] = aux_loss
        for metric_name, metric in self.aux_metrics:
            metrics_eval[f'test_{metric_name}'] = metric(aux_y_hat, aux_y)
        
        if self.wandb_log:
            self.logger.experiment.log(metrics_eval, step=self.trainer.global_step)
        
        self.SSres += torch.sum((main_y - main_y_hat)**2)
        self.SStot += torch.sum((main_y - self.y_mean)**2)    
        self.accumalate_confusion.append(self.confusion_matrix(aux_y_hat, aux_y))
        
        return metrics_eval


    def test_epoch_end(self, outputs):
        # manually accumulate coefficient of determination over batches
        R2Score = 1 - (self.SSres / self.SStot)
        aggregate_metrics = {'test_R2Score' : R2Score}
        self.SSres, self.SStot = 0.0, 0.0
        
        # manually accumulate confusion matrix over batches
        self.accumalate_confusion = torch.stack(self.accumalate_confusion, dim=0)
        self.accumalate_confusion = torch.sum(self.accumalate_confusion, dim=0)
        print(f'confusion_matrix: {self.accumalate_confusion}')
        aggregate_metrics['confusion_matrix'] = f'[[TN, FP],[FN, TP]] = {self.accumalate_confusion}'
        self.accumalate_confusion = []
        
        for metric_name, _ in self.main_metrics + self.aux_metrics:
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
        parser.add_argument('--out_channels', type=int, default=1)
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--aux_loss_weight', type=float, default=0.3, help='weight of the auxillary loss')
        return parser
    