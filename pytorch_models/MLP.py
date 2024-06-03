import torch
from torch import nn
from typing import Optional
import copy
import wandb
from pytorch_models.BphPQUANT import BphPQUANT


def inherit_mlp_class_from_parent(parent_class):

    class MLP(parent_class):
        
        def __init__(
                self,
                in_channels : int, # : number of input channels
                out_channels : int, # : number of output channels (segmentation classes)
                *args, **kwargs
            ):
            
            super(MLP, self).__init__(*args, **kwargs)
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            
            self.layer1 = nn.Sequential(
                nn.Linear(in_channels, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.fc3 = nn.Linear(512, out_channels)
        
        def forward(self, x):
            x = x.transpose(1, 3)
            shape = x.shape
            x = x.view(-1, self.in_channels)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.fc3(x)
            
            x = x.view(shape[0], shape[1], shape[2], self.out_channels)
            x = x.transpose(1, 3)
            return x
                
    return MLP