import torch
from torch import nn
from typing import Optional
import copy
import wandb


def inherit_segformer_class_from_parent(parent_class):

    class BphP_segformer(parent_class):
        
        def __init__(
                self,
                net, # : nn.Module : pretrained segformer model
                in_channels : int, # : number of input channels
                out_channels : int, # : number of output channels (segmentation classes)
                *args, **kwargs
            ):
            
            super(BphP_segformer, self).__init__(*args, **kwargs)
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.net = net
            
            '''
            @article{xie2021segformer,
            title={SegFormer: Simple and efficient design for semantic segmentation with transformers},
            author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
            journal={Advances in Neural Information Processing Systems},
            volume={34},
            pages={12077--12090},
            year={2021}
            }
            '''
            
            with torch.no_grad():
                # augment the input channels of the first layer
                proj = self.net.segformer.encoder.patch_embeddings[0].proj
            
                kernel_size = copy.deepcopy(proj.kernel_size)
                stride = copy.deepcopy(proj.stride)
                padding = copy.deepcopy(proj.padding)
                weight = copy.deepcopy(proj.weight)
                weight = torch.sum(weight, dim=1, keepdim=True)
                weight = weight.repeat(1, in_channels, 1, 1)
                bias = copy.deepcopy(proj.bias.data)
                
                proj = nn.Conv2d(
                    in_channels, 32, kernel_size, stride, padding
                )
                proj.weight.data = weight
                proj.bias.data = bias
                self.net.segformer.encoder.patch_embeddings[0].proj = proj
                
                # augment output channels of the last layer and re initialise the weights
                classifier = self.net.decode_head.classifier
                
                kernel_size = copy.deepcopy(classifier.kernel_size)
                stride = copy.deepcopy(classifier.stride)
                padding = copy.deepcopy(classifier.padding)
                
                classifier = nn.Conv2d(
                    256, out_channels, kernel_size, stride, padding
                )
                self.net.decode_head.classifier = classifier
                
            
        def forward(self, x):
            logits = self.net.forward(x).logits
            return nn.functional.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            
            
    return BphP_segformer

