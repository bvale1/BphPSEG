from typing import Optional
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import copy


def inherit_unetplusplus_class_from_parent(parent_class):
    # create class for U-Net++.
    # the parent_class argument specifies which class to inherit from.
    # inheret from BphPSEG for semantic segmentation models and BphPQUANT for
    # quantification models of the Bacterial Phytochrome.
    
    class UNet_plusplus(parent_class):
        def __init__(
                self, net, # : nn.Module : pretrained segformer model
                in_channels : int, # : number of input channels
                out_channels : int, # : number of output channels (segmentation classes)
                *args, **kwargs
            ):
            # scale is useful for reducing the number of parameters in the network during debugging
            
            super(UNet_plusplus, self).__init__(*args, **kwargs)
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.net = net

            print(self.net)
        
            # TODO: print self.net to see the structure of the model
            # and modify the code below to augment the input and output channels
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
            return self.net
        
    return UNet_plusplus