import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
            self,
            in_channels : int, 
            out_channels : int, 
            mid_channels : Optional[int] = None # number of channels in the middle layer
        ):
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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
            self,
            in_channels : int, # number of input channels
            out_channels : int, # number of output channels
            bilinear : Optional[bool] = True # whether to use bilinear interpolation or transposed convolutions
        ):
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


class OutConv(nn.Module):
    def __init__(self, in_channels : int, out_channels : int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def inherit_unet_class_from_parent(parent_class):
    # create class for U-Net. This specifies the model from scratch without
    # any pretrained weights, this is old code as I now purely use pretrained.
    # The parent_class argument specifies which class to inherit from.
    # inheret from BphPSEG for semantic segmentation models and BphPQUANT for
    # quantification models of the Bacterial Phytochrome.
    
    class UNet(parent_class):
        def __init__(
                self, 
                in_channels : int, # number of input channels
                out_channels :int, # number of output channels
                bilinear : Optional[bool] = False, # whether to use bilinear interpolation or transposed convolutions
                scale : Optional[int] = 1, # scale downsizes the number of channels in the network by a factor of 'scale'
                *args, **kwargs
            ):
            # scale is useful for reducing the number of parameters in the network during debugging
            
            super(UNet, self).__init__(*args, **kwargs)
            
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
        
    return UNet


def inherit_unet_pretrained_class_from_parent(parent_class):
    # create class for U-Net.
    # the parent_class argument specifies which class to inherit from.
    # inheret from BphPSEG for semantic segmentation models and BphPQUANT for
    # quantification models of the Bacterial Phytochrome.
    
    class UNet(parent_class):
        def __init__(
                self, net, # : nn.Module : pretrained segformer model
                in_channels : int, # : number of input channels
                out_channels : int, # : number of output channels (segmentation classes)
                *args, **kwargs
            ):
            # scale is useful for reducing the number of parameters in the network during debugging
            
            super(UNet, self).__init__(*args, **kwargs)
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.net = net
            '''
            @inproceedings{ronneberger2015u,
            title={U-net: Convolutional networks for biomedical image segmentation},
            author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
            booktitle={Medical image computing and computer-assisted intervention--MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18},
            pages={234--241},
            year={2015},
            organization={Springer}
            }
            
            @inproceedings{zhou2018unet++,
            title={Unet++: A nested u-net architecture for medical image segmentation},
            author={Zhou, Zongwei and Rahman Siddiquee, Md Mahfuzur and Tajbakhsh, Nima and Liang, Jianming},
            booktitle={Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 20, 2018, Proceedings 4},
            pages={3--11},
            year={2018},
            organization={Springer}
            }
            
            @article{zhou2019unet++,
            title={Unet++: Redesigning skip connections to exploit multiscale features in image segmentation},
            author={Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
            journal={IEEE transactions on medical imaging},
            volume={39},
            number={6},
            pages={1856--1867},
            year={2019},
            publisher={IEEE}
            }'''

        
        def forward(self, x):
            return self.net.forward(x)
        
    return UNet