"""
This file contains the configuration for the DcGAN model.
"""
import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class Config:
    """
    Configuration for the DcGAN model using dataclass.
    """
    nc: int = 3  # Number of channels in the training images. For color images this is 3
    nz: int = 100  # Size of z latent vector (i.e. size of generator input)
    ngf: int = 64  # Size of feature maps in generator
    ndf: int = 64  # Size of feature maps in discriminator

def weights_init(m):
    """
    Custom weights initialization called on ``netG`` and ``netD``.
    """
    classname = m.__class__.__name__
    if 'Conv' in classname:  # 檢查類名是否包含 "Conv"
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:  # 檢查類名是否包含 "BatchNorm"
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
