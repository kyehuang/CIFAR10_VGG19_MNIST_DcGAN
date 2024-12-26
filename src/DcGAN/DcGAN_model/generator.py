"""
DcGAN Generator model
"""
import torch.nn as nn
from src.DcGAN.DcGAN_model.config import Config

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        print(Config.nz, Config.ngf, Config.nc)
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( Config.nz, Config.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(Config.ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(Config.ngf * 8, Config.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( Config.ngf * 4, Config.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( Config.ngf * 2, Config.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(Config.ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( Config.ngf, Config.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
