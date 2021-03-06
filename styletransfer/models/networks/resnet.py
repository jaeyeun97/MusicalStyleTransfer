import torch
import torch.nn as nn
from .util import option_setter

options = {
    'conv_size': 5,
    'conv_pad': 2,
    'nc': 1025,
    'use_bias': False,
    'norm_layer': nn.BatchNorm1d,
    'use_dropout': False
}

class Resnet2d(nn.Module):
    """Define a Resnet block"""

    def __init__(self, **kwargs):
        super(Resnet2d, self).__init__()
        option_setter(self, options, kwargs)

        conv_block = [
                nn.Conv2d(self.nc, self.nc,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          bias=self.use_bias),
                self.norm_layer(self.nc),
                nn.ReLU(True)
            ]

        if self.use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [
                nn.Conv2d(self.nc, self.nc,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          bias=self.use_bias),
                self.norm_layer(self.nc)
            ]

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        x = x + self.model(x)
        return x


class Resnet1d(nn.Module):
    """Define a Resnet block"""

    def __init__(self, **kwargs): 
        super(Resnet1d, self).__init__()
        option_setter(self, options, kwargs)

        conv_block = [
                nn.Conv1d(self.nc, self.nc,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          bias=self.use_bias),
                nn.ReLU(True)
            ]

        if self.use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [
                nn.Conv1d(self.nc, self.nc,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          bias=self.use_bias),
                # self.norm_layer(self.nc)
            ]

        self.model = nn.Sequential(*conv_block)

    def forward(self, x):
        x = x + self.model(x)
        return x
