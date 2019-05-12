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


class GatedResnet2d(nn.Module):
    """Define a Resnet block"""

    def __init__(self, **kwargs):
        super(GatedResnet2d, self).__init__()
        option_setter(self, options, kwargs)

        self.in_conv = nn.Conv2d(self.nc, self.nc * 2,
                                 kernel_size=self.conv_size,
                                 padding=self.conv_pad,
                                 bias=self.use_bias)
        nn.init._xavier_uniform(self.in_conv.weight, gain=nn.init.calculate_gain('tanh'))

        self.out_conv = nn.Conv2d(self.nc, self.nc,
                                  kernel_size=1,
                                  bias=self.use_bias)
        nn.init._xavier_uniform(self.out_conv.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, input):
        x = self.in_conv(input)
        x = torch.tanh(x[:, :self.nc, :, :]) * torch.sigmoid(x[:, self.nc:, :, :])
        return input + x


class GatedResnet1d(nn.Module):
    """Define a Resnet block"""

    def __init__(self, **kwargs):
        super(GatedResnet1d, self).__init__()
        option_setter(self, options, kwargs)
        self.in_conv = nn.Conv1d(self.nc, self.nc * 2,
                                 kernel_size=self.conv_size,
                                 padding=self.conv_pad,
                                 bias=self.use_bias)
        nn.init._xavier_uniform(self.in_conv.weight, gain=nn.init.calculate_gain('tanh'))

        self.out_conv = nn.Conv1d(self.nc, self.nc,
                                  kernel_size=1,
                                  bias=self.use_bias)
        nn.init._xavier_uniform(self.out_conv.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, input):
        x = self.in_conv(input)
        x = torch.tanh(x[:, :self.nc, :]) * torch.sigmoid(x[:, self.nc:, :])
        return input + x
