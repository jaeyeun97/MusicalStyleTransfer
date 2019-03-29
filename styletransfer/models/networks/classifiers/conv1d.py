import torch
import numpy as np
import torch.nn as nn
from ..util import option_setter

options = { 
    'conv_size': 3,
    'conv_pad': 2,
    'pool_size': 3,
    'pool_pad': 1,
    'pool_stride': 2,
    'norm_layer': nn.BatchNorm2d,
    'use_bias': False,
    'shrinking_filter': False,
    'tensor_size': 1025
}

class Conv1dClassifier(nn.Module):
    """Defines a CNN classifier without phase input"""

    def __init__(self, **kwargs):
        
        super(Conv1dClassifier, self).__init__()

        option_setter(self, options, kwargs) 
        
        self.n_layers = int(np.log2(self.tensor_size - 1))

        self.model = list()
 
        if self.shrinking_filter:
            self.conv_pad = 5 * (2 ** (self.n_layers - 1))
            self.conv_size = self.conv_pad + 1

        mult = self.tensor_size
        for n in range(1, self.n_layers):
            self.model += [
                nn.Conv1d(mult, mult,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          dilation=2,
                          bias=self.use_bias), 
                nn.MaxPool1d(mult, mult,
                             kernel_size=self.pool_size,
                             padding=self.pool_pad,
                             stride=self.pool_stride,
                             bias=self.use_bias),
                self.norm_layer(mult),
                nn.Tanh(True)
            ]
        # should be 3 here
        self.model += [
            nn.Conv1d(mult, mult, kernel_size=3, bias=self.use_bias),
            self.norm_layer(mult),
            nn.Tanh(True),
            Flatten(-1)
        ]
        # now self.tensor_sizex1, prediction per frequency
 
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """Standard forward."""
        phase = input[:, 1, :, :]
        input = input[:, 0, :, :]
        return torch.stack((self.model(input), phase), 1)


class Flatten(nn.Module):
    def __init__(self, size):
        super(Flatten, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(size) 
