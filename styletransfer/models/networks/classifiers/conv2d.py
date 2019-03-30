import torch
import numpy as np
import torch.nn as nn
from ..util import option_setter
from ...util.debug import Print

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

class Conv2dClassifier(nn.Module):
    """Defines a CNN classifier without phase input"""

    def __init__(self, **kwargs):
        
        super(Conv2dClassifier, self).__init__()

        option_setter(self, options, kwargs) 
        
        self.n_layers = int(np.log2(self.tensor_size - 1))

        model = list()
 
        if self.shrinking_filter:
            self.conv_pad = 5 * (2 ** (self.n_layers - 1))
            self.conv_size = self.conv_pad + 1

        mult = 1 
        for n in range(1, self.n_layers):
            next_mult = mult * 2
            model += [
                Print('disc %s' % n),
                nn.Conv2d(mult, next_mult,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          dilation=2,
                          bias=self.use_bias), 
                nn.MaxPool2d(self.pool_size,
                             padding=self.pool_pad,
                             stride=self.pool_stride),
                nn.InstanceNorm2d(next_mult, affine=False, track_running_stats=False),
                nn.Tanh()
            ]
            mult = next_mult

        # should be 3 here
        model += [
            nn.Conv2d(mult, 1, kernel_size=3, bias=self.use_bias),
            nn.InstanceNorm2d(next_mult, affine=False, track_running_stats=False),
            nn.Tanh()
        ]
        # now self.tensor_sizex1, prediction per frequency
 
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        input = input.view(1, 1, self.tensor_size, self.tensor_size)
        input = self.model(input)
        return input
