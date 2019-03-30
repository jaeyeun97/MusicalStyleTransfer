import torch
import numpy as np
import torch.nn as nn
from ..util import option_setter
from ...util.debug import Print

options = { 
    'ndf': 8,
    'conv_size': 5,
    'conv_pad': 4,
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
        
        self.n_layers = 5 # int(np.log2(self.tensor_size - 1))

        model = list()
 
        if self.shrinking_filter:
            self.conv_pad = 5 * (2 ** (self.n_layers - 1))
            self.conv_size = self.conv_pad + 1

        model += [
            nn.Conv2d(1, self.ndf,
                      kernel_size=self.conv_size,
                      padding=self.conv_pad,
                      dilation=2,
                      bias=self.use_bias),  
            nn.InstanceNorm2d(self.ndf, affine=True, track_running_stats=False),
            nn.ReLU(True)
        ]
        mult = self.ndf

        for n in range(self.n_layers):
            next_mult = mult * 2
            model += [
                nn.Conv2d(mult, next_mult,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          dilation=2,
                          stride=2,
                          bias=self.use_bias), 
                # nn.MaxPool2d(self.pool_size,
                #              padding=self.pool_pad,
                #              stride=self.pool_stride),
                nn.InstanceNorm2d(next_mult, affine=False, track_running_stats=False),
                nn.ReLU(True)
            ]
            mult = next_mult

        ts = ((self.tensor_size - 1) // (2 ** self.n_layers) + 1) ** 2
        model += [
            nn.Conv2d(mult, self.ndf,
                      kernel_size=self.conv_size,
                      padding=self.conv_pad,
                      dilation=2,
                      bias=self.use_bias), 
            Flatten(),
            nn.Linear(self.ndf * ts, self.ndf * ts * 2),
            nn.Sigmoid(),
            nn.Linear(self.ndf * ts * 2, self.ndf * ts),
            nn.Sigmoid(),
        ]
 
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        input = input.unsqueeze(1)
        input = self.model(input)
        return input


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view((1, -1)) 
