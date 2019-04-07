import torch
import numpy as np
import torch.nn as nn
from ..util import option_setter
from ...util.debug import Print

options = { 
    'ndf': 8,
    'conv_size': 5,
    'conv_pad': 4,
    'pool_size': (1, 3),
    'pool_pad': (0, 1),
    'pool_stride': (1, 2),
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
        # self.conv_pad = self.conv_pad - (kernel_size - 1) // 2

        model = [
            nn.Conv2d(1, self.ndf,
                      kernel_size=self.conv_size,
                      padding=self.conv_pad,
                      dilation=2,
                      bias=self.use_bias),  
            self.norm_layer(self.ndf),
            nn.LeakyReLU(0.2, True)
        ]
        mult = self.ndf

        first = int(np.log2(self.conv_size - 1))
        for n in range(first, self.n_layers):
            next_mult = mult * 2
            model += [
                nn.Conv2d(mult, next_mult,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          dilation=2,
                          bias=self.use_bias), 
                nn.AvgPool2d(self.pool_size,
                             padding=self.pool_pad,
                             stride=self.pool_stride),
                self.norm_layer(next_mult),
                nn.LeakyReLU(0.2, True)
            ]
            mult = next_mult

        # ts = ((self.tensor_size - 1) // (2 ** self.n_layers) + 1) ** 2
        model += [
            nn.Conv2d(mult, 1,
                      kernel_size=self.conv_size,
                      padding=(self.conv_pad, 0),
                      dilation=(2, 1),
                      bias=self.use_bias), 
            # self.norm_layer(mult),
            # nn.LeakyReLU(0.2, True),
            # Flatten(),
            # nn.Linear(self.tensor_size, 1),
            # nn.Sigmoid(),
        ]
 
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        input = input.unsqueeze(1)
        input = self.model(input)
        return input

