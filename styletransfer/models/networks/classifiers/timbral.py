import torch
import numpy as np
import torch.nn as nn
from ..util import option_setter
from ...util.debug import Print

options = { 
    'ndf': 8,
    'conv_size': 5,
    'conv_pad': 4,
    'pool_size': (1, 2),
    'pool_pad': (0, 1),
    'pool_stride': (1, 2),
    'norm_layer': nn.BatchNorm2d,
    'use_bias': False,
    'shrinking_filter': False,
    'tensor_size': 1025,
    'duration_ratio': 1
}

class TimbralClassifier(nn.Module):
    """Defines a CNN classifier without phase input"""

    def __init__(self, **kwargs):
        
        super(TimbralClassifier, self).__init__()

        option_setter(self, options, kwargs) 

        print(self.duration_ratio) 
        self.tensor_size = (self.tensor_size - 1) // self.duration_ratio + 1
        self.n_layers = int(np.log2(self.tensor_size - 1))
        # self.conv_pad = self.conv_pad - (kernel_size - 1) // 2

        model = [
            nn.Conv2d(1, self.ndf,
                      kernel_size=3, # self.conv_size,
                      padding=1,
                      bias=self.use_bias),  
            nn.ReLU()
        ]
        mult = self.ndf

        first = int(np.log2(self.conv_size - 1))
        for n in range(first, self.n_layers):
            next_mult = mult * 2
            model += [
                nn.Conv2d(mult, next_mult,
                          kernel_size=3, # self.conv_size,
                          padding=1, # self.conv_pad,
                          bias=self.use_bias), 
                nn.AvgPool2d(self.pool_size,
                             padding=self.pool_pad,
                             stride=self.pool_stride),
                nn.ReLU()
            ]
            mult = next_mult

        # ts = ((self.tensor_size - 1) // (2 ** self.n_layers) + 1) ** 2
        model += [
            nn.Conv2d(mult, 1, kernel_size=self.conv_size, padding=(1, 0)),
            # nn.LeakyReLU(0.2, True),
            # Flatten(),
            # nn.Linear(self.tensor_size, 2),
            # nn.Tanh(),
            # nn.Linear(2, 2),
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
        return input.view(1, -1)