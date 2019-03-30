import torch
import numpy as np
import torch.nn as nn
from ..util import option_setter
from ...util.debug import Print

options = { 
    'ndf': 8, 
    'conv_size': 5,
    'conv_pad': 4, 
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

        if self.shrinking_filter:
            self.conv_pad = (2 ** (self.n_layers - 1))
            self.conv_size = self.conv_pad + 1

        mult = (self.tensor_size - 1) * self.ndf + 1
        self.model = [ 
                nn.Conv1d(self.tensor_size, mult,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          dilation=2,
                          bias=self.use_bias),  
                self.norm_layer(mult),
                nn.ReLU(True)
            ]
 
        first = int(np.log2(self.conv_size - 1))
        for n in range(first, self.n_layers):
            next_mult = (mult - 1) // 2 + 1 if n % 2 == 0 else next_mult
            self.model += [
                nn.Conv1d(mult, next_mult,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          dilation=2,
                          stride=2,
                          bias=self.use_bias),  
                self.norm_layer(mult),
                nn.ReLU(True)
            ]
            mult = next_mult

        self.model += [
            nn.Conv1d(mult, mult, kernel_size=self.conv_size, bias=self.use_bias),
            self.norm_layer(mult),
            nn.ReLU(True),
            # test layer
            Flatten(),
            nn.Linear(mult, mult * 2),
            nn.Sigmoid(),
            nn.Linear(mult * 2, mult),
            nn.Sigmoid()
        ]
 
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """Standard forward."""
        input = self.model(input)
        return input

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view((1, -1)) 
