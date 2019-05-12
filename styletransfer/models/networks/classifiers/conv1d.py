import torch
import numpy as np
import torch.nn as nn
from ..util import option_setter
from ...util.debug import Print

options = { 
    'n_layers': 3,
    'ndf': 2048,
    'conv_size': 5,
    'conv_pad': 4, 
    'pool_size': 3,
    'pool_pad': 1,
    'pool_stride': 2,
    'norm_layer': nn.BatchNorm2d,
    'use_bias': False,
    'tensor_height': 1025
}

class Conv1dClassifier(nn.Module):
    """Defines a CNN classifier without phase input"""

    def __init__(self, **kwargs):
        
        super(Conv1dClassifier, self).__init__()

        option_setter(self, options, kwargs) 
        
        self.model = list() 

        mult = self.tensor_height
        for n in range(self.n_layers):
            next_mult = min(self.ndf, mult * 2)
            self.model += [
                nn.Conv1d(mult, next_mult,
                          kernel_size=self.conv_size,
                          bias=self.use_bias),  
                nn.ReLU()
            ]
            mult = next_mult

        for module in self.model:
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                
        last_conv = nn.Conv1d(mult, 1, 1, bias=self.use_bias)
        nn.init.xavier_uniform_(last_conv.weight, gain=nn.init.calculate_gain('linear'))
        self.model.append(last_conv)  
 
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """Standard forward."""
        input = self.model(input)
        input = input.mean(dim=2).squeeze(1)
        return torch.sigmoid(input)
