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
    'tensor_width': 1025,
    'tensor_height': 1025,
    'duration_ratio': 1,
    'input_nc': 1,
    'flatten': False
}

class ShallowClassifier(nn.Module):
    """Defines a CNN classifier without phase input"""

    def __init__(self, **kwargs):
        
        super(ShallowClassifier, self).__init__()

        option_setter(self, options, kwargs) 

        model = [
            nn.Conv2d(self.input_nc, self.ndf,
                      kernel_size=self.conv_size, 
                      padding=self.conv_pad,
                      stride=2,
                      bias=self.use_bias),  
            self.norm_layer(self.ndf),
            nn.ReLU()
        ]

        mult = self.ndf
        self.n_layers = 4
        for n in range(1, self.n_layers):
            next_mult = mult * 2
            model += [
                nn.Conv2d(mult, next_mult,
                          kernel_size=self.conv_size,
                          padding=self.conv_pad,
                          stride=2,
                          bias=self.use_bias), 
                self.norm_layer(self.ndf),
                nn.ReLU()
            ]
            mult = next_mult

        model += [
            nn.Conv2d(mult, 1, kernel_size=self.conv_size, padding=self.conv_pad),
        ]
 
        self.model = nn.ModuleList(model) # nn.Sequential(*model)

    def forward(self, input):
        """Standard forward."""
        if len(input.size()) < 4:
            input = input.unsqueeze(1)
        for model in self.model:
            input = model(input) 
        if self.flatten:
            input = input.view(input.size(0), -1)
            input = input.mean(dim=1)
        return input

    def to(self, device):
        self.device = device
        return super(ShallowClassifier, self).to(device)
