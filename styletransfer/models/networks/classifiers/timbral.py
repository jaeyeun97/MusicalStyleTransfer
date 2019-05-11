import torch
import numpy as np
import torch.nn as nn
from ..util import option_setter
from ...util.debug import Print

options = { 
    'ndf': 4,
    'conv_size': 5,
    'conv_pad': 4,
    'norm_layer': nn.BatchNorm2d,
    'use_bias': False,
    'shrinking_filter': False,
    'tensor_height': 1025,
    'tensor_width': 1025,
    'duration_ratio': 1,
    'input_nc': 1,
    'flatten': False
}

class TimbralClassifier(nn.Module):
    """Defines a CNN classifier without phase input"""

    def __init__(self, **kwargs):
        
        super(TimbralClassifier, self).__init__()

        option_setter(self, options, kwargs) 

        mult = self.ndf 
        model = [
            nn.Conv2d(self.input_nc, self.ndf,
                      kernel_size=self.conv_size, 
                      padding=(self.conv_pad, 0),
                      bias=self.use_bias),  
            # self.norm_layer(self.ndf),
            nn.ReLU()
        ]
        
        self.n_layers = int(np.log2(self.tensor_height - 1))
        first = int(np.log2(self.conv_size - 1))
        for n in range(first, self.n_layers):
            next_mult = mult * 2 
            model += [
                nn.Conv2d(mult, next_mult,
                          kernel_size=self.conv_size,
                          padding=(self.conv_pad, 0),
                          stride=(2, 1),
                          bias=self.use_bias), 
                nn.ReLU()
            ]
            mult = next_mult

        for module in model:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))

        last_conv = nn.Conv2d(mult, 1, kernel_size=3)
        nn.init.xavier_uniform_(last_conv.weight, gain=nn.init.calculate_gain('linear'))
        model.append(last_conv)
        self.model = nn.ModuleList(model)

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
