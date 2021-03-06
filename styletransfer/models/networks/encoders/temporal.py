import torch.nn as nn
from ..wavenet import Conv
from ..util import option_setter

options = {
    'width': 128,
    'layers': 30,
    'stages': 10,
    'bottleneck_width': 16,
    'pool_length': 512,
}

class TemporalEncoder(nn.Module):
    def __init__(self, **kwargs): 
        super(TemporalEncoder, self).__init__()
        option_setter(self, options, kwargs) 
        self.device = None
 
        self.model = [
            ('conv_init', nn.Conv1d(1, self.width, kernel_size=3, padding=1))
        ]

        for i in range(self.layers):
            dilation = 2 ** (i % self.stages)
            self.model += [
                ('relu_first_%s' % i, nn.ReLU()),
                ('nc_dil_conv_%s' % i, nn.Conv1d(self.width, self.width,
                                                 kernel_size=3,
                                                 padding=dilation,
                                                 dilation=dilation)),
                ('relu_second_%i' % i, nn.ReLU()),
                ('1x1_conv_%s' % i, Conv(self.width, self.width)),
            ]
 
        self.model += [
            ('conv_final', Conv(self.width, self.bottleneck_width)),
            ('avgpool', nn.AvgPool1d(self.pool_length, stride=self.pool_length))
        ]

        for name, module in self.model:
            self.add_module(name, module)
            if 'nc_dil' in name:
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                # nn.init.kaiming_uniform_(module.weight)
            elif 'init' in name:
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                # nn.init.kaiming_uniform_(module.weight)



    def forward(self, input):
        for name, module in self.model:
            if 'relu_first' in name:
                skip_input = input  
            input = module(input)
            if '1x1_conv' in name:
                input = skip_input + input
        return input

    def to(self, device):
        self.device = device
        return super(TemporalEncoder, self).to(device)
