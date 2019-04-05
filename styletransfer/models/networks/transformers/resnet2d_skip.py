import torch.nn as nn
from ..resnet import Resnet2d
from ..util import option_setter

options = { 
    'num_trans_layers': 9,
    'channel_size': 32,
}

class Resnet2dTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet2dTransformer, self).__init__()

        option_setter(self, options, kwargs)

        self.model = list()
        kwargs['nc'] = self.channel_size
        for i in range(self.num_trans_layers):
            self.model += [('resnet_%s' % i, Resnet2d(**kwargs))]

        for name, module in self.model:
            self.add_modules(name, module)

    def forward(self, i):
        output = None
        for name, module in self.model:
            i = module(i)
            if output is None:
                output = i
            else:
                output += i 
        return output
