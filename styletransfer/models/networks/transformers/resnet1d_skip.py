import torch.nn as nn
from ..resnet import Resnet1d
from ..util import option_setter

options = { 
    'num_trans_layers': 9,
    'input_size': 1025,
}

class Resnet1dSkipTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet1dTransformer, self).__init__()

        option_setter(self, options, kwargs)

        self.model = list()
        kwargs['nc'] = self.input_size
        for i in range(self.num_trans_layers):
            self.model += [('resnet_%s' % i, Resnet1d(**kwargs))]

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
