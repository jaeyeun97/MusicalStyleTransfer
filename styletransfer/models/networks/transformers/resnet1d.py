import torch.nn as nn
from ..resnet import Resnet1d
from ..util import option_setter

options = { 
    'num_trans_layers': 9,
    'input_size': 1025,
}

class Resnet1dTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet1dTransformer, self).__init__()

        option_setter(self, options, kwargs)

        self.model = list()
        kwargs['nc'] = self.input_size
        for i in range(self.num_trans_layers):
            self.model += [Resnet1d(**kwargs)]

        self.model = nn.Sequential(*self.model)

    def forward(self, i):
        return self.model(i)
