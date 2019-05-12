import torch.nn as nn
from ..gated_resnet import GatedResnet2d
from ..util import option_setter

options = { 
    'num_trans_layers': 9,
    'channel_size': 32,
}

class GatedResnet2dTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(GatedResnet2dTransformer, self).__init__()

        option_setter(self, options, kwargs)

        self.model = list()
        kwargs['nc'] = self.channel_size
        for i in range(self.num_trans_layers):
            self.model += [GatedResnet2d(**kwargs)]

        self.model = nn.Sequential(*self.model)

    def forward(self, i):
        return self.model(i)
