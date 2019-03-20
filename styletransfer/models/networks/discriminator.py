import torch.nn as nn

from .classifiers import *
from .util import get_norm_layer, init_weights


def getDiscriminator(opt, device):    
    return Discriminator(opt).to(device)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        norm_layer = get_norm_layer(opt.norm_layer)
        model = opt.discriminator
        self.device = None

        if model == 'conv':
            self.net = ConvClassifier(ndf=opt.ndf, n_layers=opt.disc_layers, norm_layer=norm_layer)
        else:
            raise NotImplementedError('Discriminator not implmented')
        init_weights(self.net, opt.init_type, opt.init_gain)
        
    def forward(self, i):
        return self.net(i)

    def to(self, device):
        self.device = device
        return super(Discriminator, self).to(device)
