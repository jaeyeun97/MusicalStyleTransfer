import functools
import torch.nn as nn

from .networks.classifiers import *
from .networks.util import get_norm_layer, init_weights, get_use_bias


def getDiscriminator(opt, device):    
    disc = Discriminator(opt).to(device)
    init_weights(disc, 'xavier', nn.init.calculate_gain('linear'))
    return disc

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        args = dict(opt.__dict__)
        model = opt.discriminator
        self.device = None

        if 'shallow' in model:
            pass
        elif '2d' in model:
            args['norm_layer'] = get_norm_layer(2, opt.norm_layer)
        elif '1d' in model:
            args['norm_layer'] = get_norm_layer(1, opt.norm_layer)
        else:
            raise NotImplementedError('Neither 1 or 2d')

        args['use_bias'] = get_use_bias(args)
 
        if model == 'conv1d':
            self.net = Conv1dClassifier(**args)
        elif model == 'conv2d':
            self.net = Conv2dClassifier(**args)
        elif model == 'shallow':
            self.net = ShallowClassifier(**args)
        else:
            raise NotImplementedError('Discriminator not implmented')
        
    def forward(self, i):
        return self.net(i)

    def to(self, device):
        self.device = device
        return super(Discriminator, self).to(device)
