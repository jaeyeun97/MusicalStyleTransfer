import torch.nn as nn

from .classifiers import *
from .util import get_norm_layer, init_weights


def getDiscriminator(opt, device):    
    return Discriminator(opt).to(device)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        args = dict(opt.__dict__)
        args['norm_layer'] = get_norm_layer(opt.norm_layer)

        model = opt.discriminator
        self.device = None

        if type(args['norm_layer']) == functools.partial:
            args['use_bias'] = args['norm_layer'].func != nn.BatchNorm2d
        else:
            args['use_bias'] = args['norm_layer'] != nn.BatchNorm2d


        if model == 'conv1d':
            self.net = Conv1dClassifier(**args)
        elif model == 'conv2d':
            self.net = Conv2dClassifier(**args)
        else:
            raise NotImplementedError('Discriminator not implmented')
        init_weights(self.net, opt.init_type, opt.init_gain)
        
    def forward(self, i):
        return self.net(i)

    def to(self, device):
        self.device = device
        return super(Discriminator, self).to(device)
