import torch.nn as nn
from .conv import ConvAutoencoder
from .resnet import Resnet
from .util import get_norm_layer, init_weights


def getGenerator(device, opt):
    return Generator(opt).to(device)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        args = dict(opt.__dict__)
        encoding_model = opt.autoencoder
        transformer_model = opt.transformer
        args['norm_layer'] = get_norm_layer(opt.norm_layer)
        self.device = None

        if transformer_model == 'resnet':
            args['transformer'] = Resnet(**args)
        else:
            args['transformer'] = None

        if encoding_model == 'conv':
            self.net = ConvAutoencoder(**args)
        else:
            raise NotImplementedError('Encoding Model not implemented') 
        init_weights(self.net, opt.init_type, opt.init_gain)

    def forward(self, i):
        return self.net(i)

    def to(self, device):
        self.device = device
        return super(Generator, self).to(device)
