import functools
import torch.nn as nn
from .networks.encoders import Conv1dEncoder, Conv2dEncoder, CRNNEncoder
from .networks.util import get_norm_layer, init_weights


def getGenerator(device, opt):
    return Generator(opt).to(device)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        args = dict(opt.__dict__)
        encoding_model = opt.encoder
        transformer_model = opt.transformer
        args['norm_layer'] = get_norm_layer(opt.norm_layer)
        self.device = None

        if type(args['norm_layer']) == functools.partial:
            args['use_bias'] = args['norm_layer'].func != nn.BatchNorm1d
        else:
            args['use_bias'] = args['norm_layer'] != nn.BatchNorm1d

        if transformer_model == 'resnet':
            args['transformer'] = Resnet
        else:
        args['transformer'] = None

        if encoding_model == 'conv2d':
            self.net = Conv2dEncoder(**args)
        elif encoding_model == 'conv1d':
            self.net = Conv1dEncoder(**args)
        elif encoding_model == 'crnn':
            self.net = CRNNEncoder(**args)
        else:
            raise NotImplementedError('Encoding Model not implemented') 
        init_weights(self.net, opt.init_type, opt.init_gain)

    def forward(self, i):
        return self.net(i)

    def to(self, device):
        self.device = device
        return super(Generator, self).to(device)
