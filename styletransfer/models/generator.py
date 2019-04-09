import functools
import torch.nn as nn
from .networks.encoders import Conv1dEncoder, Conv2dEncoder, CRNNEncoder, Res1dEncoder
from .networks.transformers import * 
from .networks.util import get_norm_layer, init_weights, get_use_bias


def getGenerator(device, opt):
    generator = Generator(opt).to(device)
    # init_weights(generator, 'xavier', nn.init.calculate_gain('leaky_relu', 0.2))
    return generator


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        args = dict(opt.__dict__)
        encoding_model = opt.encoder
        transformer_model = opt.transformer
        self.device = None

        if 'none' in transformer_model:
            pass
        elif '2d' in transformer_model:
            args['norm_layer'] = get_norm_layer(2, opt.norm_layer)
        elif '1d' in transformer_model or 'lstm' in transformer_model:
            args['norm_layer'] = get_norm_layer(1, opt.norm_layer)
        else:
            raise NotImplementedError('Neither 1 or 2d')

        args['use_bias'] = get_use_bias(args)
 
        if transformer_model == 'resnet1d':
            args['transformer'] = Resnet1dTransformer
        elif transformer_model == 'resnet2d':
            args['transformer'] = Resnet2dTransformer 
        elif transformer_model == 'lstm':
            args['transformer'] = LSTMTransformer 
        else:
            args['transformer'] = None

        if 'res' in encoding_model:
            pass
        elif '2d' in encoding_model:
            args['norm_layer'] = get_norm_layer(2, opt.norm_layer)
        elif '1d' in encoding_model or 'crnn' in encoding_model:
            args['norm_layer'] = get_norm_layer(1, opt.norm_layer)
        else:
            raise NotImplementedError('Neither 1 or 2d')

        args['use_bias'] = get_use_bias(args)

        if encoding_model == 'conv2d':
            self.net = Conv2dEncoder(**args)
        elif encoding_model == 'conv1d':
            self.net = Conv1dEncoder(**args)
        elif encoding_model == 'crnn':
            self.net = CRNNEncoder(**args)
        elif encoding_model == 'res1d':
            self.net = Res1dEncoder(**args)
        else:
            raise NotImplementedError('Encoding Model not implemented') 

    def forward(self, i):
        return self.net(i)

    def to(self, device):
        self.device = device
        return super(Generator, self).to(device)
