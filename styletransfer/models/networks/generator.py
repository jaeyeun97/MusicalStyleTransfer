import torch.nn as nn
from .conv import ConvAutoencoder
from .resnet import Resnet
from .util import get_norm_layer, init_weights


def getGenerator(device, opt):
    args = dict(opt.__dict__)
    encoding_model = opt.autoencoder
    transformer_model = opt.transformer
    args['norm_layer'] = get_norm_layer(opt.norm_layer)

    if transformer_model == 'resnet':
        args['transformer'] = Resnet(**args)
    else:
        args['transformer'] = None

    if encoding_model == 'conv':
        netG = ConvAutoencoder(**args)
    else:
        raise NotImplementedError('Encoding Model not implemented') 

    init_weights(netG, opt.init_type, opt.init_gain)
    return netG.to(device)
