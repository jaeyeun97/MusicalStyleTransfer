from .classifiers import *
from .util import get_norm_layer, init_weights


def getDiscriminator(opt, device):
    norm_layer = get_norm_layer(opt.norm_layer)
    model = opt.discriminator

    if model == 'conv':
        netD = ConvClassifier(ndf=opt.ndf, n_layers=opt.disc_layers, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator not implmented')
    init_weights(netD, opt.init_type, opt.init_gain)
    return netD.to(device)
