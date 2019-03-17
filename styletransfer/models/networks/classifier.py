import torch.nn as nn
import functools


class FrequencyClassifier(nn.Module):
    """Defines a CNN timbre classifier"""

    def __init__(self, ndf=16, n_layers=4, norm_layer=nn.BatchNorm2d):
        """
        Parameters:
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(ConvClassifier, self).__init__()
        if type(norm_layer) == functools.partial: 
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 3
        padw = 1
        sequence = [nn.Conv2d(2, ndf, kernel_size=kw, stride=1, padding=padw),
                    nn.Conv2d(ndf, ndf, kernel_size=kw, stride=1, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(nf_mult * 2, 16)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, padding=padw, bias=use_bias),
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=(1, kw), padding=(0, padw), stride=(1,2), bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
 
        # output size = (*, 513, 65)
        nf_mult_prev = nf_mult
        nf_mult = min(nf_mult * 2, 16)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, padding=1, bias=use_bias),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]  
        sequence += [
            nn.Conv2d(ndf * nf_mult, 2, kernel_size=kw, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True) 
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)



class PixelClassifier(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelClassifier, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(2, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
