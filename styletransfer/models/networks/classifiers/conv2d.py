import torch.nn as nn
import functools


class Conv2dClassifier(nn.Module):
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
                    nn.ReLU(True)]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(nf_mult * 2, 16)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, padding=padw, bias=use_bias),
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, padding=padw, stride=2, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.ReLU(True)
            ]
 
        # output size = (*, 513, 65)
        nf_mult_prev = nf_mult
        nf_mult = min(nf_mult * 2, 16)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, padding=1, bias=use_bias),
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.ReLU(True)
        ]  
        sequence += [
            nn.Conv2d(ndf * nf_mult, 2, kernel_size=kw, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.ReLU(True) 
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

