import torch.nn as nn
from .util import option_setter
from .reshape import Reshape


options = {
    'conv_size': 3,
    'conv_pad': 1,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 4,
    'use_bias': False,
    'shrinking_filter': False,
    'reshaped': False,
    'transformer': None,
    'nfft': 1024
}


class ReshapedConvAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(ReshapedConvAutoencoder, self).__init__()

        option_setter(self, options, kwargs)

        if self.shrinking_filter:
            self.conv_pad = 2 ** (self.n_downsample - 1)
            self.conv_size = 2 * self.conv_pad + 1

        self.indices = list()

        # Downsample
        mult = self.nfft
        self.model = [('reshape_init', Reshape((0, 2, 1, 3)))]

        for i in range(self.n_downsample):
            next_mult = mult * 2
            self.model += [
                ('conv_down_%s' % i, nn.Conv2d(mult, next_mult,
                                               kernel_size=(2, self.conv_size),
                                               padding=(0, self.conv_pad),
                                               dilation=(1, 2),
                                               stride=(1, 2),
                                               bias=self.use_bias)),
                ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('relu_down_%s' % i, nn.ReLU(True))
            ]
            if self.shrinking_filter:
                self.conv_size = self.conv_pad + 1
                self.conv_pad = self.conv_pad // 2
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            self.model.append(self.transformer)

        # Upsample
        for i in range(self.n_downsample):
            if self.shrinking_filter:
                self.conv_pad = 2 ** i
                self.conv_size = 2 * self.conv_pad + 1
            next_mult = mult // 2
            self.model += [
                ('conv_up_%s' % i, nn.ConvTranspose2d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      padding=self.conv_pad,
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('relu_up_%s' % i, nn.ReLU(True))
            ]
            mult = next_mult

        self.model += [
            ('conv_final', nn.Conv2d(mult, 2, kernel_size=self.conv_size, padding=self.conv_pad)),
            ('tanh', nn.Tanh())
        ]

        for name, module in self.model:
            self.add_module(name, module)

    def forward(self, input):
        for name, module in self.model:
            if 'pool_down' in name:
                input, indices = module(input)
                self.indices.append(indices)
            elif 'unpool_up' in name:
                indices = self.indices.pop()
                input = module(input, indices)
            else:
                input = module(input)
        return input
