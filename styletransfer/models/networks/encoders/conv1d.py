import torch
import torch.nn as nn
from ..util import option_setter
from ..reshape import Reshape


options = {
    'conv_size': 3,
    'conv_pad': 1,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 2,
    'use_bias': False,
    'shrinking_filter': False,
    'transformer': None,
    'tensor_size': 1025,
    'k': 2,  # out_channel / in_channel
}


class Conv1dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dEncoder, self).__init__()

        option_setter(self, options, kwargs)

        if self.shrinking_filter:
            self.conv_pad = 5 * (2 ** (self.n_downsample - 1))
            self.conv_size = self.conv_pad + 1

        self.models = list()

        # Downsample
        mult = self.tensor_size

        for i in range(self.n_downsample):
            next_mult = mult * self.k
            self.model += [
                ('conv_down_%s' % i, nn.Conv1d(mult, next_mult,
                                               kernel_size=self.conv_size,
                                               padding=self.conv_pad,
                                               dilation=2,
                                               stride=2,
                                               bias=self.use_bias)),
                ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('tanh_down_%s' % i, nn.Tanh())
            ]
            if self.shrinking_filter:
                self.conv_size = self.conv_pad + 1
                self.conv_pad = self.conv_pad // 2
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            self.model.append(('transformer', self.transformer))

        # Upsample
        for i in range(self.n_downsample):
            if self.shrinking_filter:
                self.conv_pad = 2 ** i
                self.conv_size = 2 * self.conv_pad + 1
            next_mult = int(mult * self.k)
            self.model += [
                ('conv_up_%s' % i, nn.ConvTranspose1d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      padding=self.conv_pad,
                                                      dilation=2,
                                                      stride=2,
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('Tanh_up_%s' % i, nn.Tanh())
            ]
            mult = next_mult

        self.model += [
            ('conv_final', nn.Conv2d(mult, self.tensor_size, kernel_size=self.conv_size, padding=self.conv_pad)),
            ('tanh', nn.Tanh())
        ]

        for name, module in self.model:
            self.add_module(name, module)

    def forward(self, input):
        phase = input[:, 1, :, :]
        input = input[:, 0, :, :]
        for name, module in self.model:
            if 'transformer' in name:
                input = module(torch.stack((input, phase), 1))
                phase = input[:, 1, :, :]
                input = input[:, 0, :, :]
            else:
                input = module(input)
        return torch.stack((input, phase), 1)
