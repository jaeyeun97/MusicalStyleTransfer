import torch
import torch.nn as nn
from ..util import option_setter
from ..reshape import Reshape


options = {
    'ngf': 4097,
    'conv_size': 5,
    'conv_pad': 4,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 2,
    'use_bias': False,
    'shrinking_filter': False,
    'transformer': None,
    'tensor_size': 1025,
    'mgf': 0.5  # out_channel / in_channel
}


class Conv1dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dEncoder, self).__init__()

        option_setter(self, options, kwargs)

        if self.shrinking_filter:
            self.conv_pad = 5 * (2 ** (self.n_downsample - 1))
            self.conv_size = self.conv_pad + 1


        mult = self.ngf
        self.model = [
            ('conv_init', nn.Conv1d(self.tensor_size, mult,
                                    kernel_size=self.conv_size,
                                    padding=self.conv_pad,
                                    dilation=2,
                                    bias=self.use_bias)),
            ('norm_init', self.norm_layer(mult)),
            ('tanh_init', nn.Tanh())
        ]

        
        # Downsample
        for i in range(self.n_downsample):
            next_mult = int((mult - 1) * self.mgf) + 1
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
            next_mult = int((mult - 1) // self.mgf) + 1
            self.model += [
                ('conv_up_%s' % i, nn.ConvTranspose1d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      padding=self.conv_pad,
                                                      dilation=2,
                                                      stride=2,
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('tanh_up_%s' % i, nn.Tanh())
            ]
            mult = next_mult

        self.model += [
            ('conv_final', nn.ConvTranspose1d(mult, self.tensor_size,
                                              kernel_size=self.conv_size,
                                              padding=self.conv_pad,
                                              dilation=2,
                                              bias=self.use_bias)),
            ('norm_final', self.norm_layer(self.tensor_size)),
            ('tanh_final', nn.Tanh())
        ]

        for name, module in self.model:
            self.add_module(name, module)

    def forward(self, input):
        for name, module in self.model:
            input = module(input)
        return input 
