import torch
import torch.nn as nn
from ..util import option_setter
from ..reshape import Reshape


options = {
    'ngf': 8,
    'conv_size': 3,
    'conv_pad': 1,
    'pool_size': 3,
    'pool_pad': 1,
    'pool_stride': 2,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 3,
    'use_bias': False,
    'transformer': None,
    'tensor_size': 1025,
    'mgf': 0.5  # out_channel / in_channel
}


class Conv1dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dEncoder, self).__init__()

        option_setter(self, options, kwargs)
 
        self.indices = list()

        mult = self.ngf
        self.model = [
            ('conv_init', nn.Conv1d(self.tensor_size, mult,
                                    kernel_size=self.conv_size + 2,
                                    padding=self.conv_pad + 1,
                                    bias=self.use_bias)),
            # ('norm_init', self.norm_layer(mult)),
            ('relu_init', nn.ReLU(True))
        ]
 
        # Downsample
        for i in range(self.n_downsample):
            next_mult = int(mult * self.mgf) 
            self.model += [
                ('conv_down_%s' % i, nn.Conv1d(mult, next_mult,
                                               kernel_size=self.conv_size,
                                               padding=self.conv_pad,
                                               bias=self.use_bias)), 
                # ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('relu_down_%s' % i, nn.ReLU(True)),
            ]
            if self.pool_stride != 1:
                self.model.append(('pool_down_%s' % i, nn.MaxPool1d(self.pool_size,
                                                                    stride=self.pool_stride,
                                                                    padding=self.pool_pad,
                                                                    return_indices=True)))
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            kwargs['input_size'] = mult 
            self.model.append(('trans', self.transformer(**kwargs)))

        # Upsample
        for i in range(self.n_downsample):
            next_mult = int(mult // self.mgf)
            if self.pool_stride != 1:
                self.model.append(('unpool_up_%s' % i, nn.MaxUnpool1d(self.pool_size,
                                                                      stride=self.pool_stride,
                                                                      padding=self.pool_pad)))
            self.model += [
                ('conv_up_%s' % i, nn.ConvTranspose1d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      padding=self.conv_pad,
                                                      bias=self.use_bias)),
                # ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('relu_up_%s' % i, nn.ReLU(True))
            ]
            mult = next_mult

        self.model += [
            ('conv_final', nn.Conv1d(mult, self.tensor_height,
                                     kernel_size=self.conv_size + 2,
                                     padding=self.conv_pad + 1,
                                     bias=self.use_bias)),
            # ('tanh_final', nn.Tanh())
        ]

        for name, module in self.model:
            if 'conv_final' in name: 
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('linear'))
            elif 'conv' in name:
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
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
