import torch
import torch.nn as nn
from ..util import option_setter
from ..reshape import Reshape


options = {
    'ngf': 8,
    'conv_size': 5,
    'conv_pad': 4,
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

        mult = (self.tensor_size - 1) * self.ngf + 1
        self.model = [
            ('conv_init', nn.Conv1d(self.tensor_size, mult,
                                    kernel_size=self.conv_size,
                                    padding=self.conv_pad,
                                    dilation=2,
                                    bias=self.use_bias)),
            ('norm_init', self.norm_layer(mult)),
            ('relu_init', nn.Tanh())
        ]
 
        # Downsample
        for i in range(self.n_downsample):
            next_mult = int((mult - 1) * self.mgf) + 1 if i % 2 == 0 else mult
            self.model += [
                ('conv_down_%s' % i, nn.Conv1d(mult, next_mult,
                                               kernel_size=self.conv_size,
                                               padding=self.conv_pad,
                                               dilation=2,
                                               bias=self.use_bias)), 
                ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('relu_down_%s' % i, nn.Tanh()),
                ('pool_down_%s' % i, nn.MaxPool1d(self.pool_size,
                                                  stride=self.pool_stride,
                                                  padding=self.pool_pad,
                                                  return_indices=True)),
            ] 
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            kwargs['input_size'] = mult 
            self.model.append(('trans', self.transformer(**kwargs)))

        # Upsample
        for i in range(self.n_downsample):
            next_mult = int((mult - 1) // self.mgf) + 1 if (self.n_downsample - i) % 2 == 1 else mult
            self.model += [
                ('unpool_up_%s' % i, nn.MaxUnpool1d(self.pool_size,
                                                    stride=self.pool_stride,
                                                    padding=self.pool_pad)),
                ('conv_up_%s' % i, nn.ConvTranspose1d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      padding=self.conv_pad,
                                                      dilation=2,
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('relu_up_%s' % i, nn.Tanh())
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
            if 'pool_down' in name:
                input, indices = module(input)
                self.indices.append(indices)
            elif 'unpool_up' in name:
                indices = self.indices.pop()
                input = module(input, indices)
            else:
                input = module(input)
        return input 
