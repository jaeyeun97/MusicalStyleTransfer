import torch.nn as nn
from ..util import option_setter
from .vconv2d import VConv2d


options = { 
    'ngf': 32,
    'conv_size': 3,
    'conv_pad': 2,
    'pool_size': 3,
    'pool_pad': 1,
    'pool_stride': 2,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 4,
    'use_bias': False,
    'shrinking_filter': False,
    'reshaped': False,
    'transformer': None,
    'tensor_size': 1025,
}

class VConv2dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(VConv2dEncoder, self).__init__()

        option_setter(self, options, kwargs) 

        self.indices = list()

        # Downsample
        mult = self.ngf
        self.model = [
            ('conv_init', VConv2d(1, mult,
                                  kernel_size=self.conv_size,
                                  padding=self.conv_pad,
                                  dilation=2,
                                  bias=self.use_bias))
        ]

        for i in range(self.n_downsample):
            next_mult = mult * 2
            self.model += [
                ('conv_down_%s' % i, nn.Conv2d(mult, next_mult,
                                               kernel_size=self.conv_size,
                                               padding=self.conv_pad,
                                               dilation=2,
                                               bias=self.use_bias)), 
                # ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('relu_down_%s' % i, nn.LeakyReLU(0.2, True)),
                ('pool_down_%s' % i, nn.MaxPool2d(self.pool_size,
                                                  stride=self.pool_stride,
                                                  padding=self.pool_pad,
                                                  return_indices=True)),
            ]
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            ts = (self.tensor_size - 1) // (2 ** self.n_downsample) + 1
            kwargs['input_size'] = (ts, ts)
            kwargs['channel_size'] = mult 
            self.model.append(('trans', self.transformer(**kwargs)))

        # Upsample
        for i in range(self.n_downsample): 
            next_mult = mult // 2
            self.model += [
                ('unpool_up_%s' % i, nn.MaxUnpool2d(self.pool_size,
                                                    stride=self.pool_stride,
                                                    padding=self.pool_pad)),
                ('conv_up_%s' % i, nn.ConvTranspose2d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      padding=self.conv_pad,
                                                      dilation=2,
                                                      bias=self.use_bias)),
                # ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('relu_up_%s' % i, nn.LeakyReLU(0.2, True))
            ]
            mult = next_mult

        self.model += [
            ('conv_final', nn.Conv2d(mult, 1,
                                     kernel_size=self.conv_size,
                                     padding=self.conv_pad,
                                     dilation=2,
                                     bias=self.use_bias)),
            # ('tanh', nn.Tanh())
        ]

        for name, module in self.model:
            self.add_module(name, module)

    def forward(self, input):
        input = input.unsqueeze(1)
        for name, module in self.model:
            if 'pool_down' in name:
                input, indices = module(input)
                self.indices.append(indices)
            elif 'unpool_up' in name:
                indices = self.indices.pop()
                input = module(input, indices)
            else:
                input = module(input)
        return input.squeeze(1)
