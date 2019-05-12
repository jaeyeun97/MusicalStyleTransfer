import torch.nn as nn
from ..util import option_setter


options = { 
    'ngf': 64,
    'conv_size': 3,
    'conv_pad': 1,
    'pool_size': (3, 1),
    'pool_pad': (1, 0),
    'pool_stride': (2, 1),
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 2,
    'use_bias': False,
    'shrinking_filter': False,
    'reshaped': False,
    'transformer': None,
    'tensor_height': 1025,
    'tensor_width': 1025,
}

class Conv2dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Conv2dEncoder, self).__init__()

        option_setter(self, options, kwargs) 

        self.indices = list()

        # Downsample
        mult = self.ngf
        self.model = [
            ('conv_init', nn.Conv2d(1, mult,
                                    kernel_size=7,
                                    padding=3,
                                    bias=self.use_bias)),
            ('norm_init', self.norm_layer(mult)),
            ('relu_init', nn.ReLU(True)),
        ]

        for i in range(self.n_downsample):
            next_mult = mult * 2
            self.model += [
                ('conv_down_%s' % i, nn.Conv2d(mult, next_mult,
                                               kernel_size=self.conv_size,
                                               padding=self.conv_pad,
                                               stride=2,
                                               bias=self.use_bias)), 
                ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('relu_down_%s' % i, nn.ReLU(True)),
                # ('pool_down_%s' % i, nn.MaxPool2d(self.pool_size,
                #                                   stride=self.pool_stride,
                #                                   padding=self.pool_pad,
                #                                   return_indices=True)),
            ]
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            kwargs['channel_size'] = mult 
            self.model.append(('trans', self.transformer(**kwargs)))
 
 
        # Upsample
        for i in range(self.n_downsample): 
            next_mult = mult // 2
            self.model += [
                # ('unpool_up_%s' % i, nn.MaxUnpool2d(self.pool_size,
                #                                     stride=self.pool_stride,
                #                                     padding=self.pool_pad)),
                ('conv_up_%s' % i, nn.ConvTranspose2d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      padding=self.conv_pad,
                                                      stride=2,
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('relu_up_%s' % i, nn.ReLU(True))
            ]
            mult = next_mult

        self.model += [
            ('conv_final', nn.Conv2d(mult, 1,
                                     kernel_size=7,
                                     padding=3,
                                     bias=self.use_bias)),
            ('tanh', nn.Tanh())
        ]

        for name, module in self.model:
            self.add_module(name, module)

    def forward(self, input):
        flag = False
        if len(input.size()) < 4:
            flag = True
            input = input.unsqueeze(1)
        for name, module in self.model:
            # if 'pool_down' in name:
            #     input, indices = module(input)
            #     self.indices.append(indices)
            # elif 'unpool_up' in name:
            #     indices = self.indices.pop()
            #     input = module(input, indices)
            # else:
            input = module(input)
        return input.squeeze(1) if flag else input
