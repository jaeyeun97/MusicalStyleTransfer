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
    'tensor_size': 1025,
}

class Conv2dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Conv2dEncoder, self).__init__()

        option_setter(self, options, kwargs) 

        self.indices = list()

        # Downsample
        mult = self.ngf
        self.model = [
            ('pad_init', nn.ReflectionPad2d(3)),
            ('conv_init', nn.Conv2d(1, mult,
                                    kernel_size=7,
                                    bias=self.use_bias)),
            ('norm_init', self.norm_layer(mult)),
            ('relu_init', nn.ReLU(True)),
        ]

        for i in range(self.n_downsample):
            next_mult = mult * 2
            self.model += [
                ('conv_down_%s' % i, nn.Conv2d(mult, next_mult,
                                               kernel_size=3,
                                               padding=1,
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
        print(self.transformer)
        if self.transformer is not None:
            ts = (self.tensor_size - 1) // (2 ** self.n_downsample) + 1
            kwargs['input_size'] = (ts, ts)
            kwargs['channel_size'] = mult 
            self.model.append(('trans', self.transformer(**kwargs)))
 
            if 'Skip' in self.transformer.__name__:
                mult = mult * kwargs['num_trans_layers']

        # Upsample
        for i in range(self.n_downsample): 
            next_mult = mult // 2
            self.model += [
                # ('unpool_up_%s' % i, nn.MaxUnpool2d(self.pool_size,
                #                                     stride=self.pool_stride,
                #                                     padding=self.pool_pad)),
                ('conv_up_%s' % i, nn.ConvTranspose2d(mult, next_mult,
                                                      kernel_size=3,
                                                      padding=1,
                                                      stride=2,
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('relu_up_%s' % i, nn.ReLU(True))
            ]
            mult = next_mult

        self.model += [
            ('pad_final', nn.ReflectionPad2d(3)),
            ('conv_final', nn.Conv2d(mult, 1,
                                     kernel_size=7,
                                     bias=self.use_bias)),
            ('tanh', nn.Tanh())
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
