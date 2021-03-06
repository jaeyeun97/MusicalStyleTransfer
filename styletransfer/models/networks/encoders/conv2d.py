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
    'tanh': False
}


class Conv2dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Conv2dEncoder, self).__init__()

        option_setter(self, options, kwargs)

        # Downsample
        mult = self.ngf
        self.model = [
            ('conv_init', nn.Conv2d(1, mult,
                                    kernel_size=7,
                                    padding=3,
                                    bias=self.use_bias)),
            # ('norm_init', self.norm_layer(mult)),
            ('relu_init', nn.ReLU(True)),
        ]

        conv_sizes = list()
        height = self.tensor_height
        width = self.tensor_width
        for i in range(self.n_downsample):
            next_mult = mult * 2
            kh = 4 if height % 2 == 0 else 3
            kw = 4 if width % 2 == 0 else 3
            conv_size = (kh, kw)
            self.model += [
                ('conv_down_%s' % i, nn.Conv2d(mult, next_mult,
                                               kernel_size=conv_size,
                                               padding=1,
                                               stride=2,
                                               bias=self.use_bias)),
                # ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('relu_down_%s' % i, nn.ReLU(True))
            ]
            conv_sizes.append(conv_size)
            height = int((height - kh + 2) / 2 + 1)
            width = int((width - kw + 2) / 2 + 1)
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            kwargs['channel_size'] = mult
            self.model.append(('trans', self.transformer(**kwargs)))

        # Upsample
        for i in range(self.n_downsample):
            next_mult = mult // 2
            self.model += [
                ('conv_up_%s' % i, nn.ConvTranspose2d(mult, next_mult,
                                                      kernel_size=conv_sizes.pop(),
                                                      padding=1,
                                                      stride=2,
                                                      bias=self.use_bias)),
                # ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('relu_up_%s' % i, nn.ReLU(True))
            ]
            mult = next_mult

        self.model.append(('conv_final', nn.Conv2d(mult, 1,
                                                   kernel_size=7,
                                                   padding=3,
                                                   bias=self.use_bias)))

        if self.tanh:
            self.model.append(('tanh_final', nn.Tanh()))

        for name, module in self.model:
            if 'conv_final' in name: 
                if self.tanh:
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
                else:
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('linear'))
            elif 'conv' in name:
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))

            self.add_module(name, module)

    def forward(self, input):
        flag = False
        if len(input.size()) < 4:
            flag = True
            input = input.unsqueeze(1)
        for name, module in self.model: 
            input = module(input)
        return input.squeeze(1) if flag else input
