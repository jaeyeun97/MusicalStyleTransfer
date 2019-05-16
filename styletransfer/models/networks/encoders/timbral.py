import torch.nn as nn
from ..util import option_setter


options = {
    'ngf': 64,
    'conv_size': 3,
    'conv_pad': 1,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 2,
    'use_bias': False,
    'shrinking_filter': False,
    'reshaped': False,
    'transformer': None,
    'tensor_height': 1025,
    'tensor_width': 1025,
    'tanh': False,
}


class TimbralEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(TimbralEncoder, self).__init__()

        option_setter(self, options, kwargs)

        # Downsample
        mult = self.ngf
        self.model = [
            ('conv_init', nn.Conv2d(1, mult,
                                    kernel_size=4,
                                    bias=self.use_bias)),
            ('relu_init', nn.ReLU(True)),
        ]

        height = self.tensor_height - 3
        conv_sizes = list()
        for i in range(self.n_downsample):
            next_mult = mult * 2
            conv_size = 4 if height % 2 == 0 else 3
            height = int((height - conv_size) / 2 + 1)
            conv_sizes.append(conv_size)
            self.model += [
                ('conv_down_%s' % i, nn.Conv2d(mult, next_mult,
                                               kernel_size=conv_size,
                                               stride=(2, 1),
                                               bias=self.use_bias)),
                ('relu_down_%s' % i, nn.ReLU(True))
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
                ('conv_up_%s' % i, nn.ConvTranspose2d(mult, next_mult,
                                                      kernel_size=conv_sizes.pop(),
                                                      stride=(2, 1),
                                                      bias=self.use_bias)),
                ('relu_up_%s' % i, nn.ReLU(True))
            ]
            mult = next_mult

        self.model.append(('conv_final', nn.ConvTranspose2d(mult, 1,
                                                            kernel_size=4,
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
