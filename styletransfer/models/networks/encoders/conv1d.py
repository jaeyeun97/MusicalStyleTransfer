import torch.nn as nn
from ..util import option_setter

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
    'tensor_height': 1025,
    'mgf': 0.5  # out_channel / in_channel
}


class Conv1dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Conv1dEncoder, self).__init__()

        option_setter(self, options, kwargs)

        mult = self.tensor_height
        # Downsample
        for i in range(self.n_downsample):
            next_mult = min(self.ngf, int(mult * self.mgf))
            self.model += [
                ('conv_down_%s' % i, nn.Conv1d(mult, next_mult,
                                               kernel_size=self.conv_size,
                                               bias=self.use_bias)),
                # ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('relu_down_%s' % i, nn.ReLU(True)),
            ]
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            kwargs['input_size'] = mult
            self.model.append(('trans', self.transformer(**kwargs)))

        # Upsample
        for i in range(self.n_downsample):
            next_mult = int(mult // self.mgf)
            self.model += [
                ('conv_up_%s' % i, nn.ConvTranspose1d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      stride=2,
                                                      bias=self.use_bias)),
                # ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('relu_up_%s' % i, nn.ReLU(True))
            ]
            mult = next_mult

        self.model.append(('conv_final', nn.ConvTranspose1d(mult, self.tensor_height,
                                                            kernel_size=self.conv_size,
                                                            bias=self.use_bias)))

        if self.tanh:
            self.model.append('tanh', nn.Tanh())

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
        for name, module in self.model:
            input = module(input)
        return input
