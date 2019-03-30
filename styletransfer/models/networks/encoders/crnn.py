import torch
import torch.nn as nn
from ..util import option_setter


options = {
    'ngf': 1025,
    'mgf': 2,
    'conv_size': 3,
    'conv_pad': 2,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 4,
    'use_bias': False,
    'shrinking_filter': False,
    'reshaped': False,
    'transformer': None,
    'tensor_size': 1025,
    'num_rnn_layers': 1,
}


class CRNNEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(CRNNEncoder, self).__init__()

        option_setter(self, options, kwargs)

        if self.shrinking_filter:
            self.conv_pad = 3 * 2 ** (self.n_downsample - 1)
            self.conv_size = 2 * self.conv_pad + 1

        self.indices = list()

        mult = self.tensor_size
        # Downsample
        self.model = []
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
                # (N, Freq, Len) -> (N, Len, Freq)
                ('lstm_down_%s' % i, nn.LSTM(input_size=next_mult,
                                             hidden_size=next_mult,
                                             num_layers=self.num_rnn_layers,
                                             batch_first=True,
                                             bias=self.use_bias)),
            ]
            if self.shrinking_filter:
                self.conv_size = self.conv_pad + 1
                self.conv_pad = self.conv_pad // 2
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            self.model.append(self.transformer)

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
                                                      stride=2,
                                                      dilation=2,
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('lstm_up_%s' % i, nn.LSTM(input_size=next_mult,
                                           hidden_size=next_mult,
                                           num_layers=self.num_rnn_layers,
                                           batch_first=True,
                                           bias=self.use_bias))
            ]
            mult = next_mult 

        for name, module in self.model:
            self.add_module(name, module)

    def forward(self, input):
        for name, module in self.model:
            if 'lstm' in name:
                input = input.permute(0, 2, 1)
                input, _ = module(input)
                input = input.permute(0, 2, 1)
            else:
                input = module(input)
        return input
