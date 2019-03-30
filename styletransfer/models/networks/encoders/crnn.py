import torch
import torch.nn as nn
from ..util import option_setter
from ..reshape import Reshape


options = {
    'conv_size': 3,
    'conv_pad': 2,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 4,
    'use_bias': False,
    'shrinking_filter': False,
    'reshaped': False,
    'transformer': None,
    'tensor_size': 1025,
    'rnn_layers': 1,
    'rnn_hidden_size': 1024
}


class CRNNEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(CRNNEncoder, self).__init__()

        option_setter(self, options, kwargs)

        if self.shrinking_filter:
            self.conv_pad = 3 * 2 ** (self.n_downsample - 1)
            self.conv_size = 2 * self.conv_pad + 1

        self.indices = list()

        mult = self.ngf
        self.model = [
            ('conv_init', nn.Conv1d(self.tensor_size, mult,
                                    kernel_size=self.conv_size,
                                    padding=self.conv_pad,
                                    dilation=2,
                                    bias=self.use_bias)),
            ('norm_init', self.norm_layer(mult)),
            ('resh_init', Reshape(0, 2, 1)),
            ('lstm_init', nn.LSTM(input_size=mult,
                                  hidden_size=mult - 1,
                                  num_layers=self.num_rnn_layers,
                                  batch_first=True,
                                  bias=self.use_bias))
        ]

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
                ('resh_down_%s' % i, Reshape((0, 2, 1))),
                ('lstm_down_%s' % i, nn.LSTM(input_size=next_mult,
                                             hidden_size=next_mult - 1,
                                             num_layers=self.num_rnn_layers,
                                             batch_first=True,
                                             bias=self.use_bias)),
            ]
            if self.shrinking_filter:
                self.conv_size = self.conv_pad + 1
                self.conv_pad = self.conv_pad // 2

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
                ('resh_down_%s' % i, Reshape((0, 2, 1))),
                ('lstm_up_%s' % i, nn.LSTM(input_size=next_mult,
                                           hidden_size=next_mult - 1,
                                           num_layers=self.num_rnn_layers,
                                           batch_first=True,
                                           bias=self.use_bias))
            ]
            mult = next_mult

        self.model += [
            ('conv_final', nn.ConvTranspose1d(mult, self.tensor_size,
                                              kernel_size=self.conv_size,
                                              padding=self.conv_pad,
                                              dilation=2,
                                              bias=self.use_bias)),
            ('norm_final', self.norm_layer(self.tensor_size)),
            ('resh_final', Reshape((0, 2, 1))),
            ('lstm_final', nn.LSTM(input_size=self.tensor_size,
                                   hidden_size=self.tensor_size - 1,
                                   num_layers=self.num_rnn_layers,
                                   batch_first=True,
                                   bias=self.use_bias))
        ]

        for name, module in self.model:
            self.add_module(name, module)

    def forward(self, input):
        phase = input[:, 1, :, :]
        input = input[:, 0, :, :]
        for name, module in self.model:
            if 'lstm' in name:
                input, _ = module(torch.stack((input, phase), 3))
                phase = input[:, :, :, 1]
                input = input[:, :, :, 0]
            elif 'transformer' in name:
                input = module(torch.stack((input, phase), 1))
                phase = input[:, 1, :, :]
                input = input[:, 0, :, :]
            else:
                input = module(input)
        input = torch.stack((input, phase), 1)
        return input
