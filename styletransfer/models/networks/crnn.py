import torch.nn as nn
from .util import option_setter
from .reshape import Reshape


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


class CRNNAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(CRNNAutoencoder, self).__init__()

        option_setter(self, options, kwargs)

        # if self.shrinking_filter:
        #     self.conv_pad = 2 ** (self.n_downsample - 1)
        #     self.conv_size = 2 * self.conv_pad + 1

        self.indices = list()

        # Downsample
        # (N, 2, Freq, Len) -> (N, Len, Freq, 2)
        self.model = [('reshape_init', Reshape((0, 3, 2, 1)))]
        mult = self.tensor_size
        for i in range(self.n_downsample):
            next_mult = (mult - 1) // 2 + 1
            self.model += [
                ('conv_down_%s' % i, nn.Conv2d(self.tensor_size, self.tensor_size,
                                               kernel_size=(self.conv_size, 2),
                                               padding=(self.conv_pad, 0),
                                               dilation=(2, 1),
                                               stride=(2, 1),
                                               bias=self.use_bias)),
                ('norm_down_%s' % i, self.norm_layer(mult)),
                ('lstm_down_%s' % i, nn.LSTM(input_size=(next_mult, 2),
                                             hidden_size=next_mult - 1,
                                             num_layers=self.num_rnn_layers,
                                             batch_first=True,
                                             bias=self.use_bias)),
            ]
            # if self.shrinking_filter:
            #     self.conv_size = self.conv_pad + 1
            #     self.conv_pad = self.conv_pad // 2

        # Transformer
        if self.transformer is not None:
            self.model.append(self.transformer)

        # Upsample
        for i in range(self.n_downsample):
            # if self.shrinking_filter:
            #     self.conv_pad = 2 ** i
            #     self.conv_size = 2 * self.conv_pad + 1
            next_mult = (mult - 1) * 2 + 1
            self.model += [
                ('conv_up_%s' % i, nn.ConvTranspose2d(self.tensor_size, self.tensor_size,
                                                      kernel_size=(self.conv_size, 2),
                                                      padding=(self.conv_pad, 0),
                                                      stride=(2, 1),
                                                      dilation=(2, 1),
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('lstm_up_%s' % i, nn.LSTM(input_size=(next_mult, 2),
                                           hidden_size=next_mult - 1,
                                           num_layers=self.num_rnn_layers,
                                           batch_first=True,
                                           bias=self.use_bias))
            ]
            mult = next_mult

        self.model += [('reshape_final', Reshape((0, 3, 2, 1)))]

        for name, module in self.model:
            self.add_module(name, module)

    def forward(self, input):
        for name, module in self.model:
            if 'lstm' in name:
                input, _ = module(input)
            else:
                input = module(input)
        return input
