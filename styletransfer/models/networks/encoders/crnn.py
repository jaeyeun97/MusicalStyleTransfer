import torch
import torch.nn as nn
from ..util import option_setter


options = {
    'ngf': 2,
    'mgf': 2,
    'conv_size': 3,
    'conv_pad': 2,
    'pool_size': 3,
    'pool_pad': 1,
    'pool_stride': 2,
    'norm_layer': nn.BatchNorm2d,
    'n_downsample': 4,
    'use_bias': False,
    'reshaped': False,
    'transformer': None,
    'tensor_size': 1025,
    'num_rnn_layers': 1,
}


class CRNNEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(CRNNEncoder, self).__init__()

        option_setter(self, options, kwargs)

        self.indices = list()

        mult = self.tensor_size * self.ngf
        # Downsample
        self.model = [
            ('conv_init', nn.Conv1d(self.tensor_size, mult,
                                    kernel_size=self.conv_size,
                                    padding=self.conv_pad,
                                    dilation=2,
                                    bias=self.use_bias)),
            ('norm_init', self.norm_layer(mult)),
            ('lstm_init', nn.RNN(input_size=mult,
                                  hidden_size=mult,
                                  num_layers=self.num_rnn_layers,
                                  batch_first=True,
                                  nonlinearity='relu',
                                  bias=self.use_bias)),
        ]

        for i in range(self.n_downsample):
            next_mult = int(mult * self.mgf) # if i % 2 == 0 else mult
            self.model += [
                ('conv_down_%s' % i, nn.Conv1d(mult, next_mult,
                                               kernel_size=self.conv_size,
                                               padding=self.conv_pad,
                                               dilation=2,
                                               bias=self.use_bias)), 
                ('norm_down_%s' % i, self.norm_layer(next_mult)),
                ('lstm_down_%s' % i, nn.RNN(input_size=next_mult,
                                            hidden_size=next_mult,
                                            num_layers=self.num_rnn_layers,
                                            batch_first=True,
                                            nonlinearity='relu',
                                            bias=self.use_bias)), 
                # ('pool_down_%s' % i, nn.MaxPool1d(self.pool_size,
                #                                   stride=self.pool_stride,
                #                                   padding=self.pool_pad,
                #                                   return_indices=True)),
            ]
            mult = next_mult

        # Transformer
        if self.transformer is not None:
            kwargs['input_size'] = mult 
            self.model.append(('trans', self.transformer(**kwargs)))

        # Upsample
        for i in range(self.n_downsample):
            next_mult = int(mult // self.mgf)  # if (self.n_downsample - i) % 2 == 1 else mult
            self.model += [
                # ('unpool_up_%s' % i, nn.MaxUnpool1d(self.pool_size,
                #                                     stride=self.pool_stride,
                #                                     padding=self.pool_pad)),
                ('conv_up_%s' % i, nn.ConvTranspose1d(mult, next_mult,
                                                      kernel_size=self.conv_size,
                                                      padding=self.conv_pad,
                                                      dilation=2,
                                                      bias=self.use_bias)),
                ('norm_up_%s' % i, self.norm_layer(next_mult)),
                ('lstm_up_%s' % i, nn.RNN(input_size=next_mult,
                                          hidden_size=next_mult,
                                          num_layers=self.num_rnn_layers,
                                          batch_first=True,
                                          nonlinearity='relu',
                                          bias=self.use_bias))
            ]
            mult = next_mult 
        self.model += [
            ('conv_final', nn.ConvTranspose1d(mult, self.tensor_size,
                                              kernel_size=self.conv_size,
                                              padding=self.conv_pad,
                                              dilation=2,
                                              bias=self.use_bias)),
            # ('norm_final', self.norm_layer(self.tensor_size)),
            ('lstm_final', nn.RNN(input_size=self.tensor_size,
                                       hidden_size=self.tensor_size,
                                       num_layers=self.num_rnn_layers,
                                       nonlinearity='relu',
                                       batch_first=True,
                                       bias=self.use_bias))
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
            elif 'lstm' in name:
                input = input.permute(0, 2, 1)
                input, _ = module(input)
                input = input.permute(0, 2, 1)
            else:
                input = module(input)
        return input
