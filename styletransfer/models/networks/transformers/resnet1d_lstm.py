import torch.nn as nn
from ..util import option_setter

options = {
    'num_trans_layers': 9,
    'input_size': 1025,
    'num_rans_layer': 9,
    'input_size': 1025,
    'use_bias': True,
}

class LSTMTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMTransformer, self).__init__()

        option_setter(self, options, kwargs)

        self.model = list()

        kwargs['nc'] = self.input_size
        self.model +=
                    nn.LSTM(input_size=self.input_size,
                             hidden_size=self.input_size,
                             num_layers=self.num_trans_layers,
                             batch_first=True,
                             bias=self.use_bias)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        input, _ = self.model(input)
        input = input.permute(0, 2, 1)
        return input
