import torch.nn as nn

options = {
    'num_trans_layers': 9,
    'input_size': 1025,
}

class LSTMTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMTransformer, self).__init__()

        option_setter(self, options, kwargs)

        self.model = nn.LSTM(input_size=self.input_size,
                             hidden_size=self.input_size,
                             num_layers=self.num_trans_layers,
                             batch_first=True,
                             bias=self.use_bias)

    def forward(self, input):
        input, _ = self.model(input)
        return input
