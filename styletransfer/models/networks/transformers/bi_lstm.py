import torch.nn as nn

options = {
    'num_trans_layers': 9,
    'input_size': 1025,
}

class BiLSTMTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(BiLSTMTransformer, self).__init__()

        option_setter(self, options, kwargs)

        self.model = nn.LSTM(input_size=self.input_size,
                             hidden_size=self.input_size,
                             num_layers=self.num_trans_layers,
                             batch_first=True,
                             bidirectioanl=True,
                             bias=self.use_bias)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        input, _ = self.model(input)
        input = input.permute(0, 2, 1)
        return input
