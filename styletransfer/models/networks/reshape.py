import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, sizes):
        super(Reshape, self).__init__()
        self.sizes = sizes

    def forward(self, input):
        return input.permute(*self.sizes)

