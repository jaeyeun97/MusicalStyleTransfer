import torch.nn as nn


class Resize(nn.Module):
    def __init__(self, sizes):
        self.sizes = sizes

    def forward(self, input):
        return input.reshape(self.sizes)

