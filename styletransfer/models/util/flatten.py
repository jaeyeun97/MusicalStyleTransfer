import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(2, -1)

