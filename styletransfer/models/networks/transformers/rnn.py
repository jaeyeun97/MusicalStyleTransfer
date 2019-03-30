import torch.nn as nn

class RNNTransformer(nn.Module):
    def __init__(self):
        super(RNNTransformer, self).__init__()

        self.model = []
