import torch.nn as nn

msg = """----------------------
Layer Name: {}
Size of Input: {}
----------------------"""


class DebugPrintLayer(nn.Module):
    def __init__(self, layer_name):
        super(DebugPrintLayer, self).__init__()
        self.layer_name = layer_name
    
    def forward(self, x):
        print(msg.format(self.layer_name, x.size()))
        return x
    
