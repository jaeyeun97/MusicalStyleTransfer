import torch
import torch.nn as nn
from ..util import option_setter


options = {
    'conv_size': 3,
    'tensor_size': 1025,
    'bottleneck_width': 129,
    'layers': 4
}


class Res1dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Res1dEncoder, self).__init__()

        option_setter(self, options, kwargs)
 
        self.encodeList = nn.ModuleList()

        self.encodeList.append(nn.Conv1d(self.tensor_size, self.tensor_size,
                                         kernel_size=self.conv_size,
                                         padding=((self.conv_size - 1) // 2)))
 
        dilation = 1
        for i in range(self.layers):
            self.encodeList.extend([
                nn.ReLU(),
                nn.Conv1d(self.tensor_size, self.tensor_size,
                          kernel_size=self.conv_size,
                          padding=((self.conv_size - 1) * dilation // 2),
                          dilation=dilation), 
                nn.ReLU(), 
                nn.Conv1d(self.tensor_size, self.tensor_size, kernel_size=1)
            ])
            dilation *= 2

        self.pool_length = self.tensor_size // 128
        self.bottleneck = nn.Conv1d(self.tensor_size, self.bottleneck_width, kernel_size=1)
        self.pool = nn.AvgPool1d(self.pool_length, stride=self.pool_length)

        ## NO TRANSFORMERS

        self.upsample = nn.Conv1d(self.bottleneck_width, self.tensor_size, kernel_size=1)
        self.inConvs = nn.ModuleList()
        self.resConvs = nn.ModuleList()
        dilation=1
        for i in range(self.layers):
            self.inConvs.append(nn.Conv1d(self.tensor_size, self.tensor_size * 2,
                                          kernel_size=self.conv_size,
                                          padding=((self.conv_size - 1) * dilation // 2),
                                          dilation=dilation))
            self.resConvs.append(nn.Conv1d(self.tensor_size, self.tensor_size, kernel_size=1))
            dilation *= 2

        self.final = nn.Conv1d(self.tensor_size, self.tensor_size, kernel_size=1)

        for module in self.encodeList[:-1]:
            if isinstance(module, nn.Conv1d):
                self.init_weight(module, 'relu')
        self.init_weight(self.encodeList[-1], 'linear')
        self.init_weight(self.bottleneck, 'linear')
        self.init_weight(self.upsample, 'tanh')
        for module in self.inConvs:
            self.init_weight(module, 'linear')
        for module in self.resConvs:
            self.init_weight(module, 'linear')
                          
    
    def forward(self, input):
        for module in self.encodeList:
            input = module(input) 
        input = self.bottleneck(input)
        input = self.pool(input)
        input = nn.functional.interpolate(input, size=self.tensor_size)
        input = self.upsample(input)
        input = torch.tanh(input)

        for i in range(self.layers): 
            # No skips like Wavenet, 
            save = input
            input = self.inConvs[i](input)
            input = torch.tanh(input[:, :self.tensor_size, :]) * \
                    torch.sigmoid(input[:, self.tensor_size:, :])
            input = self.resConvs[i](input) 
            input = input + save

        input = nn.functional.relu(input)
        input = self.final(input)
        return input

    @staticmethod
    def init_weight(module, gain_name='linear'):
        torch.nn.init.xavier_uniform_(
            module.weight, gain=torch.nn.init.calculate_gain(gain_name))
