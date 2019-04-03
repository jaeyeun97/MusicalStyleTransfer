import torch
import torch.nn as nn


class VConv2d(nn.Module):
    def __init__(self, input_nc, output_nc, input_size,
                 kernel_size=3, padding=0, dilation=1, stride=1, use_bias=False):
        super(VConv1d, self).__init__()
        self.convs = list()
        self.kernel_size = kernel_size

        self.pad = None
        if type(padding) == tuple:
            self.pad = ZeroPad(padding[0], 0) 
            padding = padding[1]
        else:
            self.pad = ZeroPad(padding, 0)
        for i in range(input_size):
            self.convs.append(nn.Conv2d(input_nc, output_nc,
                                        kernel_size=kernel_size,
                                        padding=(0, padding),
                                        dilation=dilation,
                                        stride=stride,
                                        use_bias=use_bias)
        self.width = 1 + dilation * (kernel_size - 1)
          
    def forward(self, input):
        results = list()
        for i in range(input.size(2) - self.kernel_size):
            results.append(self.convs[i](input[:, :, i:i+self.kernel_size, :]))
        return torch.stack(results, 2)
