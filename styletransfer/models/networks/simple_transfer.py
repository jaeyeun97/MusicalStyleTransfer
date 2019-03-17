import torch
import torch.nn as nn
import torch.nn.init as init


class ChannelTranspose(nn.Module):
    """ Transpose (N, C, H, W)->(N, H, C, W) in accordance with github.com/DmitryUlyanov"""
    def __init__(self):
        super(ChannelTranspose, self).__init__()
    
    def forward(self, tensor):
        return tensor.permute(0, 2, 1, 3)


class RandomTransferNetwork(nn.Module):
    def __init__(self, input_nc=2, nf=64, transpose=False):
        """ If setting transpose, remember to use input_nc=nfft//2+1, nf=4096 or higher"""
        super(RandomTransferNetwork, self).__init__() 
        nets = list()
        ks = 11
        if transpose:
           nets.append(ChannelTranspose()) 
           ks = (1, 11)  
        conv = nn.Conv2d(input_nc, nf, kernel_size=ks, stride=1, group=2)
        nets.append(conv)
        nets.append(nn.LeakyReLU()) 
        init.xavier_normal_(conv.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
        self.net = nn.Sequential(*nets)

    def forward(self, i):
        return self.net(i)


__all__ = [RandomTransferNetwork]
