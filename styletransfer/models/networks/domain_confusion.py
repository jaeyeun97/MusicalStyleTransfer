import math
import torch.nn as nn
from .wavenet import Conv

def hook_factory(l=1.0):  
    def hook(grad):
        return -l * grad.clone()
    return hook 



class DomainConfusion(nn.Module):
    def __init__(self, layers, num_domain, in_channel, out_channel, input_size):
        super(DomainConfusion, self).__init__()
        self.device = None

        model = [
            GradientFlip(1e-2),
            nn.Conv1d(in_channel, out_channel, kernel_size=2, bias=False),
            nn.ELU(),
        ]

        for i in range(1, layers - 1):
            model += [
                nn.Conv1d(out_channel, out_channel, kernel_size=2, bias=False), 
                nn.ELU(),
            ]

        model += [
            nn.Conv1d(out_channel, num_domain, kernel_size=2, bias=False), 
            nn.ELU(),
        ]

        for module in model:
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=math.sqrt(1.55 / module.in_channels))

        self.model = nn.Sequential(*model) 

    def forward(self, i):
        # i = i.clone()
        # i.register_hook(hook_factory(1e-2))
        i = self.model(i)
        i = i.mean(dim=2)
        return i

    def to(self, device):
        self.device = device
        return super(DomainConfusion, self).to(device)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view((1, -1)) 

class GradientFlip(nn.Module):
    def __init__(self, l=1.0):
        super(GradientFlip, self).__init__()
        self.l = l

    def forward(self, i):
        return i.clone()
    
    def backward(self, grad):
        return -self.l * grad
