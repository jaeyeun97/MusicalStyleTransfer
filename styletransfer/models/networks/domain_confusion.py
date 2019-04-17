import math
import torch.nn as nn
from torch.autograd import Function

def hook_factory(l=1.0):  
    def hook(grad):
        return -l * grad.clone()
    return hook 



class DomainConfusion(nn.Module):
    def __init__(self, layers, num_domain, in_channel, out_channel, input_size):
        super(DomainConfusion, self).__init__()
        self.device = None
        self.num_domain = num_domain

        conv = nn.Conv1d(in_channel, out_channel, groups=1, kernel_size=3) 
        elu = nn.ELU()
        nn.init.xavier_uniform_(conv.weight, gain=math.sqrt(1.55 / in_channel))
        # nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
        # model = [GradientFlip(1e-2), conv, elu]
        model = [conv, elu]

        for i in range(1, layers - 1):
            conv = nn.Conv1d(out_channel, out_channel, groups=1, kernel_size=3)
            elu = nn.ELU()
            nn.init.xavier_uniform_(conv.weight, gain=math.sqrt(1.55 / out_channel))
            # nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            model += [conv, elu]


        conv = nn.Conv1d(out_channel, num_domain, groups=1, kernel_size=3) 
        nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain('linear'))

        model += [conv,
            # nn.AvgPool1d(input_size - 2 * layers),
            # Flatten(),
            # nn.Linear(num_domain * (input_size - 2 * layers), 2),
        ]

        self.model = nn.Sequential(*model) 

    def forward(self, i):
        i = i.clone()
        i.register_hook(hook_factory(0.01))
        # GRL.apply(i, 0.1)
        i = self.model(i)
        i = i.mean(dim=2)
        return i # .view(1, self.num_domain)

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
        return i.view_as(i)
    
    def backward(self, grad):
        return -1 * self.l * grad.clone()

class GRL(Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None
