import torch.nn as nn

def flip_gradient(tensor, l=1.0):  
    def hook(grad):
        return  -l * grad.clone()
    return tensor.register_hook(hook)



class DomainConfusion(nn.Module):
    def __init__(self, layers, num_domain, in_channel, out_channel, input_size):
        super(DomainConfusion, self).__init__()

        model = [
            nn.Conv1d(in_channel, out_channel, kernel_size=2),
            nn.ELU(inplace=True)
        ]
        for i in range(1, layers):
            model += [
                nn.Conv1d(out_channel, out_channel, kernel_size=2),
                nn.ELU(inplace=True)
            ]

        model += [
            Flatten(),
            nn.Linear(out_channel * (input_size - layers), num_domain),
            nn.Tanh(),
            nn.Linear(num_domain, num_domain)
        ]

        self.model = nn.Sequential(*model) 

    def forward(self, i):
        flip_gradient(i, 1e-2)
        i = self.model(i)
        return i



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view((1, -1)) 
