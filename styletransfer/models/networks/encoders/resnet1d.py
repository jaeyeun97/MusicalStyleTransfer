import torch.nn as nn
from ..util import option_setter
from ..resnet import Resnet1d


options = { 
    'mult': 2, 
    'input_nc': 513,
    'output_nc': 129,
    'num_layers': 9
}

class Resnet1dEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet1dEncoder, self).__init__()

        option_setter(self, options, kwargs) 
 
        ngf = (self.input_nc - 1) * self.mult + 1
        self.model = [nn.Conv1d(self.input_nc, ngf, kernel_size=1)] 
        kwargs['nc'] = ngf
        self.model += [Resnet1d(**kwargs) for i in range(self.num_layers)] 
        self.model += [nn.ConvTranspose1d(ngf, self.output_nc, kernel_size=1)]
        self.model = nn.Sequential(*self.model)
         
    def forward(self, input):
        input = self.model(input)
        return input

    def to(self, device):
        self.device = device
        return super(Resnet1dEncoder, self).to(device)
