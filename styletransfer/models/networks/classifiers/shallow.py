import torch.nn as nn
from ..util import option_setter

options = { 
    'conv_size': 3,
    'tensor_size': 1025,
    'width': 257,
    'layers': 4
}

class ShallowClassifier(nn.Module):
    """Defines a CNN classifier without phase input"""

    def __init__(self, **kwargs):
        
        super(ShallowClassifier, self).__init__()
        option_setter(self, options, kwargs) 

        self.model = [
            nn.Conv1d(self.tensor_size, self.width,
                      kernel_size=self.conv_size),   
            nn.ELU()
        ] 
        for n in range(1, self.layers):
            self.model += [
                nn.Conv1d(self.width, self.width,
                          kernel_size=self.conv_size),   
                nn.ELU()
            ]

        self.model += [ 
            Flatten(),
            nn.Linear(self.width * (self.tensor_size - ((self.conv_size - 1) * self.layers)), 2),
            nn.Tanh(),
            nn.Linear(2, 2)
        ]
 
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        """Standard forward."""
        input = self.model(input)
        return input

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(1, -1) 
