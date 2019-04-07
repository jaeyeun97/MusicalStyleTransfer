import itertools
import torch
import numpy as np
import torch.nn as nn
from ..util.audio import pitch_shift, hz_to_mel
from .base_model import BaseModel
from .networks.encoders import TemporalEncoder
from .networks.domain_confusion import DomainConfusion
from .networks.wavenet import WaveNet

MU = 128

def mulaw(x, MU):
    return np.sign(x) * np.log(1. + MU * np.abs(x)) / np.log(1. + MU)

class TranslatorModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults()
        if is_train:
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['loss_C_A', 'loss_C_B', 'loss_D_A', 'loss_D_B']
        if opt.isTrain:
            self.output_names = ['real_A', 'real_B', 'rec_A', 'rec_B']
        else:
            self.output_names = ['real_A', 'real_B', 'fake_A', 'fake_B']
        self.model_names = ['E', 'C', 'D_A', 'D_B']

        self.netE = TemporalEncoder(**{
            'bottleneck_width': 64,
            'pool_length': 512, 
        }).to(self.devices[0])
        self.netC = DomainConfusion(3, 2, 64, 128, self.opt.audio_length // 512).to(self.devices[0])
        self.netD_A = WaveNet(128, 9, 10, 64, 128, 128, 64, 1, 1).to(self.devices[-1])
        self.netD_B = WaveNet(128, 9, 10, 64, 128, 128, 64, 1, 1).to(self.devices[-1])
        self.upsample = nn.Upsample(scale_factor=1024, mode='nearest')

        if self.isTrain:
            self.A_target = torch.LongTensor([0]).to(self.devices[0])
            self.B_target = torch.LongTensor([1]).to(self.devices[0])
            self.criterionDomConfusion = nn.CrossEntropyLoss(reduction='mean')
            self.criterionReconstruction = nn.CrossEntropyLoss(reduction='mean')
            self.optimizer = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netC.parameters(), self.netD_A.parameters(), self.netD_B.parameters()))
            self.optimizers = [self.optimizer] 

    def set_input(self, input): 
        self.real_A = self.preprocess(input[0]).to(self.devices[0])
        self.real_B = self.preprocess(input[1]).to(self.devices[0])
        
    def preprocess(self, y):
        y = y.squeeze().numpy()
        if 'shift' in self.preprocesses: 
            print('Pitch Shift Activated')
            y, self.shift_steps = pitch_shift(y, self.opt.sample_rate)
        if 'mel' in self.preprocesses:
            y = hz_to_mel(y) 
        y = mulaw(y, MU)
        return torch.from_numpy(y).unsqueeze(0)

    def get_indices(self, y):
        y = y + 0.5 * MU
        return y.clamp(0, MU)

    def to_onehot(self, y):
        y = y + 0.5 * MU
        y = y.type(torch.LongTensor).clamp(0, MU).view(-1, 1)
        y = torch.zeros(y.size()[0], MU).scatter_(1, y, 1)
        return y

    def postprocess(self, y):
        pass

    def train(self):
        self.optimizer.zero_grad()
        encoded_A = self.netE(self.real_A.unsqueeze(1) / MU)
        encoded_B = self.netE(self.real_B.unsqueeze(1) / MU) 

        self.loss_C_A = self.criterionDomConfusion(self.netC(encoded_A), self.A_target)
        self.loss_C_B = self.criterionDomConfusion(self.netC(encoded_B), self.B_target)

        encoded_A = self.upsample(encoded_A).to(self.devices[-1])
        encoded_B = self.upsample(encoded_B).to(self.devices[-1])
        self.real_A = self.to_onehot(self.real_A).to(self.devices[-1])
        self.real_B = self.to_onehot(self.real_B).to(self.devices[-1]) 

        self.rec_A = self.netD_A((encoded_A, self.real_A)) 
        self.rec_B = self.netD_B((encoded_B, self.real_B))

        self.loss_D_A = self.criterionReconstruction(self.rec_A, self.get_indices(self.real_A).to(self.devices[0]))
        self.loss_D_B = self.criterionReconstruction(self.rec_B, self.get_indices(self.real_B).to(self.devices[0]))

        self.loss = self.loss_C_A + self.loss_C_B + self.loss_D_A + self.loss_D_B
        self.loss.backward()
        self.optimizer.step()

    def test(self):
        encoded_A = self.netE(self.real_A / MU)
        encoded_B = self.netE(self.real_B / MU) 

        encoded_A = self.upsample(encoded_A)
        encoded_B = self.upsample(encoded_B)

        self.fake_A = self.netD_A((encoded_B, self.to_onehot(self.real_A).to(self.devices[0])))
        self.fake_B = self.netD_B((encoded_A, self.to_onehot(self.real_B).to(self.devices[0])))
