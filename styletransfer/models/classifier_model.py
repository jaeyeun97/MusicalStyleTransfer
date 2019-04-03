import torch
from torch.nn import MSELoss
from .base_model import BaseModel
from .discriminator import getDiscriminator
from .loss import ClassificationLoss


class ClassifierModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        super(ClassifierModel, self).__init__(opt)
        self.loss_names = ['D_A', 'D_B']
        self.output_names = []
        self.model_names = ['D']

        self.netD = getDiscriminator(opt, self.devices[0])
        self.criterion = MSELoss() # ClassificationLoss().to(self.devices[0])
        self.A_label = torch.zeros(1).to(self.devices[0])
        self.B_label = torch.ones(1).to(self.devices[0])
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.netD.parameters())
            self.optimizers = [self.optimizer]

    def set_input(self, input):
        _, _, self.A = self.preprocess(input[0])
        _, _, self.B = self.preprocess(input[1])
        self.A = self.A.to(self.devices[0])
        self.B = self.B.to(self.devices[0])
    def train(self):
        pred_A = self.netD(self.A)
        pred_B = self.netD(self.B)

        self.optimizer.zero_grad()   # set D_A and D_B's gradients to zero
        self.loss_D_A = self.criterion(pred_A, self.A_label) 
        self.loss_D_B = self.criterion(pred_B, self.B_label)
        self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D.backward()
        self.optimizer.step() 

    def test(self):
        pred_A = self.netD(self.A)
        pred_B = self.netD(self.B) 
        return pred_A, pred_B 
