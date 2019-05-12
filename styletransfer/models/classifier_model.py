import torch
import torch.nn as nn
from adabound import AdaBound
from .base_model import BaseModel
from .discriminator import getDiscriminator


class ClassifierModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        opt, _ = parser.parse_known_args()
        parser.set_defaults(preprocess=opt.preprocess+',stft', flatten=True)
        parser.add_argument('--sigmoid', action='store_true', help='sigmoid')
        return parser

    def __init__(self, opt):
        super(ClassifierModel, self).__init__(opt)
        self.loss_names = ['D_A', 'D_B']
        self.model_names = ['D']

        self.netD = getDiscriminator(opt, self.devices[0])
        # self.criterion = nn.CrossEntropyLoss() 
        self.A_target = torch.zeros(opt.batch_size).to(self.devices[0])
        self.B_target = torch.ones(opt.batch_size).to(self.devices[0])
        self.criterion = nn.MSELoss()

        if self.isTrain:
            # self.optimizer = torch.optim.Adam(self.netD.parameters())
            self.optimizer = AdaBound(self.netD.parameters(), lr=opt.lr)
            self.optimizers = [self.optimizer]

    def set_input(self, input):
        self.A, _ = input[0]
        self.B, _ = input[1]
        self.A = self.A.to(self.devices[0])
        self.B = self.B.to(self.devices[0])

    def train(self):
        pred_A = self.netD(self.A)
        pred_B = self.netD(self.B)
         
        self.optimizer.zero_grad()   # set D_A and D_B's gradients to zero
        self.loss_D_A = self.criterion(pred_A, self.A_target) 
        self.loss_D_B = self.criterion(pred_B, self.B_target)
        self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D.backward()
        self.optimizer.step() 

    def test(self):
        pred_A = self.netD(self.A)
        pred_B = self.netD(self.B) 
        return pred_A, pred_B 
