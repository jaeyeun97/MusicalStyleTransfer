import itertools
import torch
import torch.nn as nn
from .base_model import BaseModel
from .networks.encoders import TemporalEncoder
from .networks.domain_confusion import DomainConfusion
from .networks.wavenet import WaveNet

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
        opt, _ = parser.parse_known_args() 
        parser.set_defaults(preprocess=opt.preprocess
                                          .replace('mel', '')
                                          .replace('normalize', '')
                                          .replace('stft', '') + ',mulaw')
        parser.add_argument('--bottleneck', type=int, default=64, help='bottle neck for temporal encoder')
        parser.add_argument('--pool_length', type=int, default=512, help='pool length')
        
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['C_A', 'C_B', 'D_A', 'D_B']
        if opt.isTrain:
            self.output_names = ['real_A', 'real_B', 'rec_A', 'rec_B']
        else:
            self.output_names = ['real_A', 'real_B', 'fake_B', 'fake_A']
        self.params_names = ['params_A', 'params_B'] * 2
        self.model_names = ['E', 'C', 'D_A', 'D_B']

        self.netE = TemporalEncoder(**{
            'bottleneck_width': opt.bottleneck,
            'pool_length': opt.pool_length, 
        }).to(self.devices[0])
        self.netC = DomainConfusion(3, 2, opt.bottleneck, 128, opt.audio_length // opt.pool_length).to(self.devices[0])
        self.netD_A = WaveNet(opt.mu, 30, 10, 64, 256, 256, opt.bottleneck, 1, 1).to(self.devices[-1])
        self.netD_B = WaveNet(opt.mu, 30, 10, 64, 256, 256, opt.bottleneck, 1, 1).to(self.devices[-1])

        if self.isTrain:
            self.A_target = torch.LongTensor([0]).to(self.devices[0])
            self.B_target = torch.LongTensor([1]).to(self.devices[0])
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
            self.optimizer_C = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netC.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_C, self.optimizer_D] 

    def set_input(self, input): 
        A, params_A = input[0]  
        B, params_B = input[1] 
        self.real_A = A.to(self.devices[0])
        self.real_B = B.to(self.devices[0]) 
        self.params_A = self.decollate_params(params_A)
        self.params_B = self.decollate_params(params_B)

    def get_indices(self, y):
        y = (y + 1.) * .5 * self.opt.mu
        return y.clamp(0, self.opt.mu).long()

    def inv_indices(self, y):
        return y.float() / (0.5  * self.opt.mu) - 1.
  
    def train(self):
        self.optimizer_C.zero_grad()
        self.optimizer_D.zero_grad()

        encoded_A = self.netE(self.real_A.unsqueeze(1))
        encoded_B = self.netE(self.real_B.unsqueeze(1)) 
 
        pred_C_A = self.netC(encoded_A)
        pred_C_B = self.netC(encoded_B)

        self.loss_C_A = self.criterion(pred_C_A, self.A_target)
        self.loss_C_B = self.criterion(pred_C_B, self.B_target)
        self.loss_C = self.loss_C_A + self.loss_C_B
        self.loss_C.backward(retain_graph=True)

        encoded_A = nn.functional.interpolate(encoded_A, scale_factor=self.opt.pool_length).to(self.devices[-1])
        encoded_B = nn.functional.interpolate(encoded_B, scale_factor=self.opt.pool_length).to(self.devices[-1])
          
        target_A = self.get_indices(self.real_A).to(self.devices[-1])
        target_B = self.get_indices(self.real_B).to(self.devices[-1])

        rec_A = self.netD_A((encoded_A, target_A)) 
        rec_B = self.netD_B((encoded_B, target_B))

        self.rec_A = self.inv_indices(rec_A)
        self.rec_B = self.inv_indices(rec_B)
 
        self.loss_D_A = self.criterion(rec_A, target_A).to(self.devices[0])
        self.loss_D_B = self.criterion(rec_B, target_B).to(self.devices[0])
        self.loss_D = self.loss_D_A + self.loss_D_B
        self.loss_D.backward()

        self.optimizer_C.step()
        self.optimizer_D.step()

    def test(self):
        with torch.no_grad():
            encoded_A = self.netE(self.real_A)
            encoded_B = self.netE(self.real_B) 

            encoded_A = self.upsample(encoded_A)
            encoded_B = self.upsample(encoded_B)

            self.fake_A = self.netD_A((encoded_B, self.to_onehot(self.real_A).to(self.devices[0])))
            self.fake_B = self.netD_B((encoded_A, self.to_onehot(self.real_B).to(self.devices[0])))
