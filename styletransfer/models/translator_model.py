import itertools
import torch
import torch.nn as nn
from adabound import AdaBound
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
        parser.add_argument('--bottleneck', type=int, default=64, help='bottleneck')
        parser.add_argument('--dc_width', type=int, default=128, help='width')
        parser.add_argument('--width', type=int, default=128, help='width')
        parser.add_argument('--pool_length', type=int, default=128, help='pool length')
        
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
            'width': opt.width,
            'bottleneck_width': opt.bottleneck,
            'pool_length': opt.pool_length, 
        }).to(self.devices[0])
        self.vector_length = opt.audio_length // opt.pool_length
        self.netC = DomainConfusion(3, 2, opt.bottleneck, opt.dc_width, self.vector_length).to(self.devices[0])
        self.netD_A = WaveNet(opt.mu+1, 40, 10, 
                              opt.width, 256, 256,
                              opt.bottleneck, 1, 1).to(self.devices[-1]) # opt.pool_length, opt.pool_length
        self.netD_B = WaveNet(opt.mu+1, 40, 10,
                              opt.width, 256, 256,
                              opt.bottleneck, 1, 1).to(self.devices[-1]) # opt.pool_length, opt.pool_length
        self.softmax = nn.LogSoftmax(dim=1) # (1, 256, audio_len) -> pick 256

        self.prev_A = None
        self.prev_B = None

        self.pred_C_A = None
        self.pred_C_B = None
        # self.test_D_A = self.netD_A.embed.weight
        # self.test_D_B = self.netD_B.embed.weight
        # self.i = 0
        
        if self.isTrain:
            self.A_target = torch.LongTensor([0]).to(self.devices[0])
            self.B_target = torch.LongTensor([1]).to(self.devices[0])
            self.criterionDC = nn.CrossEntropyLoss(reduction='mean')
            self.criterionDecode = nn.NLLLoss(reduction='mean')
            # self.optimizer = torch.optim.Adam(itertools.chain(self.netE.parameters(), self.netC.parameters(), self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer = AdaBound(itertools.chain(self.netE.parameters(), self.netC.parameters(), self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, final_lr=0.1)
            self.optimizers = [self.optimizer] 

    def set_input(self, input): 
        A, params_A = input[0]  
        B, params_B = input[1] 
        self.real_A = A.to(self.devices[0])
        self.real_B = B.to(self.devices[0]) 
        self.params_A = self.decollate_params(params_A)
        self.params_B = self.decollate_params(params_B)

    def get_indices(self, y):
        y = (y + 1.) * .5 * (self.opt.mu + 1)
        return y.clamp(0, self.opt.mu).long()

 
    def to_onehot(self, y, device):
        y = self.get_indices(y).view(-1, 1)
        y = torch.zeros(y.size()[0], self.opt.mu + 1).to(device).scatter_(1, y, 1)
        return y.transpose(0, 1).unsqueeze(0)

    def inv_indices(self, y):
        return y.float() / (self.opt.mu + 1) * 2. - 1.

    @staticmethod
    def sample(logits):
        dist = torch.distributions.categorical.Categorical(logits=logits.transpose(1, 2))
        return dist.sample()
 
    def train(self):
        if self.prev_A is None:
            self.prev_A = self.real_A
        if self.prev_B is None:
            self.prev_B = self.real_B 

        self.optimizer.zero_grad()

        encoded_A = self.netE(self.real_A.unsqueeze(1)) # Input range: (-1, 1) Output: R^64
        encoded_B = self.netE(self.real_B.unsqueeze(1)) 
        pred_C_A = self.netC(encoded_A) # (encoded_A + 1.) * self.vector_length / 2)
        pred_C_B = self.netC(encoded_B) # (encoded_B + 1.) * self.vector_length / 2)
        self.loss_C_A = self.criterionDC(pred_C_A, self.A_target)
        self.loss_C_B = self.criterionDC(pred_C_B, self.B_target)
        loss_C = self.loss_C_A + self.loss_C_B
        loss_C.backward(retain_graph=True)

        encoded_A = nn.functional.interpolate(encoded_A, size=self.opt.audio_length).to(self.devices[-1])
        encoded_B = nn.functional.interpolate(encoded_B, size=self.opt.audio_length).to(self.devices[-1])
        # encoded_A = encoded_A.to(self.devices[-1])
        # encoded_B = encoded_B.to(self.devices[-1])

        self.prev_A = self.to_onehot(self.prev_A, self.devices[-1])
        self.prev_B = self.to_onehot(self.prev_B, self.devices[-1])
        # self.prev_A = self.prev_A.to(self.devices[-1]).unsqueeze(1)
        # self.prev_B = self.prev_B.to(self.devices[-1]).unsqueeze(1)

        pred_D_A = self.netD_A((encoded_A, self.prev_A))
        rec_A = self.softmax(pred_D_A) 
        pred_D_B = self.netD_B((encoded_B, self.prev_B))
        rec_B = self.softmax(pred_D_B)   

        self.loss_D_A = self.criterionDecode(rec_A, self.get_indices(self.real_A).to(self.devices[-1]))
        self.loss_D_B = self.criterionDecode(rec_B, self.get_indices(self.real_B).to(self.devices[-1]))
        loss_D = self.loss_D_A + self.loss_D_B
        loss_D.backward()

        self.optimizer.step()
 
        self.prev_A = self.real_A
        self.prev_B = self.real_B

        self.rec_A = self.inv_indices(self.sample(rec_A))
        self.rec_B = self.inv_indices(self.sample(rec_B))
        
        # DEBUG
        if self.pred_C_A is None:
            self.pred_C_A = pred_C_A 
        if self.pred_C_B is None:
            self.pred_C_B = pred_C_B 

        if (self.pred_C_A - pred_C_A).max() == 0.:
            print(pred_C_A)
        if (self.pred_C_B - pred_C_B).max() == 0.:
            print(pred_C_B)

        # if (self.test_D_A - self.netD_A.embed.weight).sum() == 0.:
        #     print('iter {}: Weight for D_A not changed'.format(self.i))
        # if (self.test_D_B - self.netD_B.embed.weight).sum() == 0.:
        #     print('iter {}: Weight for D_B not changed'.format(self.i))
        # self.i += 1

        # self.test_D_A = self.netD_A.embed.weight
        # self.test_D_B = self.netD_B.embed.weight

        self.pred_C_A = pred_C_A
        self.pred_C_B = pred_C_B
          
  
    def test(self):
        with torch.no_grad():
            encoded_A = self.netE(self.real_A.unsqueeze(1))
            encoded_B = self.netE(self.real_B.unsqueeze(1))

            encoded_A = nn.functional.interpolate(encoded_A, scale_factor=self.opt.pool_length).to(self.devices[-1])
            encoded_B = nn.functional.interpolate(encoded_B, scale_factor=self.opt.pool_length).to(self.devices[-1])
            real_A = self.to_onehot(self.real_A, self.devices[-1])
            real_B = self.to_onehot(self.real_B, self.devices[-1])

            fake_A = self.softmax(self.netD_A((encoded_B, real_A)))
            fake_B = self.softmax(self.netD_B((encoded_A, real_B)))

            self.fake_A = self.inv_indices(self.sample(fake_A))
            self.fake_B = self.inv_indices(self.sample(fake_B)) 