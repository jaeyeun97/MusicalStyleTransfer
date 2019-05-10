import itertools
import torch
import torch.nn as nn
from adabound import AdaBound
from .base_model import BaseModel
from .networks.encoders.resnet1d import Resnet1dEncoder
from .networks.classifiers import Conv2dClassifier
from .networks.wavenet import WaveNet
from .networks.nv_wavenet import NVWaveNet, Impl

class CQTModel(BaseModel):
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
        preprocess = 'mulaw,cqt'
        parser.set_defaults(preprocess=preprocess)
        parser.add_argument('--wavenet_layers', type=int, default=30, help='wavenet layers')
        parser.add_argument('--wavenet_blocks', type=int, default=15, help='wavenet layers')
        parser.add_argument('--width', type=int, default=128, help='width')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['D_A', 'D_B']
        if opt.isTrain:
            self.output_names = [] # ['aug_A', 'aug_B', 'rec_A', 'rec_B']
        else:
            self.output_names = ['real_A', 'real_B', 'fake_B', 'fake_A']
            self.params_names = ['params_A', 'params_B'] * 2
        self.model_names = ['D_A', 'D_B'] 
 
        self.netD_A = WaveNet(opt.mu+1, opt.wavenet_layers, opt.wavenet_blocks, 
                              opt.width, 256, 256,
                              84, 512, 512).to(self.devices[-1]) 
        self.netD_B = WaveNet(opt.mu+1, opt.wavenet_layers, opt.wavenet_blocks,
                              opt.width, 256, 256,
                              84, 512, 512).to(self.devices[-1])
        self.softmax = nn.LogSoftmax(dim=1) # (1, 256, audio_len) -> pick 256
        
        if self.isTrain:
            self.criterionDecode = nn.CrossEntropyLoss(reduction='mean')
            self.optimizer_D_A = AdaBound(self.netD_A.parameters(), lr=opt.lr, final_lr=0.1)
            self.optimizer_D_B = AdaBound(self.netD_B.parameters(), lr=opt.lr, final_lr=0.1)
            self.optimizers = [self.optimizer_D_A, self.optimizer_D_B] 
        else:
            self.preprocesses = []
            load_suffix = str(opt.load_iter) if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
            self.netD_A.eval()
            self.netD_B.eval()
             
            self.infer_A = NVWaveNet(**(self.netD_A.export_weights()))
            self.infer_B = NVWaveNet(**(self.netD_B.export_weights()))

    def set_input(self, input): 
        A, params_A = input[0]  
        B, params_B = input[1] 
         
        self.real_A = params_A['original'].to(self.devices[0])
        self.real_B = params_B['original'].to(self.devices[0])
        self.aug_A = A.to(self.devices[0])
        self.aug_B = B.to(self.devices[0])

        self.params_A = self.decollate_params(params_A)
        self.params_B = self.decollate_params(params_B)

    def get_indices(self, y):
        y = (y + 1.) * .5 * self.opt.mu
        return y.long() 

    def inv_indices(self, y):
        return y.float() / self.opt.mu * 2. - 1.
 
    def train(self): 
        self.optimizer_D_A.zero_grad() 
        real_A = self.get_indices(self.real_A).to(self.devices[-1])
        pred_D_A = self.netD_A((self.aug_A, real_A))
        self.loss_D_A = self.criterionDecode(pred_D_A, real_A)
        self.loss_D_A.backward()
        self.optimizer_D_A.step() 

        self.optimizer_D_B.zero_grad() 
        real_B = self.get_indices(self.real_B).to(self.devices[-1]) 
        pred_D_B = self.netD_B((self.aug_B, real_B))
        self.loss_D_B = self.criterionDecode(pred_D_B, real_B)
        self.loss_D_B.backward()
        self.optimizer_D_B.step() 
  
    def test(self):  
        with torch.no_grad():   
            self.fake_B = self.infer_A.infer(self.netD_A.get_cond_input(self.aug_A), Impl.AUTO)
            self.fake_A = self.infer_B.infer(self.netD_B.get_cond_input(self.aug_B), Impl.AUTO)
            self.fake_B = self.inv_indices(self.fake_B)
            self.fake_A = self.inv_indices(self.fake_A)
