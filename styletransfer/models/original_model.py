import itertools
import torch
import torch.nn as nn
from adabound import AdaBound
from .base_model import BaseModel
from .networks.encoders.resnet1d import Resnet1dEncoder
from .networks.classifiers import Conv2dClassifier
from .networks.wavenet import WaveNet
from .networks.nv_wavenet import NVWaveNet, Impl

class OriginalModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        preprocess = 'mel,normalize,mulaw,stft'
        parser.set_defaults(preprocess=preprocess)
        parser.add_argument('--wavenet_layers', type=int, default=30, help='wavenet layers')
        parser.add_argument('--wavenet_blocks', type=int, default=15, help='wavenet layers')
        parser.add_argument('--bottleneck', type=int, default=65, help='channel bottleneck')
        parser.add_argument('--width', type=int, default=128, help='width')
        parser.add_argument('--dc_lambda', type=float, default=0.01, help='dc lambda') 
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['C_A_right', 'C_B_right', 'C_A_wrong', 'C_B_wrong', 'D_A', 'D_B']
        if opt.isTrain:
            self.output_names = [] # ['aug_A', 'aug_B', 'rec_A', 'rec_B']
        else:
            self.output_names = ['real_A', 'real_B', 'fake_B', 'fake_A']
        self.params_names = ['params_A', 'params_B'] * 2
        self.model_names = ['E', 'C', 'D_A', 'D_B']

        args = dict(opt.__dict__)
        args['input_nc'] = opt.tensor_size
        args['output_nc'] = opt.bottleneck
        self.netE = Resnet1dEncoder(**args).to(self.devices[0])
        args['input_nc'] = 1
        args['tensor_height'] = opt.bottleneck
        args['flatten'] = True
        self.netC = Conv2dClassifier(**args).to(self.devices[0])

        stride = 2 * ((opt.nfft // 8) - 1)
        window =  opt.nfft // opt.duration_ratio
        self.netD_A = WaveNet(opt.mu+1, opt.wavenet_layers, opt.wavenet_blocks, 
                              opt.width, 256, 256,
                              opt.bottleneck, window, stride).to(self.devices[-1]) # opt.pool_length, opt.pool_length
        self.netD_B = WaveNet(opt.mu+1, opt.wavenet_layers, opt.wavenet_blocks,
                              opt.width, 256, 256,
                              opt.bottleneck, window, stride).to(self.devices[-1]) # opt.pool_length, opt.pool_length
        self.softmax = nn.LogSoftmax(dim=1) # (1, 256, audio_len) -> pick 256
        
        if self.isTrain:
            self.A_target = torch.LongTensor([0] * opt.batch_size).to(self.devices[0])
            self.B_target = torch.LongTensor([1] * opt.batch_size).to(self.devices[0])
            self.criterionDC = nn.CrossEntropyLoss(reduction='mean')
            self.criterionDecode = nn.CrossEntropyLoss(reduction='mean')
            self.optimizer_C = AdaBound(self.netC.parameters(), lr=opt.lr, final_lr=0.1)
            self.optimizer_D = AdaBound(itertools.chain(self.netE.parameters(), self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, final_lr=0.1)
            self.optimizers = [self.optimizer_C, self.optimizer_D] 
        else:
            self.preprocesses = []
            # TODO change structure of test.py and setup() instead
            load_suffix = str(opt.load_iter) if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
            self.netC.eval()
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
        self.optimizer_C.zero_grad() 
        encoded_A = self.netE(self.aug_A) # Input range: (-1, 1) Output: R^64
        pred_C_A = self.netC(encoded_A)
        self.loss_C_A_right = self.criterionDC(pred_C_A, self.A_target)
        loss = self.opt.dc_lambda * self.loss_C_A_right
        loss.backward()

        encoded_B = self.netE(self.aug_B) 
        pred_C_B = self.netC(encoded_B)
        self.loss_C_B_right = self.criterionDC(pred_C_B, self.B_target)
        loss = self.opt.dc_lambda * self.loss_C_B_right
        loss.backward()
        self.optimizer_C.step()
  
        self.optimizer_D.zero_grad() 
        encoded_A = self.netE(self.aug_A) # Input range: (-1, 1) Output: R^64
        pred_C_A = self.netC(encoded_A) 
        self.loss_C_A_wrong = self.criterionDC(pred_C_A, self.A_target)
        real_A = self.get_indices(self.real_A).to(self.devices[-1])
        pred_D_A = self.netD_A((encoded_A, real_A))
        self.loss_D_A = self.criterionDecode(pred_D_A, real_A)
        loss = self.loss_D_A - self.opt.dc_lambda * self.loss_C_A_wrong
        loss.backward()
        
        encoded_B = self.netE(self.aug_B)
        pred_C_B = self.netC(encoded_B) 
        self.loss_C_B_wrong = self.criterionDC(pred_C_B, self.B_target)
        real_B = self.get_indices(self.real_B).to(self.devices[-1]) 
        pred_D_B = self.netD_B((encoded_B, real_B))
        self.loss_D_B = self.criterionDecode(pred_D_B, real_B)
        loss = self.loss_D_B - self.opt.dc_lambda * self.loss_C_B_wrong
        loss.backward()
        self.optimizer_D.step() 
  
    def test(self):  
        with torch.no_grad():   
            encoded_A = self.netE(self.aug_A)
            encoded_B = self.netE(self.aug_B)
            self.fake_B = self.infer_A.infer(self.netD_A.get_cond_input(encoded_B), Impl.AUTO)
            self.fake_A = self.infer_B.infer(self.netD_B.get_cond_input(encoded_A), Impl.AUTO)
            self.fake_B = self.inv_indices(self.fake_B)
            self.fake_A = self.inv_indices(self.fake_A)
