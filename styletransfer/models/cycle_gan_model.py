import torch
import itertools
from ..util.audio_pool import AudioPool
from .generator import getGenerator
from .discriminator import getDiscriminator
from .base_model import BaseModel
from .loss import GANLoss


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_B: G_A(A) vs. B; D_A: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']

        output_names_A = ['real_A', 'fake_B', 'rec_A']
        output_names_B = ['real_B', 'fake_A', 'rec_B'] 
        self.output_names = output_names_A + output_names_B  # combine visualizations for A and B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = getGenerator(self.devices[0], opt)
        self.netG_B = getGenerator(self.devices[-1], opt)

        if self.isTrain:  # define discriminators
            self.netD_A = getDiscriminator(opt, self.devices[-1])
            self.criterionD_A = GANLoss(opt.gan_mode).to(self.devices[-1]) 
            self.netD_B = getDiscriminator(opt, self.devices[0])
            self.criterionD_B = GANLoss(opt.gan_mode).to(self.devices[0])

            # if self.opt.use_audio_pool:
            #     self.fake_A_pool = AudioPool(opt.audio_pool_size) # create image buffer
            #     self.fake_B_pool = AudioPool(opt.audio_pool_size) # create image buffer
            # define loss functions
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        first = input[0 if AtoB else 1]
        second = input[1 if AtoB else 0]
        self.mmax, self.mmin, A = self.preprocess(first)
        _, _, B = self.preprocess(second)

        self.real_A = tuple(A.to(device=dev) for dev in self.devices)
        self.real_B = tuple(B.to(device=dev) for dev in self.devices) 


    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG_A(self.real_A[0])  # G_A(A)
            self.fake_A = self.netG_B(self.real_B[-1])  # G_B(B)

            if self.devices[0] != self.devices[-1]:
                self.fake_B = self.fake_B.to(self.devices[-1])
                self.fake_A = self.fake_A.to(self.devices[0])
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A)), device[-1]
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B)), device[0] 
   

    def train(self):  
        # Generate Fakes
        self.fake_B = self.netG_A(self.real_A[0])  # G_A(A)
        self.fake_A = self.netG_B(self.real_B[-1])  # G_B(B)

         
        # Train Disc.
        self.set_requires_grad([self.netD_A, self.netD_B], True) 
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        pred_B_A_real = self.netD_B(self.real_A[0])
        pred_B_B_real = self.netD_B(self.real_B[0])
        pred_B_B_fake = self.netD_B(self.fake_B.detach())

        pred_A_B_real = self.netD_A(self.real_B[-1])
        pred_A_A_real = self.netD_A(self.real_A[-1])
        pred_A_A_fake = self.netD_A(self.fake_A.detach())

        self.loss_D_B = (self.criterionD_B(pred_B_A_real, False)
                       + self.criterionD_B(pred_B_B_real, True)
                       + self.criterionD_B(pred_B_B_fake, False)) # / 3
        self.loss_D_B.backward()

        self.loss_D_A = (self.criterionD_A(pred_A_B_real, False)
                        + self.criterionD_A(pred_A_A_real, True)
                        + self.criterionD_A(pred_A_A_fake, False)) # / 3
        self.loss_D_A.backward()
        self.optimizer_D.step() 
        # D training done 

        # Train G  
        self.set_requires_grad([self.netD_A, self.netD_B], False) 
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero 
        self.loss_G_B = self.criterionD_B(self.netD_B(self.fake_B), True)
        self.loss_G_A = self.criterionD_A(self.netD_A(self.fake_A), True)
        if self.devices[0] != self.devices[-1]:
            self.fake_B = self.fake_B.to(self.devices[-1])
            self.fake_A = self.fake_A.to(self.devices[0])
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A)), device[-1]
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B)), device[0] 
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A[-1]) * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B[0]) * self.opt.lambda_B
        self.loss_G_A = self.loss_G_A.to(self.devices[0])
        self.loss_cycle_A = self.loss_cycle_A.to(self.devices[0])
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()
        self.optimizer_G.step()
        # G Train done

