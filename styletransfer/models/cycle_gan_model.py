import torch
import itertools
from ..util.audio_pool import AudioPool
from .generator import getGenerator
from .discriminator import getDiscriminator
from .base_model import BaseModel
from .loss import GANLoss, cal_gradient_penalty


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
        opt, _ = parser.parse_known_args()
        parser.set_defaults(preprocess='stft,'+opt.preprocess, no_dropout=True, phase='gan') #, gan_mode='wgangp')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'idt_A', 'cycle_A', 'D_B', 'G_B', 'idt_B', 'cycle_B', 'gp_A', 'gp_B']

        output_names_A = ['real_A', 'fake_B', 'rec_A']
        params_names_A = ['params_A'] * 3
        output_names_B = ['real_B', 'fake_A', 'rec_B'] 
        params_names_B = ['params_B'] * 3

        if self.isTrain:
            self.lambda_identity = opt.lambda_identity
            if self.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
                output_names_A.append('idt_B')
                params_names_A += ['params_A']
                output_names_B.append('idt_A')
                params_names_B += ['params_B']

        self.output_names = output_names_A + output_names_B  # combine visualizations for A and B
        self.params_names = params_names_A + params_names_B

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
            self.netD_B = getDiscriminator(opt, self.devices[0])
             
            self.criterionD_A = GANLoss(opt.gan_mode).to(self.devices[-1]) 
            self.criterionD_B = GANLoss(opt.gan_mode).to(self.devices[0])

            # self.fake_A_pool = AudioPool(opt.audio_pool_size) # create image buffer
            # self.fake_B_pool = AudioPool(opt.audio_pool_size) # create image buffer
            # define loss functions
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
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
        A, params_A = first
        B, params_B = second
        self.params_A = self.decollate_params(params_A)
        self.params_B = self.decollate_params(params_B)

        self.real_A = A.to(self.devices[0]) # tuple(A.to(device=dev) for dev in self.devices)
        self.real_B = B.to(self.devices[-1]) # tuple(B.to(device=dev) for dev in self.devices) 


    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        if self.devices[0] != self.devices[-1]:
            self.fake_B = self.fake_B.to(self.devices[-1])
            self.fake_A = self.fake_A.to(self.devices[0])

        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A)), device[-1]
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B)), device[0]
 

    def test(self):
        with torch.no_grad():
            self.forward()  

    def train(self):  
        self.forward()

        if self.devices[0] != self.devices[-1]:
            self.fake_B = self.fake_B.to(self.devices[0])
            self.fake_A = self.fake_A.to(self.devices[-1])
            self.real_A = self.real_A.to(self.devices[-1])
            self.real_B = self.real_B.to(self.devices[0])

        # Train G  
        self.set_requires_grad([self.netD_A, self.netD_B], False) 
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero  

        if self.lambda_identity > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_B = self.netG_A(self.real_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) \
                              * self.opt.lambda_B * self.lambda_identity
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_A = self.netG_B(self.real_A)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A) \
                              * self.opt.lambda_A * self.lambda_identity 
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_B = self.criterionD_A(self.netD_A(self.fake_A), True) # D_A(G_B(B))
        self.loss_G_A = self.criterionD_B(self.netD_B(self.fake_B), True) # D_B(G_A(A)) 

        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) \
                            * self.opt.lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) \
                            * self.opt.lambda_B

        if self.devices[0] != self.devices[-1]:
            self.loss_G_B = self.loss_G_B.to(self.devices[0])
            self.loss_cycle_A = self.loss_cycle_A.to(self.devices[0])
            self.loss_idt_A = self.loss_idt_A.to(self.devices[0])

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
        self.optimizer_G.step()
        # G Train done

        # Train Disc.
        self.set_requires_grad([self.netD_A, self.netD_B], True) 
        self.optimizer_D.zero_grad() 
        pred_B_B_real = self.netD_B(self.real_B) # D_B(B)
        fake_B = self.fake_B.detach()
        # pred_B_B_fake = self.netD_B(self.fake_B_pool.query(fake_B))
        pred_B_B_fake = self.netD_B(fake_B)
        self.loss_gp_B, _ = cal_gradient_penalty(self.netD_B, self.real_B, fake_B, self.devices[0])
        self.loss_D_B = (self.criterionD_B(pred_B_B_real, True) + 
                         self.criterionD_B(pred_B_B_fake, False) +
                         self.loss_gp_B) * 0.5 
        self.loss_D_B.backward()

        pred_A_A_real = self.netD_A(self.real_A)
        fake_A = self.fake_A.detach()
        # pred_A_A_fake = self.netD_A(self.fake_A_pool.query(fake_A))
        pred_A_A_fake = self.netD_A(fake_A)
        self.loss_gp_A, _ = cal_gradient_penalty(self.netD_A, self.real_A, fake_A, self.devices[-1]) 
        self.loss_D_A = (self.criterionD_A(pred_A_A_real, True) +
                         self.criterionD_A(pred_A_A_fake, False) +
                         self.loss_gp_A) * 0.5

        self.loss_D_A.backward()
        self.optimizer_D.step() 
        # D training done 
 
        self.lambda_identity *= 0.999

