import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from .util.scheduler import get_scheduler
from ..util.audio import (denormalize_magnitude, istft, icqt, inv_mulaw, pitch_deshift, decalc, mel_to_hz)


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.output_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.devices = [torch.device('cuda:{}'.format(gpu_id)) for gpu_id in self.gpu_ids] if self.gpu_ids else [torch.device('cpu')]
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        self.preprocesses = opt.preprocess.split(',')

        self.loss_names = []
        self.model_names = []
        self.output_names = []
        self.params_names = []
        self.optimizers = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.mmax = self.mmin = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass 
 
    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = str(opt.load_iter) if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()


    @abstractmethod
    def train(self):
        """train"""
        pass

    @abstractmethod
    def test(self):
        """Forward function used in test time. 
        use with torch.no_grad() to stop gradients
        """
        pass
        
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step(self.metric)
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_audio(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        audio_ret = OrderedDict()
        for i in range(len(self.output_names)):
            output_name = self.output_names[i]
            params_name = self.params_names[i]
            if isinstance(output_name, str):
                output = getattr(self, output_name)
                params = getattr(self, params_name)
                if type(output) == tuple:
                    output = output[0]
                output = self.postprocess(output, params)
                audio_ret[output_name] = output
        return audio_ret

    @staticmethod
    def decollator(k, v):
        if k == 'labels':
            return v
        if isinstance(v, torch.Tensor):
            v.requires_grad = False
            return v.numpy()
        else:
            return v
    
    def decollate_params(self, params):
        return {k: self.decollator(k, v) for k, v in params.items()}
             
    def postprocess(self, y, params):
        y = y.detach().cpu().numpy() 
        if 'stft' in self.preprocesses:
            if 'normalize' in self.preprocesses:
                if 'max' not in params or 'min' not in params:
                    raise Exception('No max or min')
                y = denormalize_magnitude(params['max'], params['min'], y)
            y = decalc(y, params['phase'], self.opt.smoothing_factor)
            y = istft(y)
        if 'cqt' in self.preprocesses:
            if 'normalize' in self.preprocesses:
                if 'max' not in params or 'min' not in params:
                    raise Exception('No max or min')
                y = denormalize_magnitude(params['max'], params['min'], y)
            y = decalc(y, params['phase'], self.opt.smoothing_factor)
            y = icqt(y, sr=self.opt.sample_rate, bins_per_octave=self.opt.cqt_octave_bins)
        if 'mulaw' in self.preprocesses:
            y = inv_mulaw(y, self.opt.mu)
        if 'mel' in self.preprocesses:
            y = mel_to_hz(y)
        if 'shift' in self.preprocesses:
            y = pitch_deshift(y, self.opt.sample_rate, params['start'], params['end'], params['shift'])
        return y

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    device = net.device
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(device)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=net.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
