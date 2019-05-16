import argparse
import os
import math
import torch
from ..util.util import mkdirs
from ..models import get_option_setter
from ..data import get_single_option_setter, get_pair_option_setter


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--reader', type=str, default='librosa', help='audio reader')
        parser.add_argument('--duration', type=float, default=8, help='duration of audio')
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--discriminator', type=str, default='conv1d', help='architecture of discriminator')
        parser.add_argument('--transformer', type=str, default='none', help='specify generator transformer architecture')
        parser.add_argument('--encoder', type=str, default='conv1d', help='specify generator autoencoder architecture')
        parser.add_argument('--conv_size', type=int, default=3, help='conv filter size')
        parser.add_argument('--conv_pad', type=int, default=1, help='conv padding size')

        parser.add_argument('--ngf', type=int, default=4, help='# of generator filters')
        parser.add_argument('--mgf', type=float, default=0.5, help='generator filter number multiplier')
        parser.add_argument('--ndf', type=int, default=2, help='# of discriminator filters')
        parser.add_argument('--mdf', type=int, default=2, help='generator filter number multiplier')
        parser.add_argument('--n_layers', type=int, default=3, help='# of discriminator conv layers')

        parser.add_argument('--num_trans_layers', type=int, default=9, help='# of trans layer')

        # Conv
        parser.add_argument('--n_downsample', type=int, default=2, help='Used for ConvAutoencoder: number of downsampling layers')
        parser.add_argument('--shrinking_filter', action='store_true', help='Used for ConvAutoencoder: halving/doubling filter size')

        # Conv Classifier
        parser.add_argument('--disc_layers', type=int, default=5, help='layers for conv classifier')

        # MU
        parser.add_argument('--mu', type=int, default=255, help='mu')

        # Initialization
        parser.add_argument('--norm_layer', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        # dataset parameters
        parser.add_argument('--A_dataset', type=str, default='maestro')
        parser.add_argument('--B_dataset', type=str, default='guitarset')
        parser.add_argument('--pair_dataset', type=str)
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

        # Audio Args
        parser.add_argument('--preprocess', type=str, default='normalize,mel,shift', help='scaling and cropping of images at load time [mel|normalize]')
        parser.add_argument('--sample_rate', type=int, default=16384, help='Sample Rate to resample')
        parser.add_argument('--nfft', type=int, default=2048, help='Number of Frequency bins for STFT')
        parser.add_argument('--cqt_octave_bins', type=int, default=120, help='Number of Frequency bins for STFT')
        parser.add_argument('--cqt_n_bins', type=int, default=88, help='Number of Frequency bins for STFT')
        parser.add_argument('--smoothing_factor', type=float, default=1, help='Smoothing for Log-Magnitude')

        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--phase', type=str, default='gan', help='gan, train, test')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults
        parser.set_defaults(preprocess=opt.preprocess.lstrip(','))

        # modify dataset-related parser options
        if opt.A_dataset is not None:
            if opt.B_dataset is None:
                raise ValueError('B_dataset not set')
            if opt.pair_dataset is not None:
                raise ValueError('Pair Dataset cannot be set with a Single Dataset')  
            # Single Dataset
            parser.set_defaults(single=True)
            A_option_setter = get_single_option_setter(opt.A_dataset)
            parser = A_option_setter(parser, 'A', self.isTrain)
            B_option_setter = get_single_option_setter(opt.B_dataset)    
            parser = B_option_setter(parser, 'B', self.isTrain)
        elif opt.B_dataset is not None:
            raise ValueError('A_dataset not set')
        elif opt.pair_dataset is None:
            raise ValueError('Set Single or Pair datasets')
        else:
            # Pair Dataset
            parser.set_defaults(single=False)
            pair_option_setter = get_pair_option_setter(opt.pair_dataset)
            parser = pair_option_setter(parser, self.isTrain)

        # set lengths
        if 'stft' in opt.preprocess:
            tensor_size = opt.nfft // 2 + 1
            hop_length = opt.nfft // 4
            audio_length = int(opt.duration * opt.sample_rate)
            square_length = ((tensor_size - 1) * hop_length)
            duration = square_length / opt.sample_rate

            print('Audio must be a divisor of: {}'.format(duration))
            print('Current audio length: {}'.format(opt.duration))
            assert square_length % audio_length == 0 

            ratio = square_length // audio_length
            parser.set_defaults(
                tensor_height=tensor_size,
                tensor_width=tensor_size // ratio,
                hop_length=hop_length,
                audio_length=audio_length,
                duration_ratio=ratio)

        elif 'cqt' in opt.preprocess:
            tensor_size = opt.cqt_n_bins * opt.cqt_octave_bins // 12 
            audio_length = int(opt.duration * opt.sample_rate)
            parser.set_defaults(
                tensor_height=tensor_size,
                hop_length=512,
                audio_length=audio_length,
                tensor_width=int(audio_length / 512) + 1,
            )
        else:
            parser.set_defaults(audio_length=int(opt.sample_rate * opt.duration))

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt) 

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
