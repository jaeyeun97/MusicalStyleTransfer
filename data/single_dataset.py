from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from abc import ABC, abstractmethod


class SingleDataset(BaseDataset, ABC):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt, prefix):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt) 
        self.prefix = prefix
        self.root = self.get_opt('dataroot')

    @staticmethod
    def modify_commandline_options(parser, prefix, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            prefix          -- prefix
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        add_argument = SingleDataset.get_argument_adder(parser, prefix)
        add_argument('dataroot', type=str, required=True, help='Path to Dataset')
        add_argument('max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        return parser

    def get_opt(self, attr):
        return getattr(self.opt, '{}_{}'.format(self.prefix, attr))

    @staticmethod
    def get_argument_adder(parser, prefix):
        def closure(*args, **kwargs):
            name = args[0]
            if name.startwsith('--'):
                name = name[2:] 
            return parser.add_argument('--{}_{}'.format(prefix, name), *args[1:], **kwargs)
        return closure

    @staticmethod
    def get_default_setter(parser, prefix):
        def closure(**kwargs):
            defaults = {'{}_{}'.format(prefix, k): v for k, v in kwargs.items}
            return parser.set_defaults(**defaults)
        return closure
