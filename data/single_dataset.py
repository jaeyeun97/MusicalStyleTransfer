from data.base_dataset import BaseDataset
from abc import ABC


class SingleDataset(BaseDataset, ABC):
    def __init__(self, opt, prefix):
        BaseDataset.__init__(self, opt) 
        self.prefix = prefix
        self.root = self.get_opt('dataroot')

    @staticmethod
    def modify_commandline_options(parser, prefix, is_train):
        add_argument = SingleDataset.get_argument_adder(parser, prefix)
        add_argument('dataroot', type=str, help='Path to Dataset')
        add_argument('max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        return parser

    def get_opt(self, attr):
        return getattr(self.opt, '{}_{}'.format(self.prefix, attr))

    @staticmethod
    def get_argument_adder(parser, prefix):
        def closure(*args, **kwargs):
            name = args[0]
            if name.startswith('--'):
                name = name[2:] 
            return parser.add_argument('--{}_{}'.format(prefix, name), *args[1:], **kwargs)
        return closure

    @staticmethod
    def get_default_setter(parser, prefix):
        def closure(**kwargs):
            defaults = {'{}_{}'.format(prefix, k): v for k, v in kwargs.items()}
            return parser.set_defaults(**defaults)
        return closure
