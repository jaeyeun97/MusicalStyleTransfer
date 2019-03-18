from .base_dataset import BaseDataset
from abc import ABC


class PairDataset(BaseDataset, ABC):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt) 

    @staticmethod
    def modify_commandline_options(parser, is_train): 
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        return parser
