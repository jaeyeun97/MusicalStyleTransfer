"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import audioread
import torch.utils.data
from data.base_dataset import BaseDataset
from data.single_dataset import SingleDataset
from data.pair_dataset import PairDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_single_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    if issubclass(dataset_class, SingleDataset):
        return dataset_class.modify_commandline_options
    else:
        raise TypeError('Dataset not a Single Class')

def get_pair_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    if issubclass(dataset_class, PairDataset):
        return dataset_class.modify_commandline_options
    else:
        raise TypeError('Dataset not a Pair Class')


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    if opt.single:
        A_data_loader = DatasetLoader(opt, 'A')
        B_data_loader = DatasetLoader(opt, 'B')
        return A_data_loader, B_data_loader
    else:
        data_loader = DatasetLoader(opt, 'pair')
        return data_loader


class DatasetLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, prefix):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(getattr(opt, '{}_dataset'.format(prefix)))
        if prefix == 'pair':
            self.dataset = dataset_class(opt)
            self.max_size = self.opt.max_dataset_size
        else:
            self.dataset = dataset_class(opt, prefix) 
            self.max_size = getattr(opt, '{}_max_dataset_size'.format(prefix))
        print("dataset [%s] was created" % type(self.dataset).__name__)
            
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.max_size)

    def __iter__(self):
        """Return a batch of data"""
        i = 0
        it = iter(self.dataloader)
        while i * self.opt.batch_size < self.max_size:
            try:
                yield next(it)
                i += 1
            except ValueError as e:
                print(e)
                continue
            except audioread.NoBackendError:
                print('Error Loading Data')
                continue 
            except StopIteration:
                break
