
"""FMA Dataset class

For training a CycleGAN Network.
Using FMA_large dataset, which is already trimmed to 30 seconds.
"""
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset
from util import mkdir

import csv
import os
import random


class UnalignedDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
        parser          -- original option parser
        is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
        the modified parser.
        """
        parser.add_argument('--sample_rate', type=int, default=22050, help='Sample Rate to resample')
        parser.add_argument('--nfft', type=int, default=2048, help='Number of Frequency bins for STFT')
        parser.add_argument('--mel', type=bool, default=False, help='Use the mel scale')
        parser.add_argument('--A_subdir', type=str, default='A', help='subdir that contains audio for set A')
        parser.add_argument('--B_subdir', type=str, default='B', help='subdir that contains audio for set B')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the audio path
        # TODO: search genre and maintain a list

        FMA

        # self.A_dir = os.path.isdir(os.path.join(self.root, opt.A_subdir))
        # self.B_dir = os.path.isdir(os.path.join(self.root, opt.B_subdir))

        self.A_paths = sorted(make_dataset(self.A_dir, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.B_dir, opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.sample_rate = opt.sample_rate
        self.nfft = opt.nfft
        self.mel = opt.mel

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
        index -- a random integer for data indexing
        """
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_audio = self.retrieve_audio(A_path)
        B_audio = self.retrieve_audio(B_path)

        A = self.transform(A_audio)
        B = self.transform(B_audio)

        return {'A': A, 'B': B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.frame_paths)
