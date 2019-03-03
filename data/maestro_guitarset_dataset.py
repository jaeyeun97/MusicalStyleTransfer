"""
Maestro + GuitarSet Dataset class

For training a CycleGAN Network.
"""
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset
from util import mkdir

import csv
import os
import librosa
import torch
import random
import pandas
import numpy as np
import glob


class MaestroGuitarsetDataset(BaseDataset):
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
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--maestro_dir', type=str, default='maestro-v1.0.0', help='path to maestro')
        parser.add_argument('--guitarset_dir', type=str, default='GuitarSet', help='path to GuitarSet')
        parser.set_defaults(max_dataset_size=4000, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        maestro_path = os.path.abspath(os.path.join(self.root, opt.maestro_dir))
        maestro_name = os.path.basename(maestro_path)
        maestro_file = os.path.join(maestro_path, "{}.csv".format(maestro_name))
        maestro_meta = pandas.read_csv(maestro_file)

        splits = (maestro_meta.duration // self.duration).rename('splits')
        maestro_meta = maestro_meta.join(splits)[['audio_filename', 'splits']]
        self.A_paths = list()
        for row in maestro_meta.iterrows():
            for split in range(row.spilts):
                self.A_paths.append('{}:{}'.format(row.audio_filename, str(split)))
        self.A_size = len(self.A_paths)

        guitarset_path = os.path.abspath(os.path.join(self.root, opt.guitarset_dir))
        guitarset_paths = glob.glob(os.path.join(guitarset_path, 'audio/audio_mic/*.wav')) 
        self.B_paths = list()
        for f in guitarset_paths:
            num_split = int(librosa.get_duration(filename=f) // self.duration)
            for split in range(num_split):
                self.B_paths.append('{}:{}'.format(f, str(split)))
        self.B_size = len(self.B_paths)

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

        A_audio, A_split = tuple(A_path.split(':'))
        B_audio, B_split = tuple(B_path.split(':'))

        A = self.retrieve_audio(A_audio, A_split)
        B = self.retrieve_audio(B_audio, B_split)

        a_max, a_min, A = self.transform(A)
        b_max, b_min, B = self.transform(B)

        return {
            'A': A,
            'B': B,
            'A_path': A_audio,
            'B_path': B_audio,
            'A_split': A_split,
            'B_split': B_split,
            'A_max': a_max,
            'A_min': a_min,
            'B_max': b_max,
            'B_min': b_min
        }

    def __len__(self):
        """Return the total number of audio files."""
        return max(len(self.A_paths), len(self.B_paths))
