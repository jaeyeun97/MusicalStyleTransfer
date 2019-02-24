"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch
import librosa

import torch.utils.data as data
from abc import ABC, abstractmethod
from util import (calc, stft, hz_to_mel,
                  normalize_magnitude,
                  normalize_phase,
                  combine_mag_phase)


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, audio_length):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.nfft = opt.nfft
        self.sr_dur_ratio = opt.sr_to_dur_ratio
        self.preprocess = opt.preprocess.split(',')

        self.tensor_size = self.nfft // 2 + 1
        self.hop_length = self.nfft // 4
        self.audio_length = (self.tensor_size - 1) * self.hop_length
        self.duration = audio_length * (1 + ((self.nfft - 2048) / self.sr_dur_ratio))
        self.sample_rate = int(self.audio_length / self.duration) + 1

        print("sample rate: {}".format(self.sample_rate))

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    def trim_dataset(self, paths):
        return paths[:min(self.opt.max_dataset_size, len(paths))]

    def get_data(self, path):
        y, sr = librosa.load(path, sr=self.sample_rate, duration=self.duration)
        if len(y) < self.audio_length:
            y = librosa.util.fix_length(y, self.audio_length)
        else:
            y = y[:self.audio_length]
        # Preprocess
        if 'mel' in self.preprocess:
            y = hz_to_mel(y)
        # STFT
        D = stft(y, n_fft=self.nfft)
        # Compute Magnitude and phase
        lmag, agl = calc(D, self.opt.smoothing_factor)
        # Normalize
        mmax = mmin = 0
        if 'normalize' in self.preprocess:
            lmag, mmax, mmin = normalize_magnitude(lmag)
            agl = normalize_phase(agl)

        return mmax, mmin, combine_mag_phase(lmag, agl)
