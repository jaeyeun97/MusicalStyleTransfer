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

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.nfft = opt.nfft
        self.preprocess = opt.preprocess.split(',')
        self.sample_rate = opt.sample_rate

        self.tensor_size = opt.tensor_size
        self.hop_length = opt.hop_length 
        self.audio_length = opt.audio_length 
        self.duration = opt.duration

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

    def retrieve_audio(self, path, split_num):
        y, sr = librosa.load(path,
                             sr=self.sample_rate,
                             offset=split_num*self.duration,
                             duration=self.duration)
        if len(y) < self.audio_length:
            y = librosa.util.fix_length(y, self.audio_length)
        else:
            y = y[:self.audio_length]
        return y

    def transform(self, y):
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
