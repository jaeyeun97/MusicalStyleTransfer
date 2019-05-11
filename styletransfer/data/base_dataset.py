"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch
import librosa
import audioread
import soundfile as sf

import torch.utils.data as data
from abc import ABC, abstractmethod
from ..util.audio import (calc, stft, hz_to_mel,
                          normalize_magnitude,
                          normalize_phase,
                          combine_mag_phase,
                          pitch_shift,
                          mulaw)


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
        self.preprocesses = opt.preprocess.split(',')
        self.sample_rate = opt.sample_rate

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
        count = 0
        while True:
            try: 
                y, _ = librosa.load(path,
                                    sr=self.sample_rate,
                                    offset=split_num*self.duration,
                                    duration=self.duration)
                break
            except RuntimeError as e:
                print(e)
                if count > 5:
                    raise ValueError('Some random Runtime Error')
                else:
                    continue 
        if len(y) < self.audio_length:
            y = librosa.util.fix_length(y, self.audio_length)
        else:
            y = y[:self.audio_length]
        return y 

    def preprocess(self, y, **kwargs):
        # Preprocess
        params = {'original': y, **kwargs}
        if 'shift' in self.preprocesses:
            y, params['start'], params['end'], params['shift'] = pitch_shift(y, self.opt.sample_rate)
        if 'mel' in self.preprocesses:
            y = hz_to_mel(y) 
        if 'mulaw' in self.preprocesses:
            y = mulaw(y, self.opt.mu)
            params['original'] = mulaw(params['original'], self.opt.mu)
        # STFT
        if 'stft' in self.preprocesses:
            D = stft(y, n_fft=self.opt.nfft)
            lmag, params['phase'] = calc(D, self.opt.smoothing_factor) 
            if 'normalize' in self.preprocesses:
                lmag, params['max'], params['min'] = normalize_magnitude(lmag)
            return torch.from_numpy(lmag), params
        elif 'cqt' in self.preprocesses:
            D = librosa.cqt(y, sr=self.sample_rate,
                            bins_per_octave=self.opt.cqt_octave_bins,
                            n_bins=self.opt.tensor_height)
            lmag, params['phase'] = calc(D, self.opt.smoothing_factor) 
            if 'normalize' in self.preprocesses:
                lmag, params['max'], params['min'] = normalize_magnitude(lmag)
            return torch.from_numpy(lmag), params
        else:
            return torch.from_numpy(y), params
