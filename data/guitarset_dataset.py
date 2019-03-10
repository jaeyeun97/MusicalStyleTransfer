"""
Maestro + GuitarSet Dataset class

For training a CycleGAN Network.
"""
from data.base_dataset import SingleDataset

import os
import librosa
import glob


class GuitarsetDataset(SingleDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, prefix, is_train): 
        parser = SingleDataset.modify_commandline_options(parser, prefix is_train)
        set_defaults = SingleDataset.get_default_setter(parser, prefix)
        set_defaults(dataroot='./datasets/GuitarSet', max_dataset_size=4000) 
        return parser

    def __init__(self, opt, prefix):
        """Initialize this dataset class.

        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        SingleDataset.__init__(self, opt, prefix) 
        guitarset_path = os.path.abspath(self.root)
        guitarset_paths = glob.glob(os.path.join(guitarset_path, 'audio/audio_mic/*.wav')) 
        self.paths = list()
        for f in guitarset_paths:
            num_split = int(librosa.get_duration(filename=f) // self.duration)
            for split in range(num_split):
                self.paths.append('{}:{}'.format(f, str(split)))
        self.size = len(self.paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
        index -- a random integer for data indexing
        """

        path = self.paths[index] 
        audio, split = tuple(path.split(':'))
        split = int(split)
        data = self.retrieve_audio(audio, split)
        mmax, mmin, data = self.transform(data)

        return {
                'input': data,
                'path': audio,
                'split': split,
                'max': mmax,
                'min': mmin
                }

        def __len__(self):
            """Return the total number of audio files."""
        return len(self.paths)
