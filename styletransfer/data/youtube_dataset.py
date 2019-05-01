"""
YouTube Dataset class
"""
from .single_dataset import SingleDataset

import os
import librosa
import glob


class YoutubeDataset(SingleDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, prefix, is_train): 
        parser = SingleDataset.modify_commandline_options(parser, prefix, is_train)
        set_defaults = SingleDataset.get_default_setter(parser, prefix)
        set_defaults(dataroot='./datasets/youtube', max_dataset_size=5000) 
        return parser

    def __init__(self, opt, prefix):
        """Initialize this dataset class.

        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        SingleDataset.__init__(self, opt, prefix) 
        # get the audio path
        audio_path = os.path.abspath(self.root)
        audio_paths = glob.glob(os.path.join(audio_path, '*.wav')) 
        self.paths = list()
        for f in audio_paths:
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

        return self.preprocess(data)

    def __len__(self):
        """Return the total number of audio files."""
        return min(len(self.paths), self.get_opt('max_dataset_size'))
