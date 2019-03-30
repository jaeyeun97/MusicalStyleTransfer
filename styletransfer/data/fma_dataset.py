
"""
FMA Dataset class
"""
from .single_dataset import SingleDataset
from ..util.fma import FMA

import os
import random


DATA_LEN = 30


class FMADataset(SingleDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, prefix, is_train):
        SingleDataset.modify_commandline_options(parser, prefix, is_train)
        add_argument = SingleDataset.get_argument_adder(parser, prefix)
        set_defaults = SingleDataset.get_default_setter(parser, prefix)
        add_argument('metadata_subdir', type=str, default='fma_metadata', help='FMA metadata directory')
        add_argument('audio_subdir', type=str, default='fma_medium', help='FMA audio data directory')
        add_argument('genre', type=str, default='Classical', help='Genre title of domain %s' % prefix)
        set_defaults(dataroot='./datasets/fma', max_dataset_size=1000)

        return parser

    def __init__(self, opt, prefix):
        """Initialize this dataset class.

        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        SingleDataset.__init__(self, opt, prefix)
        self.genre = set(self.get_opt('genre').split(','))

        metapath = os.path.join(self.root, self.get_opt('metadata_subdir'))
        audiopath = os.path.join(self.root, self.get_opt('audio_subdir'))

        self.num_splits = int(DATA_LEN // self.duration)

        self.fma = FMA(metapath, audiopath)
        self.paths = self.get_fma_tracks()
        self.size = len(self.paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
        index -- a random integer for data indexing
        """

        path_index = int(index // self.num_splits)
        split_index = index % self.num_splits
        path = self.paths[path_index]
        data = self.retrieve_audio(path, split_index)

        return data

    def __len__(self):
        """Return the total number of audio files."""
        return min(len(self.paths) * self.num_splits, self.get_opt('max_dataset_size'))

    def get_fma_tracks(self):
        all_genres = self.fma.get_all_genres()

        if 'all' in self.genre:
            self.genre = all_genres 

        if all(g not in all_genres for g in self.genre):
            raise Exception('Genre not available! Available genres can be found in the documentation')

        ids = self.fma.get_genre_ids(self.genre)
        paths = self.fma.get_track_ids_by_genres(ids).map(self.fma.get_audio_path).tolist()
        random.shuffle(paths)

        return paths
