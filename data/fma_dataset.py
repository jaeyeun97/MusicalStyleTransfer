
"""
FMA Dataset class

For training a CycleGAN Network.
Using FMA_large dataset, which is already trimmed to 30 seconds.
"""
from data.base_dataset import BaseDataset
from util.fma import FMA

import os
import random


DATA_LEN = 30


class FMADataset(BaseDataset):
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
        parser.add_argument('--metadata_subdir', type=str, default='fma_metadata', help='FMA metadata directory')
        parser.add_argument('--audio_subdir', type=str, default='fma_medium', help='FMA audio data directory')
        parser.add_argument('--A_genre', type=str, default='Classical', help='Genre title of domain A')
        parser.add_argument('--B_genre', type=str, default='Jazz', help='Genre title of domain B')
        parser.set_defaults(max_dataset_size=4000, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt, DATA_LEN)
        self.A_genre = opt.A_genre.split(',')
        self.B_genre = opt.B_genre.split(',')

        metapath = os.path.join(self.root, opt.metadata_subdir)
        audiopath = os.path.join(self.root, opt.audio_subdir)

        self.fma = FMA(metapath, audiopath)
        self.A_paths, self.B_paths = self.get_fma_tracks()

        self.A_size = len(self.A_paths)
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

        a_max, a_min, A = self.get_data(A_path)
        b_max, b_min, B = self.get_data(B_path)

        return {
            'A': A,
            'B': B,
            'A_path': A_path,
            'B_path': B_path,
            'A_max': a_max,
            'A_min': a_min,
            'B_max': b_max,
            'B_min': b_min
        }

    def __len__(self):
        """Return the total number of images."""
        return max(len(self.A_paths), len(self.B_paths))

    def get_fma_tracks(self):
        all_genres = self.fma.get_all_genres()

        if 'all' in self.A_genre:
            self.A_genre = all_genres
        if 'all' in self.B_genre:
            self.B_genre = all_genres

        if all(g not in all_genres for g in self.A_genre) \
                or all(g not in all_genres for g in self.B_genre):
            raise Exception('Genre not available! Available genres can be found in the documentation')

        A_ids = self.fma.get_genre_ids(self.A_genre)
        B_ids = self.fma.get_genre_ids(self.B_genre)

        A_paths = self.fma.get_track_ids_by_genres(A_ids).map(self.fma.get_audio_path).tolist()
        B_paths = self.fma.get_track_ids_by_genres(B_ids).map(self.fma.get_audio_path).tolist()

        A_paths = self.trim_dataset(A_paths)
        B_paths = self.trim_dataset(B_paths)

        return A_paths, B_paths
