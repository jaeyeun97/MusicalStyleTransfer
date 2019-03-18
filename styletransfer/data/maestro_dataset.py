"""
Maestro + GuitarSet Dataset class

For training a CycleGAN Network.
"""
from .single_dataset import SingleDataset

import os
import pandas


class MaestroDataset(SingleDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, prefix, is_train): 
        parser = SingleDataset.modify_commandline_options(parser, prefix, is_train)
        set_defaults = SingleDataset.get_default_setter(parser, prefix)
        set_defaults(dataroot='./datasets/maestro-v1.0.0', max_dataset_size=1200) 
        return parser

    def __init__(self, opt, prefix):
        SingleDataset.__init__(self, opt, prefix)

        maestro_path = self.root 
        maestro_name = os.path.basename(maestro_path)
        maestro_file = os.path.join(maestro_path, "{}.csv".format(maestro_name))
        maestro_meta = pandas.read_csv(maestro_file)

        splits = (maestro_meta.duration // self.duration).rename('splits')
        maestro_meta = maestro_meta.join(splits)[['audio_filename', 'splits']]
        self.paths = list()
        for index, row in maestro_meta.iterrows():
            for split in range(int(row.splits)):
                f = os.path.join(maestro_path, row.audio_filename)
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
        return min(len(self.paths), self.get_opt('max_dataset_size'))
