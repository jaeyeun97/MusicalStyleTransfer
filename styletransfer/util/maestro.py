import pandas
import os


class Maestro(object):

    def __init__(self, dirpath):
        self.path = os.path.abspath(dirpath)
        self.dirname = os.path.basename(dirpath) 
        meta_file = os.path.join(self.path, "{}.csv".format(self.dirname))
        self.metadata = pandas.read_csv(meta_file)

    def get_paths_splits(self, audio_length):
        splits = (self.metadata.duration // audio_length).rename('splits')
        return self.metadata.join(splits)[['audio_filename', 'splits']].to_dict(orient='records')
