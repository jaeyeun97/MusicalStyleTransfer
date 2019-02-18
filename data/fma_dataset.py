
"""
FMA Dataset class

For training a CycleGAN Network.
Using FMA_large dataset, which is already trimmed to 30 seconds.
"""
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset
from util import mkdir
from util.fma import FMA

import csv
import os
import librosa
import torch


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
        parser.add_argument('--sample_rate', type=int, default=22050, help='Sample Rate to resample')
        parser.add_argument('--nfft', type=int, default=2048, help='Number of Frequency bins for STFT')
        parser.add_argument('--mel', type=bool, default=False, help='Use the mel scale')
        parser.add_argument('--metadata_subdir', type=str, default='fma_metadata', help='FMA metadata directory')
        parser.add_argument('--audio_subdir', type=str, default='fma_medium', help='FMA audio data directory')
        parser.add_argument('--A_genre'), type=str, default='Classical', help='Genre title of domain A')
        parser.add_argument('--B_genre'), type=str, default='Jazz', help='Genre title of domain B')
        parser.set_defaults(max_dataset_size=200, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.sample_rate = opt.sample_rate
        self.nfft = opt.nfft
        self.mel = opt.mel

        metapath = os.path.join(self.root, opt.metadata_subdir)
        audiopath = os.path.join(self.root, opt.audio_path)

        self.fma = FMA(metapath, audiopath)
        self.A_path, self.B_path = self.get_fma_tracks()

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

        A_audio = self.retrieve_audio(A_path)
        B_audio = self.retrieve_audio(B_path)

        A = self.transform(A_audio)
        B = self.transform(B_audio)

        return {'A': A, 'B': B, 'A_path': A_path, 'B_path', B_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.frame_paths)

    def get_fma_tracks(self):
        all_genres = self.fma.get_all_genres()
        if opt.A_genre not in all_genres or opt.B_genre not in all_genres:
            raise Exception('Genre not available! Available genres can be found in the documentation')

        A_id = self.fma.get_genre_id(opt.A_genre)
        B_id = self.fma.get_genre_id(opt.B_genre)

        A_paths = self.fma.get_track_ids_by_genre(A_id).map(self.fma.get_audio_path).tolist()
        B_paths = self.fma.get_track_ids_by_genre(B_id).map(self.fma.get_audio_path).tolist()

        A_paths = trim_dataset(A_paths)
        B_paths = trim_dataset(B_paths)

        return A_paths, B_paths

    def trim_dataset(self, paths):
        return paths[:min(opt.max_dataset_size, len(paths)]

    def retrieve_audio(self, path):
        # y, sr = sf.read(path, dtype='float32')
        # if sr != self.sample_rate:
        #     y = librosa.resample(y, sr, self.sample_rate)
        # return y
        return librosa.load(path, sr=self.sample_rate)

    def transform(self, frame):
        if self.mel:
            frame = self.hz_to_mel(frame)
        # STFT
        D = librosa.stft(frame, nfft=self.nfft)
        lmag, agl = self.librosa_calc(D)
        # TODO: add normalization
        return combine_mag_angle(lmag, agl)

    @staticmethod
    def librosa_calc(D):
        log_mag = np.log(np.abs(D))
        agl = np.angle(D)
        return torch.from_numpy(log_mag), torch.from_numpy(agl)

    @staticmethod
    def torch_calc(D):
        x = torch.from_numpy(D)
        real = x[:, : , :, 0]
        comp = x[:, : , :, 1]
        log_mag = torch.sqrt(2 * torch.log(real) + 2 * torch.log(comp))
        agl = torch.atan(torch.div(comp, real))
        return log_mag, agl

    @staticmethod
    def combine_mag_angle(mag, agl):
        return torch.stack((mag, agl), 2)
