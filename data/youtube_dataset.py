
"""YouTube Dataset class

For training a CycleGAN Network.
"""
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset, default_loader
from util import mkdir

import csv
import os
import librosa
import soundfile as sf
import torch


class YoutubeDataset(BaseDataset):
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
        parser.add_argument('--stride', type=int, default=15, help='Stride in reading in audio (in secs)')
        parser.add_argument('--length', type=int, default=30, help='Length of each audio sample when processing (in secs)')
        parser.add_argument('--sample_rate', type=int, default=20480, help='Sample Rate to resample')
        parser.add_argument('--nfft', type=int, default=2048, help='Number of Frequency bins for STFT')
        parser.add_argument('--mel', type=bool, default=False, help='Use the mel scale')
        parser.add_argument('--subdir', type=str, default="splits", help='Subdir of audio data to use')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the audio path
        self.dirpath = os.path.isdir('{}/{}', self.root, opt.subdir)

        # TODO: Use cache if it exists
        # self.cachedir = os.path.join(self.dirpath, '__cache')
        # mkdir(self.cachedir)
        # metadata_file = os.path.join(self.cachedir, 'metadata.csv')
        # if os.path.exist(metadata_file):
        #     with open(metadata_file, 'r') as f:
        #         metadata = json.load(f)
        # else:
        #     metadata = dict()

        self.audio_paths = sorted(make_dataset(self.dirpath, opt.max_dataset_size))
        self.frame_paths = list()
        self.sample_rate = opt.sample_rate
        self.length = opt.length
        self.stride = opt.stride
        self.nfft = opt.nfft
        self.subdir = opt.subdir
        self.mel = opt.mel

        for p in self.audio_paths:
            # Load audio
            filename = os.path.splitext(os.path.basename(p))[0]
            y, sr = sf.read(p, dtype='float32')
            t = librosa.get_duration(y=y, sr=sr)
            # Resample Audio
            if sr != self.sample_rate:
                y = librosa.resample(y, sr, self.sample_rate)
            # Pad the audio
            l = t % opt.stride
            if l != 0:
                t = t + 15 - l
                y = librosa.util.fix_length(y, t * self.sample_rate)
            # Length Check
            if t < self.length:
                print('Skipping {} due to short content length'.format(p))
                continue
            # Compute frames and store
            frames = self.frame(y, sr, length=self.length, stride=self.stride)
            # cannot store all the frames; store it into files
            for i in len(frames):
                fp = os.path.join(cachedir, '{}.{}.npy'.format(filename, i))
                np.save(fp, frames[:, i])
                self.frame_paths.append({
                    'file': filename,
                    'frame': i,
                    'path': fp
                })

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
        index -- a random integer for data indexing
        """
        metadata = self.frame_paths[index]
        frame = np.load(metadata['path'])
        data = self.transform(frame)
        result = {'data': data}
        result.update(metadata)
        return result

    def __len__(self):
        """Return the total number of images."""
        return len(self.frame_paths)

    def transform(self, frame):
        if self.mel:
            frame = hz_to_mel(frame)
        # STFT
        D = librosa_stft(frame, nfft=self.nfft)
        lmag, agl = librosa_calc(D)
        # TODO: add normalization
        return combine_mag_angle(lmag, agl)

    def librosa_stft(y, nfft=2048):
        return librosa.stft(y, nfft=nfft, win_length=30*self.sample_rate/1024)

    def torch_stft(x, nfft=2048):
        return torch.stft(x, nfft=nfft)

    def librosa_calc(D):
        log_mag = np.log(np.abs(D))
        agl = np.angle(D)
        return torch.from_numpy(log_mag), torch.from_numpy(agl)

    def torch_calc(D):
        x = torch.from_numpy(D)
        real = x[:, : , :, 0]
        comp = x[:, : , :, 1]
        log_mag = torch.sqrt(2 * torch.log(real) + 2 * torch.log(comp))
        agl = torch.atan(torch.div(comp, real))
        return log_mag, agl

    def combine_mag_angle(mag, agl):
        return torch.stack((mag, agl), 2)
