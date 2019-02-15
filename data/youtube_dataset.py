
"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
-- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
-- <__init__>: Initialize this dataset class.
-- <__getitem__>: Return a data point and its metadata information.
-- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset
from data.audio_folder import make_dataset, default_loader

import librosa
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
        parser.add_argument('--hop_length', type=float, default=30.0, help='Stride in reading in audio')
        parser.add_argument('--frame_length', type=float, default=60.0, help='Length of each audio sample when processing')
        parser.add_argument('--stft-binsize', type=int, default=2048, help='Number of Frequency bins for STFT')
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
        dir = os.path.isdir('{}/{}', self.root, opt.subdir)
        self.audio_paths = sorted(make_dataset(dir, opt.max_dataset_size))
        # load audio, pad, split, store it
        for p in self.audio_paths:
            pass
        # otherwise, split the audio to set size and store
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
        index -- a random integer for data indexing

        Returns:
        a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = self.audio_paths[index]
        y, sr = librosa.load(path)
        return {'data_A': y, 'A_sr': sr, 'A_path': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.audio_paths)


    def librosa_stft(y, nfft=2048):
        return librosa.stft(y, nfft=nfft)

    def torch_stft(x, nfft=2048):
        return torch.stft(x, nfft=nfft)

    def librosa_calc(D):
        return np.array([(np.abs(D[i][j]), np.angle(D[i][j])) for i in range(len(D)) for j in range(len(D[i]))])

    def torch_calc(x):
        real = x[:, : , :, 0]
        comp = x[:, : , :, 1]
        mag = torch.sqrt(torch.add(torch.pow(real, 2), torch.pow(comp, 2)))
        agl = torch.atan(torch.div(comp, real))
        return torch.stack((mag, agl), 2)
