"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

import librosa
import os
import os.path

AUDIO_EXTENSIONS = ['.m4a', '.mp3', '.wav']


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    audios = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_audio_file(fname):
                path = os.path.join(root, fname)
                audios.append(path)
    return audios[:min(max_dataset_size, len(images))]


def default_loader(path):
    return librosa.load(path)


class AudioFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        auds = make_dataset(root)
        if len(auds) == 0:
            raise(RuntimeError("Found 0 audio files in: " + root + "\n"
                               "Supported audio extensions are: " +
                               ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.audios = auds
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.audios[index]
        audio = self.loader(path)
        if self.transform is not None:
            audio = self.transform(audio)
        if self.return_paths:
            return audio, path
        else:
            return audio

    def __len__(self):
        return len(self.audios)
