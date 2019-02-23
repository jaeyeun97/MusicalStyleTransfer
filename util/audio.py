import librosa
import numpy as np
import torch

def calc(D, eps):
    if isinstance(D, np.ndarray):
        lmag = np.log(np.abs(D) + eps)
        agl = np.angle(D)
    elif isinstance(D, torch.Tensor):
        real = x[:, 0 , :, :]
        comp = x[:, 1 , :, :]
        lmag = torch.log(torch.sqrt(torch.pow(real,2) + torch.pow(comp, 2)) + eps)
        agl = torch.atan(torch.div(comp, real))
    else:
        raise NotImplementedError('D type not recognized')
    return lmag, agl

def decalc(mag, agl, eps):
    mag = np.exp(mag) - eps
    return mag * np.cos(agl) + (mag * np.sin(agl) * complex(0, 1))

def normalize_magnitude(lmag):
    if isinstance(lmag, np.ndarray):
        mmax = np.max(lmag)
        mmin = np.min(lmag)
    elif isinstance(lmag, torch.Tensor):
        mmax = torch.max(lmag)
        mmin = torch.min(lmag)
    else:
        raise NotImplementedError('Cannot normalize.')
    lmag = 2 * (lmag - mmin) / (mmax - mmin) - 1
    return lmag, mmax, mmin

def denormalize_magnitude(mmax, mmin, lmag):
    return ((mmax - mmin) * (lmag + 1) / 2) + mmin

def normalize_phase(agl):
    return agl / np.pi

def denormalize_phase(agl):
    return agl * np.pi

def combine_mag_phase(mag, agl):
    return torch.stack((mag, agl), 0)

def stft(y, **kwargs):
    return librosa.stft(y, **kwargs)

def istft(y, **kwargs):
    return librosa.istft(y, **kwargs)

def hz_to_mel(y, **kwargs):
    return librosa.hz_to_mel(y, **kwargs)

def mel_to_hz(y, **kwargs):
    return librosa.mel_to_hz(y, **kwargs)

def frame(y, sr, length=30, stride=15):
    return librosa.frame(y, frame_length=length*sr, hop_length=stride*sr)

def split_audio(y_path, y, sr, subdir='splits'):
    splits = librosa.effects.split(y)
    print(splits.shape)
    filename = os.path.basename(y_path).split('.')[0]
    dir = '{}/{}'.format(os.path.dirname(y_path), subdir)
    mkdir(dir)
    for i in range(len(splits)):
        if splits[i][1] - splits[i][0] > sr:
            librosa.output.write_wav(os.path.join(dir, '{}.{}.wav'.format(filename, i)), y[splits[i][0]:splits[i][1]], sr)
    print('Audio split completed for {}'.format(y_path))
