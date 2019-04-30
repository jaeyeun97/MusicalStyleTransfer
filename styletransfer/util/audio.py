import librosa
import numpy as np
import torch
import math

def calc(D, eps):
    if isinstance(D, np.ndarray):
        lmag = np.log(np.abs(D) + eps)
        agl = np.angle(D)
    elif isinstance(D, torch.Tensor):
        real = D[:, 0 , :, :]
        comp = D[:, 1 , :, :]
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
        mmax = float(torch.max(lmag))
        mmin = float(torch.min(lmag))
    else:
        raise NotImplementedError('Cannot normalize.')
    if mmax - mmin > 0:
        lmag = 2 * (lmag - mmin) / (mmax - mmin) - 1
    else:
        raise ValueError('Cannot be normalized!')
        # print('Normalization Returning Zeros...')
        # lmag = np.zeros(tuple(lmag.shape))
    return lmag, mmax, mmin

def denormalize_magnitude(mmax, mmin, lmag):
    for i in lmag.shape[0]:
        mmax = float(mmax[i])
        mmin = float(mmin[i])
        return ((mmax - mmin) * (lmag[i, :] + 1) / 2) + mmin

def normalize_phase(agl):
    return agl / np.pi

def denormalize_phase(agl):
    return agl * np.pi

def combine_mag_phase(mag, agl):
    if isinstance(mag, np.ndarray):
        mag = torch.from_numpy(mag)
    if isinstance(agl, np.ndarray):
        agl = torch.from_numpy(agl)
    return torch.stack((mag, agl), 0)

def stft(y, **kwargs):
    return librosa.stft(y, **kwargs)

def istft(y, **kwargs):
    return np.stack([librosa.istft(y[i, :], **kwargs) for i in y.shape[0]])

def hz_to_mel(y, **kwargs):
    return librosa.hz_to_mel(y, **kwargs)

def mel_to_hz(y, **kwargs):
    return np.stack([librosa.mel_to_hz(y[i, :], **kwargs) for i in y.shape[0]])


def frame(y, sr, length=30, stride=15):
    return librosa.frame(y, frame_length=length*sr, hop_length=stride*sr)

def pitch_shift(y, sr):
    l = y.shape[-1]
    start = np.random.randint(0, l - sr) if l > sr else 0
    end = np.random.randint(start+0.25*sr, l-0.25*sr)
    shift = np.random.rand() - 0.5
    shifted = [
        y[:start],
        librosa.effects.pitch_shift(y[start:end], sr, shift),
        y[end:]
    ]
    return np.concatenate(shifted), start, end, shift

def pitch_deshift(y, sr, start, end, shift):
    res = list()
    for i in y.shape[0]:
        shifted = [
            y[i, :start],
            librosa.effects.pitch_shift(y[i, start:end], sr, -1 * shift),
            y[i, end:]
        ]
        res.append(np.concatenate(shifted))
    return np.stack(res)

def mulaw(x, MU):
    return np.sign(x) * np.log(1. + MU * np.abs(x)) / np.log(1. + MU)

def inv_mulaw(x, MU):
    return np.sign(x) * (1. / MU) * (np.power(1. + MU, np.abs(x)) - 1.)

