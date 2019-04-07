import librosa
import numpy as np
import torch

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
    mmax = float(mmax)
    mmin = float(mmin)
    return ((mmax - mmin) * (lmag + 1) / 2) + mmin

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
    return librosa.istft(y, **kwargs)

def hz_to_mel(y, **kwargs):
    return librosa.hz_to_mel(y, **kwargs)

def mel_to_hz(y, **kwargs):
    return librosa.mel_to_hz(y, **kwargs)

def frame(y, sr, length=30, stride=15):
    return librosa.frame(y, frame_length=length*sr, hop_length=stride*sr)

def pitch_shift(y, sr):
    l = y.shape[-1]
    s = l // sr
    rand_steps = np.random.rand(s) - 0.5
    shifted = [librosa.effects.pitch_shift(y[i*sr:min(sr*(i+1), l)], sr, rand_steps[i])
               for i in range(0, s)]
    return np.concatenate(shifted), rand_steps

def pitch_deshift(y, sr, steps):
    l = y.shape[-1]
    s = l // sr
    shifted = [librosa.effects.pitch_shift(y[i*sr:min(sr*(i+1), l)], sr, -1 * steps[i])
               for i in range(0, s)]
    return np.concatenate(shifted)

