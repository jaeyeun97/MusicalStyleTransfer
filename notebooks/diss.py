# %%
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import soundfile as sf

# %%
y, sr = librosa.load('diss/data/youtube/youtube1.45.wav')
# y_harmonic, y_percussive = librosa.effects.hpss(y)

# %%
plt.figure(figsize=(20, 10), dpi=1200)
plt.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=0, hspace=0)
plt.margins(0, 0)
plt.axis('off')
CQT = librosa.cqt(y, sr=sr)
axes = librosa.display.specshow(librosa.amplitude_to_db(np.abs(CQT), ref=np.max), y_axis='cqt_hz')
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
plt.savefig('diss/spectrogram/cqt.png', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.close()
print('Saved cqt.png')


# %%
plt.figure(figsize=(20, 10), dpi=1200)
plt.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=0, hspace=0)
plt.margins(0, 0)
plt.axis('off')
STFT = librosa.stft(y)
axes = librosa.display.specshow(librosa.amplitude_to_db(np.abs(STFT), ref=np.max), y_axis='log')
axes.get_xaxis().set_visible(False)
axes.get_yaxis().set_visible(False)
plt.savefig('diss/spectrogram/stft.png', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.close()
print('Saved sfft.png')

# %%
y_hat = librosa.icqt(C=CQT, sr=sr)
librosa.output.write_wav('diss/data/librosa_cqt.wav', y_hat, sr)

# %%
y_hat = librosa.istft(STFT)
librosa.output.write_wav('diss/data/librosa_stft.wav', y_hat, sr)

# %%
print(STFT)
print(y_hat)
print(y )

#%%
STFT = librosa.stft(y)
print(STFT.shape)
print(STFT[:, 0 ])

#%%
y2, sr2 = librosa.load('diss/data/output2.300.wav')
s2 = librosa.stft(y2)
print(s2.shape)

# %%markdown
# Way forward: Set time limit to 5 minutes, compute STFT, compute np.abs() and np.angle()
# Then train on Conv3d, with sliding window of 2.5min
# TODO: check patchGAN // design some generator
# The reverse process: compute abs cos \theta  + abs sin \theta I, run inverse STFT.
# If this works, move on to WaveNet/pix2pix approach.

# %%
import torch
yt = torch.from_numpy(y)
x = torch.stft(yt, 2048)
print(x)
d = x.numpy()
print(d.shape)
d2 = d[:, :, 0] + d[:, :, 1] * np.complex(0,1)
print(d2.shape)
print(d2)
y2 = librosa.istft(d2)
librosa.output.write_wav('diss/data/librosa_stft.wav', y2, sr)

# %%

y2 = librosa.effects.split(y, top_db=65)
print(y2.shape)

#%%
for i in range(len(y2)):
    librosa.output.write_wav('diss/data/processed/youtube.1.{}.wav'.format(i), y[y2[i][0]:y2[i][1]], sr)

# %%
# y, sr = sf.read('/home/jaeyeun/Cambridge/diss/code/datasets/fma_large/000/000144.mp3', dtype='float32')
y, sr = librosa.load('/home/jaeyeun/Cambridge/diss/code/datasets/fma_large/000/000144.mp3', sr=)
print(y.dtype)
y = y.sum(axis=1) / 2
print(sr)

print(y[0])

# %%
y, sr = librosa.load(librosa.util.example_audio_file())
a = librosa.util.frame(y, frame_length=2048, hop_length=64)
print(a.shape)
print(len(a))
print(a[:,0].shape)

# %%
import os
import pandas as pd
import ast

def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks

# %%
dataset_path = '/home/jaeyeun/Cambridge/diss/code/datasets'
metadata_path = os.path.join(dataset_path, 'fma_metadata')
data_path = os.path.join(dataset_path, 'fma_large')

genres = load(os.path.join(metadata_path, 'genres.csv'))
tracks = load(os.path.join(metadata_path, 'tracks.csv'))

# %%
jazz_id = genres.index[genres['title'] == 'Jazz'].item()
# jazz_id in tracks['track', 'genres_all']
jazz_mask = tracks['track', 'genres_all'].apply(lambda row: jazz_id in row)

def get_audio(track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(data_path, tid_str[:3], tid_str + '.mp3')

tracks[jazz_mask].index.map(get_audio).tolist()


# %%
import torch
import librosa
import numpy as np

tsr = 13000
y, sr = librosa.load(librosa.util.example_audio_file(), sr=tsr)
y = librosa.hz_to_mel(y)
D = librosa.stft(y, n_fft=1024)
print(D.shape)
lmag = np.log(np.abs(D) + 1)
agl = np.angle(D) # / np.pi
lmag, agl = torch.from_numpy(lmag), torch.from_numpy(agl)
tensor = torch.stack((lmag, agl), 0)
tensor = tensor.squeeze()
mag = tensor[0, :, :].numpy()
agl = tensor[1, :, :].numpy()
mag = np.exp(mag) - 1
stft = mag * np.cos(agl) + (mag * np.sin(agl) * np.complex(0, 1))
y_hat = librosa.istft(stft)
y = librosa.mel_to_hz(y)
y_hat = librosa.mel_to_hz(y_hat)
# y = librosa.resample(y, sr, tsr)
# y_hat = librosa.resample(y, sr, tsr)
librosa.output.write_wav('datasets/librosa_orig.wav', y, sr)
librosa.output.write_wav('datasets/librosa_stft.wav', y_hat, sr)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 8, 4, stride=3, padding=3)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, (2, 2))
        # # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net = net.to('cuda:0')
print(net)

t = torch.randn(1, 2, 1025, 1025)
t = t.to('cuda:0')
out = net(t)
print(out)
out.size()

# %%
import librosa

y, sr = librosa.load('/home/jaeyeun/133641.mp3')
