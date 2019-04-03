import os
import sys
import glob
import librosa
import numpy as np
import soundfile as sf

def split_audio(y_path, y, sr, subdir='splits_2'):
    splits = librosa.effects.split(y, top_db=65, frame_length=sr, hop_length=sr//4)
    print(splits.shape)
    filename = os.path.basename(y_path).split('.')[0]
    dir = '{}/{}'.format(os.path.dirname(y_path), subdir)
    mkdir(dir)
    for i in range(len(splits)):
        if splits[i][1] - splits[i][0] > sr:
            librosa.output.write_wav(os.path.join(dir, '{}.{}.wav'.format(filename, i)), y[splits[i][0]:splits[i][1]], sr)
    print('Audio split completed for {}'.format(y_path))

def main():
    if len(sys.argv) < 2:
        print('Path not provided.')
        return
    paths = glob.glob('{}/*.wav'.format(sys.argv[1]))
    for p in paths:
        y, sr = sf.read(p, dtype='float32')
        y = y.mean(axis=1)
        split_audio(p, y, sr)

if __name__ == '__main__':
    main()
