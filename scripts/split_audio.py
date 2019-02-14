from ..util import split_audio as sa

import sys
import glob
import librosa

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Path not provided')
        return
    paths = glob.glob('{}/*.wav'.format(sys.argv[1]))
    for p in paths:
        y, sr = librosa.load(p)
        sa(p, y, sr)
