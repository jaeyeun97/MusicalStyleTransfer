from util.fma import FMA
import IPython

meta_dir = './datasets/fma/fma_metadata'
audio_dir = './datasets/fma/fma_medium'
fma = FMA(meta_dir, audio_dir)

IPython.embed()
