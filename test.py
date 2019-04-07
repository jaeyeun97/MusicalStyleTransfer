"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from styletransfer.options.test_options import TestOptions
from styletransfer.data import create_dataset
from styletransfer.models import create_model
from styletransfer.util import mkdir
import librosa


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    # opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options 
    sample_rate = opt.sample_rate
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    mkdir(opt.results_dir)
    if opt.eval:
        model.eval()
    s = 0
    total = 0
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        res = model.test()           # run inference
        if opt.phase == 'gan':
            clips = model.get_current_audio()  # get image results
            iter_dir = os.path.join(opt.results_dir, "{:03d}".format(i))
            mkdir(iter_dir)
            for name, y in clips.items():
                librosa.output.write_wav(os.path.join(iter_dir, '{}.wav'.format(name)), y, sample_rate)
        elif opt.phase == 'test':
            s += 1 if float(res[0]) <= 0.5 else 0
            s += 1 if float(res[1]) > 0.5 else 0
        total += 2
    if opt.phase == 'test':
        print(s/total)
