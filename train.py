import os
import time
import librosa
from styletransfer.options.train_options import TrainOptions
from styletransfer.data import create_dataset
from styletransfer.models import create_model
from styletransfer.util import mkdir

if __name__ == "__main__":
    opt = TrainOptions().parse()   # get training options
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training clips = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    loss_names = model.loss_names
    loss_dir = os.path.join('losses', opt.name)
    if opt.continue_train:
        loss_file = open(os.path.join(loss_dir, 'losses.csv'), 'a')
    else:
        mkdir(loss_dir)
        loss_file = open(os.path.join(loss_dir, 'losses.csv'), 'w')
        loss_file.write('epoch,iter,' + ','.join(loss_names) + '\n')
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            try:
                model.set_input(data)         # unpack data from dataset and apply preprocessing
            except ValueError as e:
                continue
            model.train()   # calculate loss functions, get gradients, update network weights
            
            losses = model.get_current_losses()
            loss_file.write('{},{},'.format(epoch, i) + ','.join(str(losses[k]) for k in loss_names) + '\n')
            loss_file.flush() 

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print("losses: {}".format(losses))
                print("time for computation: {}".format(t_comp))    

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            iter_data_time = time.time()

        if epoch % opt.save_sample_epoch_freq == 0:
            clips = model.get_current_audio()
            epoch_dir = os.path.join(opt.results_dir, 'train', opt.name, "{:03d}".format(epoch))
            mkdir(epoch_dir)
            for name, y in clips.items():
                for i in range(y.shape[0]):
                    librosa.output.write_wav(os.path.join(epoch_dir, '{}.{}.wav'.format(name, i)), y[i, :], opt.sample_rate)

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    loss_file.close()
