import random
import torch


class AudioPool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_clips = 0
            self.clips = []

    def query(self, clips):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return clips
        return_clips = list()
        for clip in clips:
            clips = torch.unsqueeze(clip.data, 0)
            if self.num_clips < self.pool_size:   # if the buffer is not full; keep inserting current clips to the buffer
                self.num_clips = self.num_clips + 1
                self.clips.append(clip)
                return_clips.append(clip)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current clip into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.clips[random_id].clone()
                    self.clips[random_id] = clip
                    return_clips.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_clips.append(clip)
        return_clips = torch.cat(return_clips, 0)   # collect all the images and return
        return return_images
