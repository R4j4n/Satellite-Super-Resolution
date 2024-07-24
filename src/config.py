import torch
import multiprocessing
from pathlib import Path
from easydict import EasyDict as edict

root_path = str(Path(__file__).parent.parent)

c = edict()

c.dataset = edict()
c.dataset.images_dir = str(Path(root_path).joinpath("data/Train/images_png"))


# config for dataset scaling
c.images = edict()
c.images.channels = 3
c.images.scale_factor = 4
c.images.high_resolution_height = 256
c.images.high_resolution_width = 256


# dataloder
c.dataloader = edict()
c.dataloader.batch_size = 32
c.dataloader.num_workers = 16


c.device = edict()
c.device.device = "cuda" if torch.cuda.is_available() else "cpu"

# clip
c.train = edict()
c.train.n_epochs = 100
c.train.batch_size = 8
c.train.learning_rate = 0.00008
c.train.n_cpu = multiprocessing.cpu_count() // 2
c.train.b1 = 0.5  # adam: decay of first order momentum of gradient
c.train.b2 = 0.999  # adam: decay of second order momentum of gradient
c.train.decay_epoch = 100  # epoch from which to start lr decay

cfg = c
