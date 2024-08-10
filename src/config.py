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
c.images.scale_factor = 2
c.images.high_resolution_height = 256
c.images.high_resolution_width = 256


c.device = edict()
c.device.device = "cuda" if torch.cuda.is_available() else "cpu"

# clip
c.train = edict()
c.train.n_epochs = 1
c.train.batch_size = 16
c.train.learning_rate = 3e-4
c.train.n_cpu = multiprocessing.cpu_count() // 2

c.inference = edict()
c.inference.model_pth = str(Path(root_path).joinpath("output/best_model.pth"))

cfg = c
