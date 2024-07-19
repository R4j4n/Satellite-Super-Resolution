import torch
from pathlib import Path
from easydict import EasyDict as edict

root_path = str(Path(__file__).parent)

c = edict()

c.dataset = edict()
c.dataset.images_dir = str(Path(root_path).joinpath("data/Train/images_png"))


# dataloder
c.dataloader = edict()
c.dataloader.batch_size = 32
c.dataloader.num_workers = 16


c.device = edict()
c.device.device = "cuda" if torch.cuda.is_available() else "cpu"

# clip
c.clip = edict()
c.clip.temperature = 1.0


cfg = c
