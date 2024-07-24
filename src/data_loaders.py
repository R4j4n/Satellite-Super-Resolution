from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from src.config import cfg


class SuperResolutionDataLoader(Dataset):

    def __init__(
        self,
        paths,
        mean=[0.2903465, 0.31224626, 0.29810828],
        std=[0.1457739, 0.13011318, 0.12317199],
    ) -> None:
        super().__init__()

        self.items = paths

        # transforms for low resolution
        self.low_res_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (
                        cfg.images.high_resolution_height // 2,
                        cfg.images.high_resolution_width // 2,
                    ),
                    Image.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        # transforms for high resolution
        self.high_res_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (
                        cfg.images.high_resolution_height,
                        cfg.images.high_resolution_width,
                    ),
                    Image.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        img = Image.open(self.items[index % len(self.items)]).convert("RGB")

        img_lr = self.low_res_transforms(img)

        img_hr = self.high_res_transforms(img)

        return img_lr, img_hr
