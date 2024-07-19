import os
import torch
from PIL import Image
import multiprocessing
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class ImagesLoader(Dataset):

    def __init__(self, images_dir: str, transform=None) -> None:

        self.images_dir = images_dir
        self.image_paths = [
            os.path.join(images_dir, img) for img in os.listdir(images_dir)
        ]
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


class MeanSTDFinder:
    """This class finds the mean and standard deviation of RGB
    images in a directory
    """

    def __init__(self, images_dir: str) -> None:

        # initialize the data loader
        self.dataset = ImagesLoader(images_dir=images_dir)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=64,
            shuffle=False,
            num_workers=multiprocessing.cpu_count() // 2,
        )

    def __call__(self) -> dict:
        mean = 0.0
        std = 0.0
        total_images = 0

        for images in self.dataloader:
            image_count_in_a_batch = images.size(0)
            images = images.view(image_count_in_a_batch, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

            total_images += image_count_in_a_batch

        mean /= total_images
        std /= total_images

        return {"mean": mean.numpy(), "std": std.numpy()}
