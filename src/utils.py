import os
import torch
import random
from PIL import Image
import multiprocessing
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from src.data_loaders import SuperResolutionDataLoader
from src.config import cfg


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


def show_image(train_paths):
    dataset = SuperResolutionDataLoader(train_paths)
    loader = DataLoader(dataset, batch_size=2, num_workers=4)
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for idx, (low_res, high_res) in enumerate(loader):
        # Display the first image in the left subplot
        axs[0].imshow(low_res[0].permute(1, 2, 0))
        axs[0].set_title("low res")

        # Display the second image in the right subplot
        axs[1].imshow(high_res[0].permute(1, 2, 0))
        axs[1].set_title("high res")

        if idx == 0:
            break

    # Show the figure
    plt.show()


def plot_examples(gen, data_loader, epoch):

    loader = data_loader

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    chosen_batch = random.randint(0, len(loader) - 1)

    for idx, (low_res, high_res) in enumerate(loader):
        if chosen_batch == idx:
            chosen = random.randint(0, len(low_res) - 1)
            fig.suptitle(f"Epoch : {epoch + 1}")
            axs[0].imshow(low_res[chosen].permute(1, 2, 0))
            axs[0].set_title("Low Res")
            axs[0].axis("off")

            with torch.no_grad():
                upscaled_img = gen(low_res[chosen].to(cfg.device.device).unsqueeze(0))

            axs[1].imshow(upscaled_img.cpu().permute(0, 2, 3, 1)[0])
            axs[1].set_title("Predicted")
            axs[1].axis("off")

            axs[2].imshow(high_res[chosen].permute(1, 2, 0))
            axs[2].set_title("High Res")
            axs[2].axis("off")

            break

    plt.show()
    plt.savefig(f"output/images/epoch_{epoch}_{chosen_batch}.png")
    gen.train()
