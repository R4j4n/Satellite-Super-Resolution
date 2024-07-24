import math
import PIL
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.config import cfg
from src.metrics import ssim
from src.models.srgan import Generator
from src.data_loaders import SuperResolutionInference


class Infernece:

    def __init__(self) -> None:
        self.model_pth = cfg.inference.model_pth
        self.model = self.load_model()

    def load_model(self):
        generator = Generator()
        generator.load_state_dict(torch.load(cfg.inference.model_pth))
        generator.eval()
        return generator

    def convert_tensor_to_image(self, tensor: torch.tensor, batch: bool = False):
        transform = T.ToPILImage()
        if batch:
            img = transform(tensor[0])
        else:
            img = transform(tensor)
        return img

    def __call__(self, image_pth, mean_transform: bool = False) -> PIL.Image.Image:

        # initialize the dataloader
        dataset = SuperResolutionInference(single_image_pth=image_pth)

        inference_loader = DataLoader(dataset, batch_size=1, num_workers=4)

        lr, hr = next(iter(inference_loader))

        # convert all to device
        lr = lr.to(cfg.device.device)
        hr = hr.to(cfg.device.device)
        self.model = self.model.to(cfg.device.device)

        tensor = self.model(lr)

        ############ Metrics Calculation #################
        valid_results = {
            "mse": 0,
            "ssims": 0,
            "psnr": 0,
            "ssim_total": 0,
            "batch_sizes": 0,
        }

        batch_size = lr.size(0)
        valid_results["batch_sizes"] += batch_size

        batch_mse = ((tensor - hr) ** 2).data.mean()
        valid_results["mse"] += batch_mse * batch_size
        batch_ssim = ssim(tensor, hr).item()  # Calling the original ssim function
        valid_results["ssims"] += batch_ssim * batch_size
        valid_results["psnr"] = 10 * math.log10(
            (hr.max() ** 2) / (valid_results["mse"] / valid_results["batch_sizes"])
        )
        valid_results["ssim_total"] = (
            valid_results["ssims"] / valid_results["batch_sizes"]
        )

        # covert all images to pil
        predicted_lr = self.convert_tensor_to_image(tensor, batch=True)
        hr = self.convert_tensor_to_image(hr, batch=True)
        lr = self.convert_tensor_to_image(lr, batch=True)

        # delete model and empty cache
        del self.model

        return hr, lr, predicted_lr, valid_results
