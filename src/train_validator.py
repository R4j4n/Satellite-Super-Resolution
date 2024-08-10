import math
import torch
from tqdm.auto import tqdm

from src.config import cfg
from src.metrics import *


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=False)
    disc_loss_total = 0
    gen_loss_total = 0

    for idx, (low_res, high_res) in enumerate(loop):
        print(idx)
        high_res = high_res.to(cfg.device.device)
        low_res = low_res.to(cfg.device.device)

        fake = gen(low_res)

        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())

        disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = disc_loss_fake + disc_loss_real
        disc_loss_total += disc_loss.item()

        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss
        gen_loss_total += gen_loss.item()

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        loop.set_postfix(gen_loss=gen_loss.item(), disc_loss=disc_loss.item())

    return gen_loss_total / len(loop), disc_loss_total / len(loop)


def validate_fn(loader, disc, gen, mse, bce, vgg_loss):

    loop = tqdm(loader, leave=False)
    disc_loss_total = 0
    gen_loss_total = 0
    valid_results = {
        "mse": 0,
        "ssims": 0,
        "psnr": 0,
        "ssim_total": 0,
        "batch_sizes": 0,
    }  # Changed 'ssim' key to 'ssim_total'

    with torch.no_grad():
        for idx, (low_res, high_res) in enumerate(loop):

            high_res = high_res.to(cfg.device.device)
            low_res = low_res.to(cfg.device.device)

            fake = gen(low_res)

            disc_real = disc(high_res)
            disc_fake = disc(fake)

            ######### PSNR AND SSIM ###############
            batch_size = low_res.size(0)
            valid_results["batch_sizes"] += batch_size

            batch_mse = ((fake - high_res) ** 2).data.mean()
            valid_results["mse"] += batch_mse * batch_size
            batch_ssim = ssim(
                fake, high_res
            ).item()  # Calling the original ssim function
            valid_results["ssims"] += batch_ssim * batch_size
            valid_results["psnr"] = 10 * math.log10(
                (high_res.max() ** 2)
                / (valid_results["mse"] / valid_results["batch_sizes"])
            )
            valid_results["ssim_total"] = (
                valid_results["ssims"] / valid_results["batch_sizes"]
            )  # Storing the calculated SSIM value under 'ssim_total'

            disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
            disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))

            disc_loss = disc_loss_fake + disc_loss_real
            disc_loss_total += disc_loss.item()

            adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
            loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
            gen_loss = loss_for_vgg + adversarial_loss
            gen_loss_total += gen_loss.item()

            loop.set_postfix(
                gen_loss=gen_loss.item(),
                disc_loss=disc_loss.item(),
                ssim=valid_results["ssim_total"],
                psnr=valid_results["psnr"],
            )  # Accessing 'ssim_total'

    return (
        gen_loss_total / len(loop),
        disc_loss_total / len(loop),
        valid_results["psnr"],
        valid_results["ssim_total"],
    )
