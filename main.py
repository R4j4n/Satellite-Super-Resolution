# builtin
import glob
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# all imports
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from sklearn.model_selection import train_test_split


# our modules
from src.config import cfg, root_path
from src.utils import MeanSTDFinder
from src.data_loaders import SuperResolutionDataLoader
from src.models.srgan import Generator, Discriminator, VggFeatureExtractor


# create path for models checkpoint
Path(root_path).joinpath("saved_models/srgan").mkdir(exist_ok=True, parents=True)
Path(root_path).joinpath("saved_models/srgan/images").mkdir(exist_ok=True, parents=True)


# get the images dataset path
images_pth = cfg.dataset.images_dir

train_paths, test_paths = train_test_split(
    sorted(glob.glob(images_pth + "/*.*"))[:500],
    test_size=0.2,
    random_state=42,
)

# get the mean and std of the dataset
mean_std = MeanSTDFinder(images_dir=images_pth)()


# load the dataloaders
train_dataloader = DataLoader(
    SuperResolutionDataLoader(train_paths, **mean_std),
    batch_size=cfg.train.batch_size,
    shuffle=True,
    num_workers=cfg.train.n_cpu,
)
test_dataloader = DataLoader(
    SuperResolutionDataLoader(test_paths, **mean_std),
    batch_size=int(cfg.train.batch_size * 0.75),
    shuffle=True,
    num_workers=cfg.train.n_cpu,
)

########## Define the Model Parameters ##########
generator = Generator()
discriminator = Discriminator()

feature_extractor = VggFeatureExtractor()
feature_extractor.eval()

gan_loss = torch.nn.BCEWithLogitsLoss()
content_loss = torch.nn.L1Loss()


# Transfer all to the device
generator = generator.to(cfg.device.device)
discriminator = discriminator.to(cfg.device.device)
feature_extractor = feature_extractor.to(cfg.device.device)
gan_loss = gan_loss.to(cfg.device.device)
content_loss = content_loss.to(cfg.device.device)


# define the optimizers for generator and discriminator
optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=cfg.train.learning_rate,
    betas=(cfg.train.b1, cfg.train.b2),
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=cfg.train.learning_rate,
    betas=(cfg.train.b1, cfg.train.b2),
)

# train losses
train_gen_loss, train_disc_loss, train_counter = [], [], []
# test losses
test_gen_loss, test_disc_loss = [], []


for epoch in range(cfg.train.n_epochs):

    ############################ Training ####################
    gen_loss = 0
    disc_loss = 0
    train_bar = tqdm(train_dataloader, desc=f"Training")

    for batch_idx, imgs in enumerate(train_bar):

        generator.train()
        discriminator.train()

        low_res_ipt = imgs["lr"].to(cfg.device.device)
        high_res_ipt = imgs["hr"].to(cfg.device.device)
        #################### Generator ######################

        optimizer_G.zero_grad()
        generated_hr = generator(low_res_ipt)
        disc_opt = discriminator(generated_hr)

        # Adverserial loss
        loss_GAN = gan_loss(disc_opt, torch.ones_like(disc_opt))

        # content loss
        generated_features = feature_extractor(generated_hr)
        real_feaures = feature_extractor(high_res_ipt)
        loss_CONTENT = content_loss(generated_features, real_feaures)

        # total loss
        total_loss_generator = loss_CONTENT + 1e-3 * loss_GAN

        # backpropagate
        total_loss_generator.backward()
        optimizer_G.step()
        #################### discriminator ######################

        optimizer_D.zero_grad()

        real_disc_opt = discriminator(high_res_ipt)
        loss_D_real = gan_loss(real_disc_opt, torch.ones_like(real_disc_opt))

        fake_disc_opt = discriminator(generated_hr.detach())
        loss_D_fake = gan_loss(fake_disc_opt, torch.zeros_like(fake_disc_opt))

        # total loss
        total_disc_loss = (loss_D_real + loss_D_fake) / 2

        # backprop
        total_disc_loss.backward()
        optimizer_D.step()

        ################## Accumulate losses ###############

        gen_loss += total_loss_generator.item()
        disc_loss += total_disc_loss.item()

        train_bar.set_postfix(
            gen_loss=gen_loss / (batch_idx + 1), disc_loss=disc_loss / (batch_idx + 1)
        )
    train_gen_loss.append(gen_loss / len(train_dataloader))
    train_disc_loss.append(disc_loss / len(train_dataloader))

    ############################ Testing ####################
    gen_loss = 0
    disc_loss = 0
    test_bar = tqdm(test_dataloader, desc=f"Testing")

    for batch_idx, imgs in enumerate(test_bar):
        generator.eval()
        discriminator.eval()

        # get the inputs
        low_res_ipt = imgs["lr"].to(cfg.device.device)
        high_res_ipt = imgs["hr"].to(cfg.device.device)

        ############# Generator Eval ###############

        generated_hr = generator(low_res_ipt)
        disc_opt = discriminator(generated_hr)

        # Adverserial loss
        loss_GAN = gan_loss(disc_opt, torch.ones_like(disc_opt))

        # content loss
        generated_features = feature_extractor(generated_hr)
        real_feaures = feature_extractor(high_res_ipt)
        loss_CONTENT = content_loss(generated_features, real_feaures)

        # total loss
        total_loss_generator = loss_CONTENT + 1e-3 * loss_GAN

        #################### discriminator eval ######################

        real_disc_opt = discriminator(high_res_ipt)
        loss_D_real = gan_loss(real_disc_opt, torch.ones_like(real_disc_opt))

        fake_disc_opt = discriminator(generated_hr.detach())
        loss_D_fake = gan_loss(fake_disc_opt, torch.zeros_like(fake_disc_opt))

        # total loss
        total_disc_loss = (loss_D_real + loss_D_fake) / 2

        ############### Accumulate losses ##########################
        gen_loss += total_loss_generator.item()
        disc_loss += total_disc_loss.item()

        if random.uniform(0, 1) < 0.1:

            imgs_lr = nn.functional.interpolate(low_res_ipt, scale_factor=4)
            imgs_hr = make_grid(high_res_ipt, nrow=1, normalize=True)
            gen_hr = make_grid(generated_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_hr, imgs_lr, gen_hr), -1)
            save_image(
                img_grid, f"saved_models/srgan/images/{batch_idx}.png", normalize=False
            )

        test_bar.set_postfix(
            gen_loss=gen_loss / (batch_idx + 1), disc_loss=disc_loss / (batch_idx + 1)
        )
    test_gen_loss.append(gen_loss / len(test_dataloader))
    test_disc_loss.append(disc_loss / len(test_dataloader))

    torch.save(generator.state_dict(), "saved_models/srgan/generator.pth")
    torch.save(discriminator.state_dict(), "saved_models/srgan/discriminator.pth")
