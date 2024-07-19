import torch
import torch.nn as nn
from torchvision.models import vgg19


class ResidualBlock(nn.Module):
    def __init__(self, in_features) -> None:
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(num_parameters=in_features),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.residual_block(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, scale_factor) -> None:
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * scale_factor**2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.pixel_shuffle(self.conv(x)))


class Generator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_residual_block=16):
        super(Generator, self).__init__()

        self.conv_inp = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(num_parameters=64),
        )
        resblocks = [ResidualBlock(64) for _ in range(n_residual_block)]
        self.res_block = nn.Sequential(*resblocks)

        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64)
        )

        self.upsamples = nn.Sequential(
            Upsample(64, scale_factor=2), Upsample(64, scale_factor=2)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv_inp(x)
        out = self.res_block(out1)
        out2 = self.mid_conv(out)
        out = torch.add(out1, out2)
        out = self.upsamples(out)
        out = self.final_conv(out)

        return out


class DescConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, **kwargs) -> None:
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.cnn(x)))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                DescConv(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,  # 1,2,1,2,1,2......
                    padding=1,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(
                512 * 6 * 6, 1024
            ),  # opt from last DescConv block torch.Size([5, 512, 6, 6]) so
            # 512 * 6 * 6
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)


class VggFeatureExtractor(nn.Module):
    def __init__(self):
        super(VggFeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features.children())[:18]
        )

    def forward(self, img):
        return self.feature_extractor(img)
