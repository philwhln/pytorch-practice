import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Critic, self).__init__()
        channels_class = 1  # use one channel for class information
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(
                channels_img + channels_class,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d * (2 ** 0), features_d * (2 ** 1), 4, 2, 1),  # 16x16
            self._block(features_d * (2 ** 1), features_d * (2 ** 2), 4, 2, 1),  # 8x8
            self._block(features_d * (2 ** 2), features_d * (2 ** 3), 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * (2 ** 3), 1, kernel_size=4, stride=2, padding=0),  # 1x1
        )
        self.labels_embedding = nn.Embedding(num_classes, img_size * img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),    # affine ensures learnable parameters
            nn.LeakyReLU(0.2),
        )

    def forward(self, imgs, labels):
        labels_channel = self.labels_embedding(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([imgs, labels_channel], dim=1)  # add labels to img channels
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, embed_size):
        super(Generator, self).__init__()
        self.embed_size = embed_size
        self.net = nn.Sequential(
            self._block(channels_noise + embed_size, features_g * (2 ** 4), 4, 1, 0),  # N x (features_g * 16) x 4x4
            self._block(features_g * (2 ** 4), features_g * (2 ** 3), 4, 2, 1),  # 8x8
            self._block(features_g * (2 ** 3), features_g * (2 ** 2), 4, 2, 1),  # 16x16
            self._block(features_g * (2 ** 2), features_g * (2 ** 1), 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g * (2 ** 1), channels_img, 4, 2, 1, bias=False),  # 64x64
            nn.Tanh(),  # -1.0 -> 1.0
        )
        self.labels_embedding = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, noise, labels):
        labels_embedding = self.labels_embedding(labels).view(labels.shape[0], self.embed_size, 1, 1)
        x = torch.cat([noise, labels_embedding], dim=1)
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

