from torch import nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels_img,
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
            nn.Sigmoid(),  # 0.0 -> 1.0
        )

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
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * (2 ** 4), 4, 1, 0),  # N x (features_g * 16) x 4x4
            self._block(features_g * (2 ** 4), features_g * (2 ** 3), 4, 2, 1),  # 8x8
            self._block(features_g * (2 ** 3), features_g * (2 ** 2), 4, 2, 1),  # 16x16
            self._block(features_g * (2 ** 2), features_g * (2 ** 1), 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g * (2 ** 1), channels_img, 4, 2, 1, bias=False),  # 64x64
            nn.Tanh(),  # -1.0 -> 1.0
        )

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

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

