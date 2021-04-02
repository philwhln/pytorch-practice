import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
                 padding=1, padding_mode="reflect", use_batch_norm=True, relu_leak=0.2):
        super(CNNBlock, self).__init__()
        bias = not use_batch_norm
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, padding_mode=padding_mode, bias=bias)
        )
        if use_batch_norm:
            self.net.add_module("batch_norm", nn.BatchNorm2d(out_channels))
        self.net.add_module("relu", nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.net(x)


# take images x, y -> concat
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features=64):  # 256 input -> 30x30 output
        super(Discriminator, self).__init__()

        num_input_images = 2

        self.net = nn.Sequential(
            *[CNNBlock(**spec) for spec in [
                {"in_channels": img_channels * num_input_images, "out_channels": features * 1, "use_batch_norm": False},
                {"in_channels": features * 1, "out_channels": features * 2},
                {"in_channels": features * 2, "out_channels": features * 4},
                {"in_channels": features * 4, "out_channels": features * 8, "stride": 1},
            ]],
            nn.Conv2d(features * 8, out_channels=1, kernel_size=4, padding=1, padding_mode="reflect")
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def test():
    img_channels = 3
    x = torch.randn((1, img_channels, 256, 256))
    y = torch.randn((1, img_channels, 256, 256))
    model = Discriminator(img_channels)
    output = model(x, y)
    assert output.shape == (1, 1, 30, 30)
    print("test passed")


if __name__ == '__main__':
    test()
