import torch
import torch.nn as nn

from cnn_block import CNNBlock


# take images x, y -> concat
class Discriminator(nn.Module):
    def __init__(self, img_channels=3):  # 256 input -> 30x30 output
        super(Discriminator, self).__init__()

        self.kernel_size = 4
        self.padding = 1
        self.padding_mode = "reflect"
        self.relu_leak_size = 0.2

        num_input_images = 2
        feature_channels = [64, 128, 256, 512]

        self.net = nn.Sequential()

        in_channels = img_channels * num_input_images
        for idx, out_channels in enumerate(feature_channels):
            if idx == 0:
                # first block
                self._add_block(idx, in_channels, out_channels, batch_norm=False, stride=2)
            elif idx < len(feature_channels) - 1:
                # middle blocks
                self._add_block(idx, in_channels, out_channels, batch_norm=True, stride=2)
            else:
                # last blocks
                self._add_block(idx, in_channels, out_channels, batch_norm=True, stride=1)
            in_channels = out_channels

        self.net.add_module("final_cnn", nn.Conv2d(in_channels, out_channels=1, kernel_size=self.kernel_size,
                                                   padding=self.padding, padding_mode=self.padding_mode))

    def _add_block(self, idx, in_channels, out_channels, batch_norm, stride=2):
        name = str(idx)  # follow torch.nn.Module naming convention here
        self.net.add_module(name,
                            CNNBlock(in_channels, out_channels, batch_norm=batch_norm, kernel_size=self.kernel_size,
                                     stride=stride, padding=self.padding, padding_mode=self.padding_mode,
                                     relu_leak=self.relu_leak_size))

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def test():
    img_channels = 3
    x = torch.randn((1, img_channels, 256, 256))
    y = torch.randn((1, img_channels, 256, 256))
    model = Discriminator(img_channels)
    output = model(x, y)
    print(output.shape)


if __name__ == '__main__':
    test()
