import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True, use_leaky_relu: bool = True):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect",
                              bias=(not use_batch_norm))
        self.batch_norm = nn.BatchNorm2d(out_channels) if use_batch_norm else None
        self.relu = nn.LeakyReLU(0.2) if use_leaky_relu else nn.ReLU()

    def forward(self, x):
        output = self.conv(x)
        if self.batch_norm:
            output = self.batch_norm(output)
        return self.relu(output)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.5) if use_dropout else None
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.batch_norm(self.conv(x))
        if self.dropout:
            output = self.dropout(output)
        return self.relu(output)


class Generator(nn.Module):
    def __init__(self, img_channels=3, features=64):
        super(Generator, self).__init__()

        self.encoder_blocks = [EncoderBlock(**spec) for spec in [
            {"in_channels": img_channels, "out_channels": features * 1, "use_batch_norm": False},
            {"in_channels": features * 1, "out_channels": features * 2},
            {"in_channels": features * 2, "out_channels": features * 4},
            {"in_channels": features * 4, "out_channels": features * 8},
            {"in_channels": features * 8, "out_channels": features * 8},
            {"in_channels": features * 8, "out_channels": features * 8},
            {"in_channels": features * 8, "out_channels": features * 8},
            {"in_channels": features * 8, "out_channels": features * 8, "use_leaky_relu": False},
        ]]

        self.decoder_blocks = [DecoderBlock(**spec) for spec in [
            {"in_channels": features * 8 * 1, "out_channels": features * 8, "use_dropout": True},
            {"in_channels": features * 8 * 2, "out_channels": features * 8, "use_dropout": True},
            {"in_channels": features * 8 * 2, "out_channels": features * 8, "use_dropout": True},
            {"in_channels": features * 8 * 2, "out_channels": features * 8},
            {"in_channels": features * 8 * 2, "out_channels": features * 4},
            {"in_channels": features * 4 * 2, "out_channels": features * 2},
            {"in_channels": features * 2 * 2, "out_channels": features * 1},
        ]]

        self.final_conv = nn.ConvTranspose2d(features * 1 * 2, img_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        encoder_outputs = []
        output = x
        for encoder_idx, block in enumerate(self.encoder_blocks):
            output = block(output)
            encoder_outputs.append(output)

        for decoder_idx, block in enumerate(self.decoder_blocks):
            if decoder_idx > 0:
                encoder_idx = len(encoder_outputs) - decoder_idx - 1
                output = torch.cat([output, encoder_outputs[encoder_idx]], dim=1)
            output = block(output)

        output = torch.cat([output, encoder_outputs[0]], dim=1)
        output = self.final_conv(output)

        return self.tanh(output)


def test():
    img_channels = 3
    model = Generator(img_channels)
    model.eval()
    # batch norm does work with one example and 1x1 size per-channel
    x = torch.randn((2, 3, 256, 256))
    y = model(x)
    assert x.shape == y.shape
    print("test passed")


if __name__ == '__main__':
    test()
