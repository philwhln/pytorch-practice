# This is inspired by the following and attempt for me to recreate from memory
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py

import torch
from torch import nn, optim, randn
from torch.utils import data, tensorboard
from torchvision import datasets, transforms
from torchvision.utils import make_grid


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=128):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)


class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_units=256):
        super(Generator, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_units, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.seq(x)


lr = 3e-4
z_size = 64
batch_size = 32
discriminator_input_size = 1 * 28 * 28
discriminator_output_size = 1
normalize_mean = [0.5]
normalize_std = [0.5]
epochs = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize_mean, std=normalize_std),
])
mnist_dataset = datasets.MNIST("data", download=True, transform=transform)
data_loader = data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)


generator = Generator(input_size=z_size,
                      output_size=discriminator_input_size)
discriminator = Discriminator(input_size=discriminator_input_size,
                              output_size=discriminator_output_size)

loss_fn = nn.BCELoss()

generator_optimizer = optim.Adam(params=generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(params=discriminator.parameters(), lr=lr)

tb_fake_writer = tensorboard.SummaryWriter("logs/fake")
tb_real_writer = tensorboard.SummaryWriter("logs/real")

for epoch in range(1, epochs + 1):
    for batch_idx, (mnist_batch, _) in enumerate(data_loader):
        real_batch = torch.reshape(mnist_batch, (-1, discriminator_input_size))
        noise = randn(mnist_batch.shape[0], z_size)
        fake_batch = generator(noise)

        # pass both real and fake image through discriminator
        real_guesses = discriminator(real_batch)
        fake_guesses = discriminator(fake_batch)
        real_loss = loss_fn(real_guesses, torch.ones_like(real_guesses))
        fake_loss = loss_fn(fake_guesses, torch.zeros_like(fake_guesses))
        discriminator_loss = (real_loss + fake_loss) / 2

        # optimize discriminator
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()

        # pass just the fake images through the discriminator
        fake_guesses = discriminator(fake_batch)
        generator_loss = loss_fn(fake_guesses, torch.ones_like(fake_guesses))

        # optimize generator
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        if batch_idx == 0:
            print(f"epoch {epoch}")

            tb_real_writer.add_image('real', make_grid(mnist_batch), epoch)
            tb_fake_writer.add_image('fake', make_grid(torch.reshape(fake_batch, (-1, 1, 28, 28))), epoch)
