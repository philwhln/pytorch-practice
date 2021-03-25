from pathlib import Path
from time import time

import torch
from torch import nn, optim, randn
from torch.utils import data, tensorboard
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import Discriminator, Generator, initialize_weights


data_dir = Path(__file__).parent.parent / 'data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 2e-4
batch_size = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 5
features_disc = 64
features_gen = 64

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5] * channels_img,
        [0.5] * channels_img,
    ),
])

mnist_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
data_loader = data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim, channels_img, features_gen)
dis = Discriminator(channels_img, features_disc)
initialize_weights(gen)
initialize_weights(dis)

gen_optim = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
dis_optim = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))

loss_fn = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

tb_fake_writer = tensorboard.SummaryWriter("logs/fake")
tb_real_writer = tensorboard.SummaryWriter("logs/real")

step = 0
start_t = time()

for epoch in range(1, num_epochs + 1):

    for batch_idx, (real_imgs, _) in enumerate(data_loader):

        real_imgs = real_imgs.to(device)
        noise = torch.randn(real_imgs.shape[0], z_dim, 1, 1).to(device)
        fake_imgs = gen(noise)

        # train discriminator to maximize [log(D(real) + log(1 - D(G(noise))]
        dis_pred_real = dis(real_imgs).reshape(-1)
        dis_loss_real = loss_fn(dis_pred_real, torch.ones_like(dis_pred_real))
        dis_pred_fake = dis(fake_imgs).reshape(-1)
        dis_loss_fake = loss_fn(dis_pred_fake, torch.zeros_like(dis_pred_fake))
        dis_loss = (dis_loss_real + dis_loss_fake) / 2

        dis_optim.zero_grad()
        dis_loss.backward(retain_graph=True)
        dis_optim.step()

        dis_pred_fake = dis(fake_imgs).reshape(-1)
        gen_loss = loss_fn(dis_pred_fake, torch.ones_like(dis_pred_fake))

        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()

        #if batch_idx % 100 == 0:

        print(f"[{time() - start_t:0.1f}] epoch [{epoch}/{num_epochs}] batch [{batch_idx}/{len(data_loader)}]")

        real_imgs = real_imgs[:32]
        fake_imgs = gen(fixed_noise)

        tb_real_writer.add_image('real', make_grid(real_imgs, normalize=True), global_step=step)
        tb_fake_writer.add_image('fake', make_grid(fake_imgs, normalize=True),  global_step=step)

        step += 1
