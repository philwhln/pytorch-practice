from pathlib import Path
from time import time

import torch
from torch import optim
from torch.utils import data, tensorboard
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import Critic, Generator, initialize_weights


data_dir = Path(__file__).parent.parent / 'data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 5e-5
batch_size = 64
image_size = 64
channels_img = 3
z_dim = 100
num_epochs = 5
features_disc = 64
features_gen = 64
critic_iterations = 5
weights_clip = 0.01

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5] * channels_img,
        [0.5] * channels_img,
    ),
])

mnist_dataset = datasets.ImageFolder(root=(data_dir / "celeb"), transform=transform)
data_loader = data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim, channels_img, features_gen).to(device)
critic = Critic(channels_img, features_disc).to(device)
initialize_weights(gen)
initialize_weights(critic)

gen_optim = optim.RMSprop(gen.parameters(), lr=lr)
critic_optim = optim.RMSprop(critic.parameters(), lr=lr)

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)

tb_fake_writer = tensorboard.SummaryWriter("logs/fake")
tb_real_writer = tensorboard.SummaryWriter("logs/real")
tb_model_writer = tensorboard.SummaryWriter("logs/model")

step = 0
start_t = time()

for epoch in range(1, num_epochs + 1):

    for batch_idx, (real_imgs, _) in enumerate(data_loader):

        real_imgs = real_imgs.to(device)

        for _ in range(critic_iterations):
            noise = torch.randn(real_imgs.shape[0], z_dim, 1, 1).to(device)
            fake_imgs = gen(noise)

            critic_pred_real = critic(real_imgs).reshape(-1)
            critic_pred_fake = critic(fake_imgs).reshape(-1)

            critic_loss = -(torch.mean(critic_pred_real) - torch.mean(critic_pred_fake))

            critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optim.step()

            # clip weights
            for p in critic.parameters():
                p.data.clamp_(-weights_clip, weights_clip)

        critic_pred_fake = critic(fake_imgs).reshape(-1)
        gen_loss = -torch.mean(critic_pred_fake)

        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()

        #if batch_idx % 100 == 0:

        print(f"[{time() - start_t:0.1f}] epoch [{epoch}/{num_epochs}] batch [{batch_idx}/{len(data_loader)}]")

        real_imgs = real_imgs[:32]
        fake_imgs = gen(fixed_noise)

        tb_real_writer.add_image('real', make_grid(real_imgs, normalize=True), global_step=step)
        tb_fake_writer.add_image('fake', make_grid(fake_imgs, normalize=True),  global_step=step)
        tb_model_writer.add_scalar('loss/critic', critic_loss, global_step=step)
        tb_model_writer.add_scalar('loss/generator', gen_loss, global_step=step)

        step += 1