from pathlib import Path
from time import time

import torch
from torch import optim
from torch.utils import data, tensorboard
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import Critic, Generator, initialize_weights
from utils import gradient_penalty

data_dir = Path(__file__).parent.parent / 'data'

lr = 1e-4
batch_size = 512 if torch.cuda.is_available() else 32
data_loader_workers = 4 if torch.cuda.is_available() else 0
data_loader_pin_memory = torch.cuda.is_available()
img_size = 64
channels_img = 1
z_dim = 100
num_classes = 10
gen_embed_size = 100
num_epochs = 5
features_disc = 64
features_gen = 64
critic_iterations = 5
lambda_gradient_penalty = 10
batches_per_stats = 20 if torch.cuda.is_available() else 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5] * channels_img,
        [0.5] * channels_img,
    ),
])

# real_dataset = datasets.ImageFolder(root=(data_dir / dataset_name), transform=transform)
real_dataset = datasets.MNIST("data", download=True, transform=transform)

real_data_loader = data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=data_loader_workers,
                                   data_loader_pin_memory=True)

gen = Generator(z_dim, channels_img, features_gen, num_classes, gen_embed_size).to(device)
critic = Critic(channels_img, features_disc, num_classes, img_size).to(device)
initialize_weights(gen)
initialize_weights(critic)

gen_optim = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
critic_optim = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

tb_fake_writer = tensorboard.SummaryWriter("logs/fake")
tb_real_writer = tensorboard.SummaryWriter("logs/real")
tb_model_writer = tensorboard.SummaryWriter("logs/model")

step = 0
start_t = time()

for epoch in range(1, num_epochs + 1):

    for batch_idx, (real_imgs, real_labels) in enumerate(real_data_loader):

        real_imgs = real_imgs.to(device)
        real_labels = real_labels.to(device)

        # ensure these are in scope for use after for loop
        fake_imgs = None
        critic_loss = None

        for _ in range(critic_iterations):
            noise = torch.randn(real_imgs.shape[0], z_dim, 1, 1).to(device)
            fake_imgs = gen(noise, real_labels)

            critic_pred_real = critic(real_imgs, real_labels).reshape(-1)
            critic_pred_fake = critic(fake_imgs, real_labels).reshape(-1)

            penalty = gradient_penalty(critic, real_imgs, real_labels, fake_imgs, device)

            critic_loss = (
                    -(torch.mean(critic_pred_real) - torch.mean(critic_pred_fake))
                    + (lambda_gradient_penalty * penalty)
            )

            critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optim.step()

        critic_pred_fake = critic(fake_imgs, real_labels).reshape(-1)
        gen_loss = -torch.mean(critic_pred_fake)

        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()

        if batch_idx % 100 == 0:
            print(f"[{time() - start_t:0.1f}] epoch [{epoch}/{num_epochs}] batch [{batch_idx}/{len(real_data_loader)}]")

            tb_real_writer.add_image('real', make_grid(real_imgs[:32], normalize=True), global_step=step)
            tb_fake_writer.add_image('fake', make_grid(fake_imgs[:32], normalize=True), global_step=step)
            tb_model_writer.add_scalar('loss/critic', critic_loss, global_step=step)
            tb_model_writer.add_scalar('loss/generator', gen_loss, global_step=step)

            step += 1
