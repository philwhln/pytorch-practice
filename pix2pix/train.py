import os
from pathlib import Path
from time import time

import torch
from torch.utils import data, tensorboard
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

from datasets import load_dataset
from model.generator import Generator
from model.discriminator import Discriminator
import checkpoint

DEBUG = True
RUNNING_IN_INTELLIJ = bool(os.environ.get("PYCHARM_HOSTED", 0))
DEBUG_SHOW_IMAGES = DEBUG and RUNNING_IN_INTELLIJ

DATASET_NAME = "facades"
BATCH_SIZE = 8
EPOCHS = 10
NUM_WORKERS = 0
LOSS_L1_LAMBDA = 100
LEARNING_RATE = 2e-4
MOMENTUM_BETA_1 = 0.5
MOMENTUM_BETA_2 = 0.999

CHECKPOINT_NAME_DISCRIMINATOR = "pix2pix.discriminator"
CHECKPOINT_NAME_GENERATOR = "pix2pix.generator"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True

train_dataset, val_dataset = load_dataset(DATASET_NAME)

train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

if DEBUG_SHOW_IMAGES:
    import matplotlib.pyplot as plt

    x, y = next(iter(train_dataloader))
    print(x[0].shape, y[0].shape)
    assert x[0].shape == (3, 256, 256)
    assert y[0].shape == (3, 256, 256)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x[0].numpy().transpose((1, 2, 0)))
    axs[1].imshow(y[0].numpy().transpose((1, 2, 0)))
    plt.show()

discriminator = Discriminator().to(device)
generator = Generator().to(device)

bce_loss_fn = nn.BCEWithLogitsLoss()
l1_loss_fn = nn.L1Loss()

optimizer_discriminator = optim.Adam(params=discriminator.parameters(), lr=LEARNING_RATE,
                                     betas=(MOMENTUM_BETA_1, MOMENTUM_BETA_2))
optimizer_generator = optim.Adam(params=generator.parameters(), lr=LEARNING_RATE,
                                 betas=(MOMENTUM_BETA_1, MOMENTUM_BETA_2))

# TODO: use GradScaler and autocast for mixed precision training
#grad_scaler_discriminator = torch.cuda.amp.GradScaler()
#grad_scaler_generator = torch.cuda.amp.GradScaler()

start_t = time()
step = 0

tb_writer = tensorboard.SummaryWriter(Path(__file__).parent.parent / "logs" / f"{start_t:0.0f}")

checkpoint.load(CHECKPOINT_NAME_GENERATOR, generator, optimizer_generator, LEARNING_RATE, device)
checkpoint.load(CHECKPOINT_NAME_DISCRIMINATOR, discriminator, optimizer_discriminator, LEARNING_RATE, device)

for epoch in range(1, EPOCHS + 1):
    num_batches = len(train_dataloader)
    for batch_idx, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        predictions_real = discriminator(x, y)
        loss_discriminator_real = bce_loss_fn(predictions_real, torch.ones_like(predictions_real))

        y_fake = generator(x)
        predictions_fake = discriminator(x, y_fake.detach())
        loss_discriminator_fake = bce_loss_fn(predictions_fake, torch.zeros_like(predictions_fake))

        loss_discriminator = loss_discriminator_real + loss_discriminator_fake
        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()

        predictions_fake = discriminator(x, y_fake)
        loss_discriminator_fake = bce_loss_fn(predictions_fake, torch.ones_like(predictions_fake))
        loss_generator = loss_discriminator_fake + (LOSS_L1_LAMBDA * l1_loss_fn(y_fake, y))
        optimizer_generator.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        print(f"[{time() - start_t:0.1f}] epoch={epoch} batch={(batch_idx + 1):02d}/{num_batches} "
              f"loss_dis={loss_discriminator:0.3f} loss_gen={loss_generator:0.3f}")

        if batch_idx % 3 == 0:
            step += 1
            imgs = torch.cat([x[0], y[0], y_fake[0]], dim=-1)
            tb_writer.add_image('real', make_grid(imgs, normalize=True), global_step=step)
            tb_writer.add_scalar('loss/discriminator', loss_discriminator, global_step=step)
            tb_writer.add_scalar('loss/generator', loss_generator, global_step=step)

    if epoch % 5 == 0:
        checkpoint.save(CHECKPOINT_NAME_DISCRIMINATOR, discriminator, optimizer_discriminator)
        checkpoint.save(CHECKPOINT_NAME_GENERATOR, generator, optimizer_generator)
