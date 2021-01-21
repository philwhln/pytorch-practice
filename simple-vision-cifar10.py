from pathlib import Path

import torch
from torch import optim, nn, tensor
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def calculate_loss(model, loss_fn, x, y, is_train):
    with torch.set_grad_enabled(is_train):
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
    assert loss.requires_grad == is_train
    return loss


def main():

    data_path = Path(__file__).parent / 'data' / 'cifar-10'

    cifar10_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())

    imgs_train = torch.stack([img_t for img_t, _ in cifar10_train], dim=3)

    # convert shape (3, 32, 32, 50000) to (3, 32 * 32 * 50000), then per channel mean (across the second dimension)
    imgs_train_mean = imgs_train.view(3, -1).mean(dim=1)
    print(f'imgs_train_mean = {imgs_train_mean}')
    imgs_train_std = imgs_train.view(3, -1).std(dim=1)
    print(f'imgs_train_std = {imgs_train_std}')

    cifar10_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(imgs_train_mean, imgs_train_std),
    ])

    cifar10_train = datasets.CIFAR10(data_path, train=True, download=False, transform=cifar10_transforms)
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform=cifar10_transforms)

    class_names = ['airplane', 'bird']

    # select just to images for labels [0, 2] and remap labels to [0, 1]
    label_remap = {0: 0, 2: 1}
    cifar10_train = [(img, label_remap[label]) for img, label in cifar10_train if label in label_remap.keys()]
    cifar10_val = [(img, label_remap[label]) for img, label in cifar10_val if label in label_remap.keys()]

    n_in = 3 * 32 * 32
    n_hidden = 512
    n_out = len(class_names)

    learning_rate = 1e-4

    x_train, y_train = cifar10_train[0]
    x_val, y_val = cifar10_val[0]

    # reshape 3x32x32 into 1x3072
    x_train = x_train.view(-1).unsqueeze(0)
    x_val = x_val.view(-1).unsqueeze(0)

    # reshape int into 1x1 tensor
    y_train = tensor([y_train])
    y_val = tensor([y_val])

    model = nn.Sequential(
        nn.Linear(in_features=n_in, out_features=n_hidden, bias=True),
        nn.Tanh(),
        nn.Linear(in_features=n_hidden, out_features=n_out, bias=True),
        nn.LogSoftmax(dim=1)
    )

    for name, param in model.named_parameters():
        print(f'params for {name} = {param.shape}')

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(10000):

        train_loss = calculate_loss(model, loss_fn, x_train, y_train, is_train=True)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            val_loss = calculate_loss(model, loss_fn, x_val, y_val, is_train=False)
            print(f'epoch = {epoch}  train loss = {train_loss}  val loss = {val_loss}')


if __name__ == '__main__':
    main()
