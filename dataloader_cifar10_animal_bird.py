from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cache_fn_decoractor import cache


@cache('simple-vision-cifar10')
def prepare_data(batch_size: int, device=None) -> DataLoader:
    data_path = Path(__file__).parent / 'data' / 'cifar-10'

    cifar10_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())

    imgs_train = torch.stack([img_t for img_t, _ in cifar10_train], dim=3)
    if device:
        imgs_train = imgs_train.to(device)

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
    cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform=cifar10_transforms)

    class_names = ['airplane', 'bird']

    # select just to images for labels [0, 2] and remap labels to [0, 1]
    label_remap = {0: 0, 2: 1}
    cifar10_train = [(img, label_remap[label]) for img, label in cifar10_train if label in label_remap.keys()]
    cifar10_val = [(img, label_remap[label]) for img, label in cifar10_val if label in label_remap.keys()]

    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, class_names
