from pathlib import Path

import torch
from torch import optim, nn, tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cache_fn_decoractor import cache


@cache('simple-vision-cifar10')
def prepare_data(batch_size):
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

    train_loader = DataLoader(cifar10_train, batch_size=batch_size)
    val_loader = DataLoader(cifar10_val, batch_size=batch_size)

    return train_loader, val_loader, class_names


def calculate_loss(model, loss_fn, x, y, is_train):
    with torch.set_grad_enabled(is_train):
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
    assert loss.requires_grad == is_train
    return loss


def main():
    learning_rate = 1e-4
    batch_size = 64
    epochs = 100

    train_loader, val_loader, class_names = prepare_data(batch_size)

    n_in = 3 * 32 * 32
    n_hidden = 512
    n_out = len(class_names)

    model = nn.Sequential(
        nn.Linear(in_features=n_in, out_features=n_hidden, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=n_hidden, out_features=n_out, bias=True),
    )

    for name, param in model.named_parameters():
        print(f'params for {name} = {param.shape}')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        # train one epoch
        train_total = 0
        train_correct = 0
        for imgs, label_indices in train_loader:
            num_imgs = imgs.shape[0]
            imgs_1d = imgs.view(num_imgs, -1)

            output = model(imgs_1d)
            train_loss = loss_fn(output, label_indices)

            out_scores, out_indices = torch.max(output, dim=-1)
            train_total += num_imgs
            train_correct += int((out_indices == label_indices).sum())

            # update parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # occasionally check validation set performance
        if epoch % 5 == 0:
            val_total = 0
            val_correct = 0
            for imgs, label_indices in val_loader:
                num_imgs = imgs.shape[0]
                imgs_1d = imgs.view(num_imgs, -1)

                with torch.set_grad_enabled(False):
                    output = model(imgs_1d)
                    val_loss = loss_fn(output, label_indices)

                out_scores, out_indices = torch.max(output, dim=-1)
                val_total += num_imgs
                val_correct += int((out_indices == label_indices).sum())

            print(f'epoch = {epoch}  train loss = {train_loss:0.6f}  train accuracy = {train_correct / train_total}  '
                  f'val loss = {val_loss:0.6f}  val accuracy = {val_correct / val_total}')


if __name__ == '__main__':
    main()
