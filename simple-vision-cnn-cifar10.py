from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
from torch import optim, nn

from dataloader_cifar10_animal_bird import prepare_data


def main():
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Training on device {device}')

    learning_rate = 1e-2
    batch_size = 64
    epochs = 2

    train_loader, val_loader, class_names = prepare_data(batch_size, device=device)

    in_shape = (32, 32, 3)
    n_hidden = 512
    n_out = len(class_names)

    model = Net(*in_shape, n_hidden, n_out).to(device)

    describe_params(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)

    train(model, loss_fn, optimizer, train_loader, val_loader, device, epochs)

    output_path = Path(__file__).parent / "parameters" / Path(__file__).with_suffix('.pt').name

    torch.save(model.state_dict(), output_path)


def describe_params(model):
    total_trainable_params = 0
    for name, params in model.named_parameters():
        num_params = params.numel()
        trainable_params = 0
        if params.requires_grad:
            total_trainable_params += num_params
            trainable_params = num_params
        print(f'{name}: {params.shape} params={num_params} trainable={trainable_params})')
    print(f'total trainable params : {total_trainable_params}')


def train(model, loss_fn, optimizer, train_loader, val_loader, device, epochs):

    for epoch in range(epochs):

        start_time = time()
        train_accuracy, train_loss = one_epoch(model, loss_fn, device, train_loader, optimizer)
        train_time = time() - start_time

        # occasionally check validation set performance
        if epoch % 5 == 0:
            start_time = time()
            val_accuracy, val_loss = one_epoch(model, loss_fn, device, val_loader)
            val_time = time() - start_time
            print(f'epoch = {epoch}  train loss = {train_loss:0.6f}  train accuracy = {train_accuracy}  '
                  f'val loss = {val_loss:0.6f}  val accuracy = {val_accuracy}  '
                  f'train time = {train_time:0.2f}  val time = {val_time:0.2f}')


def one_epoch(model, loss_fn, device, data_loader, optimizer=None):

    update_parameters = (optimizer is not None)

    total = 0
    correct = 0
    loss_accum = 0.

    for imgs, label_indices in data_loader:
        imgs = imgs.to(device=device)
        label_indices = label_indices.to(device=device)

        num_imgs = imgs.shape[0]

        with torch.set_grad_enabled(update_parameters):
            output = model(imgs)
            loss = loss_fn(output, label_indices)

        loss_accum += loss.item()

        out_scores, out_indices = torch.max(output, dim=-1)
        total += num_imgs
        correct += int((out_indices == label_indices).sum())

        if update_parameters:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_loss = loss_accum / len(data_loader)
    accuracy = correct / total

    return accuracy, avg_loss


def validation(model, loss_fn, data_loader, device):
    total = 0
    correct = 0
    loss_accum = 0.

    for imgs, label_indices in data_loader:
        imgs = imgs.to(device=device)
        label_indices = label_indices.to(device=device)

        num_imgs = imgs.shape[0]

        with torch.set_grad_enabled(False):
            output = model(imgs)
            loss = loss_fn(output, label_indices).item()

        loss_accum += loss.item()

        out_scores, out_indices = torch.max(output, dim=-1)
        total += num_imgs
        correct += int((out_indices == label_indices).sum())

    avg_loss = loss_accum / len(data_loader)
    accuracy = correct / total

    return accuracy, avg_loss


class Net(nn.Module):
    def __init__(self, input_width: int, input_height: int, input_channels: int, hidden_units: int, output_units: int):
        super().__init__()
        conv1_out_channels = 16
        conv2_out_channels = 8
        self.conv1 = nn.Conv2d(input_channels, conv1_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((input_width // 2 // 2) * (input_height // 2 // 2) * conv2_out_channels, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.tanh(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = torch.tanh(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, self.fc1.in_features)
        out = self.fc1(out)
        out = torch.tanh(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    main()
