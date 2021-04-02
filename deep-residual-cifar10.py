from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
from torch import optim, nn

from dataloader_cifar10_animal_bird import prepare_data


def main():
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f'Training on device {device}')

    learning_rate = 3e-3
    batch_size = 512
    epochs = 100

    train_loader, val_loader, class_names = prepare_data(batch_size, device=device)

    in_shape = (32, 32, 3)
    n_res_blocks = 5
    res_conv_channels = 32
    fc_hidden_units = 512

    model = NetResDeep(*in_shape, n_res_blocks, res_conv_channels, fc_hidden_units, len(class_names))
    model = model.to(device)

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


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)

        # init weights randomly based on Kaiming Normalization
        # https://medium.com/@shoray.goel/kaiming-he-initialization-a8d9ed0b5899
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')

        # init so output distributions initially have 0.0 mean and 0.5 variance
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        # output of our layers, plus input (skip connection)
        return out + x


class NetResDeep(nn.Module):
    def __init__(self, input_width: int, input_height: int, input_channels: int, n_res_blocks: int,
                 res_conv_channels: int, fc_hidden_units: int, output_units: int):
        super().__init__()
        self.max_pool_size = 2
        self.conv1 = nn.Conv2d(input_channels, res_conv_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *(n_res_blocks * [ResBlock(n_chans=res_conv_channels)])
        )
        pre_fc1_width = (input_width // self.max_pool_size // self.max_pool_size)
        pre_fc1_height = (input_height // self.max_pool_size // self.max_pool_size)
        pre_fc1_units = pre_fc1_width * pre_fc1_height * res_conv_channels
        self.fc1 = nn.Linear(pre_fc1_units, fc_hidden_units)
        self.fc2 = nn.Linear(fc_hidden_units, output_units)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = F.max_pool2d(out, self.max_pool_size)
        out = self.res_blocks(out)
        out = F.max_pool2d(out, self.max_pool_size)
        out = out.view(-1, self.fc1.in_features)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    main()
