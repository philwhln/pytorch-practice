import torch
import torch.nn.functional as F
from torch import optim, nn

from dataloader_cifar10_animal_bird import prepare_data


def main():
    learning_rate = 1e-2
    batch_size = 64
    epochs = 100

    train_loader, val_loader, class_names = prepare_data(batch_size)

    in_shape = (32, 32, 3)
    n_hidden = 512
    n_out = len(class_names)

    model = Net(*in_shape, n_hidden, n_out)

    total_trainable_params = 0
    for name, params in model.named_parameters():
        num_params = params.numel()
        trainable_params = 0
        if params.requires_grad:
            total_trainable_params += num_params
            trainable_params = num_params
        print(f'{name}: {params.shape} params={num_params} trainable={trainable_params})')
    print(f'total trainable params : {total_trainable_params}')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        # train one epoch
        train_total = 0
        train_correct = 0
        train_loss_accum = 0.
        for imgs, label_indices in train_loader:
            num_imgs = imgs.shape[0]

            output = model(imgs)
            loss = loss_fn(output, label_indices)
            train_loss_accum += loss.item()

            out_scores, out_indices = torch.max(output, dim=-1)
            train_total += num_imgs
            train_correct += int((out_indices == label_indices).sum())

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # occasionally check validation set performance
        if epoch % 5 == 0:
            val_total = 0
            val_correct = 0
            val_loss_accum = 0.
            for imgs, label_indices in val_loader:
                num_imgs = imgs.shape[0]

                with torch.set_grad_enabled(False):
                    output = model(imgs)
                    val_loss_accum += loss_fn(output, label_indices).item()

                out_scores, out_indices = torch.max(output, dim=-1)
                val_total += num_imgs
                val_correct += int((out_indices == label_indices).sum())

            train_loss = train_loss_accum / len(train_loader)
            val_loss = val_loss_accum / len(val_loader)
            print(f'epoch = {epoch}  train loss = {train_loss:0.6f}  train accuracy = {train_correct / train_total}  '
                  f'val loss = {val_loss:0.6f}  val accuracy = {val_correct / val_total}')


class Net(nn.Module):
    def __init__(self, input_width: int, input_height: int, input_channels: int, hidden_units: int, output_units: int):
        super().__init__()
        conv1_out_channels = 16
        conv2_out_channels = 8
        self.conv1 = nn.Conv2d(input_channels, conv1_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(int(input_width / 2 / 2) * int(input_height / 2 / 2) * conv2_out_channels, hidden_units)
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
