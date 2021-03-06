import torch
from torch import tensor, optim, nn
import matplotlib.pyplot as plt


def calculate_loss(model, loss_fn, x, y, is_train):
    with torch.set_grad_enabled(is_train):
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
    assert loss.requires_grad == is_train
    return loss


def main():

    learning_rate = 1e-2

    x = tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).unsqueeze(1)
    y = tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]).unsqueeze(1)

    num_examples = x.shape[0]
    shuffled_indices = torch.randperm(num_examples)
    num_val = int(0.2 * num_examples)
    train_indices = shuffled_indices[:-num_val]
    val_indices = shuffled_indices[-num_val:]
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]

    model = nn.Sequential(
        nn.Linear(in_features=1, out_features=2, bias=True),
        nn.Tanh(),
        nn.Linear(in_features=2, out_features=1, bias=True),
    )

    for name, param in model.named_parameters():
        print(f'params for {name} = {param.shape}')

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(50000):

        train_loss = calculate_loss(model, loss_fn, x_train, y_train, is_train=True)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            val_loss = calculate_loss(model, loss_fn, x_val, y_val, is_train=False)
            print(f'epoch = {epoch}  train loss = {train_loss}  val loss = {val_loss}')

    with torch.no_grad():
        line_x = torch.linspace(min(x.squeeze(1)), max(x.squeeze(1)), 100).unsqueeze(1)
        line_y = model(line_x)
        plt.scatter(x_train, y_train)
        plt.scatter(x_val, y_val)
        plt.plot(line_x, line_y)
        plt.show()


if __name__ == '__main__':
    main()
