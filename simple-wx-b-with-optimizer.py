import torch
from torch import Tensor, tensor, optim
import matplotlib.pyplot as plt
from pprint import PrettyPrinter


def model(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return (w * x) + b


def loss_fn(y_hat: Tensor, y: Tensor) -> Tensor:
    squared_diffs = (y_hat - y) ** 2
    return squared_diffs.mean()


def main():

    learning_rate = 1e-2

    x = tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    y = tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])

    num_examples = x.shape[0]
    shuffled_indices = torch.randperm(num_examples)
    num_val = int(0.2 * num_examples)
    train_indices = shuffled_indices[:-num_val]
    val_indices = shuffled_indices[-num_val:]
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]

    params = torch.tensor([1.0, 0.0], requires_grad=True)
    optimizer = optim.Adam(params=[params], lr=learning_rate)

    for epoch in range(10000):

        y_hat = model(x_train, *params)
        loss = loss_fn(y_hat, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if epoch % 500 == 0:
                print(f'epoch = {epoch}  params = {params}  params.grad = {params.grad}  loss = {loss}')

    with torch.no_grad():

        y_hat = model(x_val, *params)
        val_loss = loss_fn(y_hat, y_val)
        print(f'val_loss = {val_loss}')

        line_x = tensor([min(x), max(x_train)])
        line_y = model(line_x, *params)
        plt.scatter(x_train, y_train)
        plt.scatter(x_val, y_val)
        plt.plot(line_x, line_y)
        plt.show()


if __name__ == '__main__':
    main()