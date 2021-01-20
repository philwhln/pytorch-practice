import torch
from torch import Tensor, tensor
import matplotlib.pyplot as plt


def model(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return (w * x) + b


def loss_fn(y_hat: Tensor, y: Tensor) -> Tensor:
    squared_diffs = (y_hat - y) ** 2
    return squared_diffs.mean()


def main():

    learning_rate = 1e-2

    x = tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    y = tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])

    x_norm = x * 0.1

    params = torch.tensor([1.0, 0.0], requires_grad=True)

    for epoch in range(10000):

        if params.grad is not None:
            params.grad.zero_()

        y_hat = model(x_norm, *params)
        loss = loss_fn(y_hat, y)
        loss.backward()

        with torch.no_grad():
            if epoch % 500 == 0:
                print(f'epoch = {epoch}  params = {params}  params.grad = {params.grad}  loss = {loss}')

            params -= learning_rate * params.grad


    with torch.no_grad():
        line_x = tensor([min(x_norm), max(x_norm)])
        line_y = model(line_x, *params)
        plt.scatter(x_norm, y)
        plt.plot(line_x, line_y)
        plt.show()


if __name__ == '__main__':
    main()