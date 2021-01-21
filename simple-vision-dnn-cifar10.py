import torch
from torch import optim, nn

from dataloader_cifar10_animal_bird import prepare_data


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
