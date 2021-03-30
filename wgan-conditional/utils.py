import torch


def gradient_penalty(critic, real_imgs, real_labels, fake_imgs, device):
    batch_size, channels, width, height = real_imgs.shape

    # epsilon is the 0. to 1. fraction of the real img to use in interpolation
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, width, height).to(device)

    interpolated_imgs = (epsilon * real_imgs) + ((1 - epsilon) * fake_imgs)

    critic_pred_inter = critic(interpolated_imgs, real_labels)
    gradient = torch.autograd.grad(
        inputs=interpolated_imgs,
        outputs=critic_pred_inter,
        grad_outputs=torch.ones_like(critic_pred_inter),
        create_graph=True,
        retain_graph=True,
    )[0]

    flattened_gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = flattened_gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty
