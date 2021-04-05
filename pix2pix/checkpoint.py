from pathlib import Path

import torch

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


# Code in this file adapted from
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/utils.py

def save(name, model, optimizer):
    print("saving checkpoint")
    path = _checkpoint_path(name)
    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load(name, model, optimizer, lr, device):
    path = _checkpoint_path(name)
    if not path.exists():
        print("no checkpoint found to load. skipping")
        return

    print("loading checkpoint")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _checkpoint_path(name):
    return CHECKPOINT_DIR / f"{name}.pth.tar"
