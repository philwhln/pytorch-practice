from pathlib import Path

import torch
from torch.utils import data
from torchvision.utils import save_image

from datasets import load_dataset
from model.generator import Generator
import checkpoint

BATCH_SIZE = 10
CHECKPOINT_NAME_GENERATOR = "pix2pix.generator"
DATASET_NAME = "facades"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "pix2pix" / DATASET_NAME / "val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset(DATASET_NAME, "val")
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE)

generator = Generator().to(device)
checkpoint.load(CHECKPOINT_NAME_GENERATOR, generator, None, None, device)
generator.eval()

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

for batch_idx, (idxs, x, y) in enumerate(dataloader):
    print(f"batch: {batch_idx}")
    y_fake = generator(x.to(device))

    for idx, x_t, y_t, y_fake_t in zip(idxs, x, y, y_fake):
        src_filename = dataset.src_filename(idx)
        dst_filename = OUTPUT_DIR / Path(src_filename).name
        imgs = torch.cat([x_t, y_t, y_fake_t])
        print(f"saving {dst_filename}")
        save_image([x_t, y_t, y_fake_t], str(dst_filename), normalize=True)
