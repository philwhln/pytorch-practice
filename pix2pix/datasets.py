from pathlib import Path

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, src_width=256, transform_input=None, transform_target=None):
        self.img_paths = sorted(
            [
                str(p)
                for p in Path(root).glob("**/*")
                if p.suffix in [".jpg", ".png"]
            ]
        )
        self.src_width = src_width
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_np = np.array(img)
        input_t = self.transform_input(image=img_np[:, self.src_width:, :])["image"]
        target_t = self.transform_target(image=img_np[:, :self.src_width, :])["image"]
        return input_t, target_t


def load_dataset(dataset_name, src_width=256):
    dataset_dir = Path(__file__).parent.parent / "data" / "kaggle-pix2pix" / dataset_name / dataset_name

    transform_input = A.Compose([
        # A.ColorJitter(0.1),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    transform_target = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return (
        ImageDataset(root=(dataset_dir / "train"), src_width=src_width, transform_input=transform_input,
                     transform_target=transform_target),
        ImageDataset(root=(dataset_dir / "val"), src_width=src_width, transform_input=transform_input,
                     transform_target=transform_target),
    )
