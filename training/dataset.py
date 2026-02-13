"""
Dataset and dataloader utilities for training.

Provides ImageMaskDataset for loading feature images and class mask pairs.
"""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from musac_las_classifier.constants import CLASS_COLORS


def rgb_to_class(mask_array, color_to_class):
    """Convert an RGB mask to a class index mask.

    Args:
        mask_array: (H, W, 3) numpy array with RGB values.
        color_to_class: Dict mapping (R, G, B) tuple to class index.

    Returns:
        (H, W) numpy array with class indices.
    """
    h, w, _ = mask_array.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for color, class_idx in color_to_class.items():
        matches = np.all(mask_array == color, axis=-1)
        class_mask[matches] = class_idx
    return class_mask


class ImageMaskDataset(Dataset):
    """Dataset for paired feature images and class mask images.

    Args:
        image_dir: Directory with input feature PNG images.
        mask_dir: Directory with ground-truth class mask PNG images.
        class_colors: Dict mapping class_id -> [R, G, B].
        transform: Optional torchvision transform for input images.
    """

    def __init__(self, image_dir, mask_dir, class_colors=None, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        if class_colors is None:
            class_colors = CLASS_COLORS
        self.color_to_class = {tuple(v): k for k, v in class_colors.items()}

        self.image_filenames = sorted(self.image_dir.glob("*.png"))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        mask_np = np.array(mask)
        mask_class = rgb_to_class(mask_np, self.color_to_class)

        if self.transform:
            image = self.transform(image)

        mask_tensor = torch.from_numpy(mask_class).long()
        return image, mask_tensor


def get_dataloaders(image_dir, mask_dir, class_colors=None,
                    train_ratio=0.8, batch_size=8, seed=42):
    """Create train and validation dataloaders.

    Args:
        image_dir: Directory with feature images.
        mask_dir: Directory with class mask images.
        class_colors: Class color mapping.
        train_ratio: Fraction of data for training.
        batch_size: Batch size.
        seed: Random seed for deterministic train/val split.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    transform = T.Compose([T.ToTensor()])
    dataset = ImageMaskDataset(image_dir, mask_dir, class_colors, transform)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
