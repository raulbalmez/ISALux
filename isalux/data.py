"""Paired low/normal-light datasets: LOL-v1, LOL-v2-real, LOL-v2-synthetic.

Expected layout (<root> is the dataset folder itself):
  LOL-v1:   <root>/our485/{low,high}      <root>/eval15/{low,high}
  LOL-v2:   <root>/{Train,Test}/{Input,GT}   (or the original
            <root>/{Real_captured,Synthetic}/{Train,Test}/{Low,Normal})
"""
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_DIRS = {
    "lolv1": {"train": ["our485/low", "lol_dataset/our485/low"],
              "test": ["eval15/low", "lol_dataset/eval15/low"]},
    "lolv2_real": {"train": ["Train/Input", "Real_captured/Train/Low"],
                   "test": ["Test/Input", "Real_captured/Test/Low"]},
    "lolv2_syn": {"train": ["Train/Input", "Synthetic/Train/Low"],
                  "test": ["Test/Input", "Synthetic/Test/Low"]},
}
_HIGH = {"low": "high", "Input": "GT", "Low": "Normal"}
_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def _key(f):  # low00001 / normal00001 / r00816405t / 00816405t -> numeric key
    return re.sub(r"^[A-Za-z]+", "", os.path.splitext(f)[0])


def list_pairs(root, dataset, split):
    for low_sub in _DIRS[dataset][split]:
        low_dir = os.path.join(root, low_sub)
        if os.path.isdir(low_dir):
            head, tail = os.path.split(low_sub)
            high_dir = os.path.join(root, head, _HIGH[tail])
            break
    else:
        raise FileNotFoundError(f"{dataset}/{split} not found under {root}")
    low = {_key(f): f for f in os.listdir(low_dir) if f.lower().endswith(_EXT)}
    high = {_key(f): f for f in os.listdir(high_dir) if f.lower().endswith(_EXT)}
    keys = sorted(set(low) & set(high))
    assert keys and len(keys) == len(low) == len(high), "low/high mismatch"
    return ([os.path.join(low_dir, low[k]) for k in keys],
            [os.path.join(high_dir, high[k]) for k in keys])


def _load(path):
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


class PairedDataset(Dataset):
    """Train: random crop + random flips + random 90-degree rotations.
    Test: full images."""

    def __init__(self, root, dataset, split, patch_size=256, length=None):
        self.low, self.high = list_pairs(root, dataset, split)
        self.train = split == "train"
        self.patch = patch_size
        self.length = length  # virtual length for iteration-based training

    def __len__(self):
        return self.length or len(self.low)

    def __getitem__(self, i):
        i = i % len(self.low)
        low, high = _load(self.low[i]), _load(self.high[i])
        if not self.train:
            return low, high
        _, h, w = low.shape
        y, x = random.randint(0, h - self.patch), random.randint(0, w - self.patch)
        low = low[:, y:y + self.patch, x:x + self.patch]
        high = high[:, y:y + self.patch, x:x + self.patch]
        if random.random() < 0.5:
            low, high = low.flip(-1), high.flip(-1)
        if random.random() < 0.5:
            low, high = low.flip(-2), high.flip(-2)
        k = random.randint(0, 3)
        if k:
            low, high = torch.rot90(low, k, (1, 2)), torch.rot90(high, k, (1, 2))
        return low.contiguous(), high.contiguous()
