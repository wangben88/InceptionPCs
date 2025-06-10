import numpy as np
import torch
import os
from torch.utils.data import Dataset
from PIL import Image

from .base import ImagePaths


class CelebA256(Dataset):
    def __init__(self, root=None, train=True, num_samples=None, transform_fns=None):
        super(CelebA256, self).__init__()
        self.root = root
        self.train = train
        self.ns = num_samples

        self.transform_fns = transform_fns

        self._load()

        self.loaded_samples = 0
        self.sshuffle = np.random.permutation(len(self.data))

    def __len__(self):
        l = len(self.data) if self.ns is None else self.ns
        return l

    def __getitem__(self, i):
        if self.ns is not None:
            self.loaded_samples += 1
            if self.loaded_samples >= len(self):
                self.sshuffle = np.random.permutation(len(self.data))
                self.loaded_samples = 0
            return self.data[self.sshuffle[i]]
        else:
            return self.data[i]

    def _load(self):
        if self.train:
            base_dir = os.path.join(self.root, "train256/")
        else:
            base_dir = os.path.join(self.root, "val256/")

        self.abspaths = []
        for fname in os.listdir(base_dir):
            abspath = os.path.join(base_dir, fname)
            if os.path.exists(abspath):
                self.abspaths.append(abspath)

        self.abspaths.sort()

        self.data = ImagePaths(self.abspaths, transform_fns=self.transform_fns)


class CelebA256Train(CelebA256):
    def __init__(self, root, **kwargs):
        super().__init__(root, train = True, **kwargs)


class CelebA256Validation(CelebA256):
    def __init__(self, root, **kwargs):
        super().__init__(root, train = False, **kwargs)