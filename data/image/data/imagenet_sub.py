import numpy as np
import torch
import os
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import v2

from data.image.utils import instantiate_from_config


class ImageNetSub(Dataset):
    def __init__(self, img_size = 32, root_dir = "/scratch/anji/data/ImageNet", train = True, transform_fns = None):
        self.train = train
        if self.train:
            root_dir = os.path.join(root_dir, "train/")
        else:
            root_dir = os.path.join(root_dir, "val/")

        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)
        self.datasets = []
        self.labels = []
        self.img2dataset = dict()
        self.img2idx = dict()
        sample_idx = 0
        for file in self.files:
            print("> Loading {}".format(file))
            fname = os.path.join(self.root_dir, file)
            data = np.load(fname)
            self.datasets.append(data["data"].reshape(-1, 3, img_size, img_size))
            self.labels.append(data["labels"])
            for i in range(self.datasets[-1].shape[0]):
                self.img2dataset[sample_idx] = len(self.datasets) - 1
                self.img2idx[sample_idx] = i
                sample_idx += 1
                
        self.length = sample_idx

        if transform_fns is not None:
            transforms = []
            for transform_fn in transform_fns:
                transforms.append(instantiate_from_config(transform_fn))
            self.transforms = v2.Compose(transforms)
        else:
            self.transforms = lambda x: x

    def __getitem__(self, index):
        img = self.datasets[self.img2dataset[index]][self.img2idx[index]]
        label = self.labels[self.img2dataset[index]][self.img2idx[index]]

        img = torch.from_numpy(img).type(torch.uint8).float() / 127.5 - 1 # Normalize to [-1, 1]
        sample = {"img": img, "label": label}

        return self.transforms(sample)

    def __len__(self):
        return self.length

class ImageNet64Train(ImageNetSub):
    def __init__(self, root = "/scratch/anji/data/ImageNet", transform_fns = None):
        root = os.path.join(root, "imagenet64/")

        super(ImageNet64Train, self).__init__(img_size = 64, root_dir = root, train = True, transform_fns = transform_fns)


class ImageNet64Validation(ImageNetSub):
    def __init__(self, root = "/scratch/anji/data/ImageNet", transform_fns = None):
        root = os.path.join(root, "imagenet64/")

        super(ImageNet64Validation, self).__init__(img_size = 64, root_dir = root, train = False, transform_fns = transform_fns)


class ImageNet32Train(ImageNetSub):
    def __init__(self, root = "/scratch/anji/data/ImageNet", transform_fns = None):
        root = os.path.join(root, "imagenet32/")

        super(ImageNet32Train, self).__init__(img_size = 32, root_dir = root, train = True, transform_fns = transform_fns)


class ImageNet32Validation(ImageNetSub):
    def __init__(self, root = "/scratch/anji/data/ImageNet", transform_fns = None):
        root = os.path.join(root, "imagenet32/")

        super(ImageNet32Validation, self).__init__(img_size = 32, root_dir = root, train = False, transform_fns = transform_fns)


class ImageNet16Train(ImageNetSub):
    def __init__(self, root = "/scratch/anji/data/ImageNet", transform_fns = None):
        root = os.path.join(root, "imagenet16/")

        super(ImageNet16Train, self).__init__(img_size = 16, root_dir = root, train = True, transform_fns = transform_fns)


class ImageNet16Validation(ImageNetSub):
    def __init__(self, root = "/scratch/anji/data/ImageNet", transform_fns = None):
        root = os.path.join(root, "imagenet16/")

        super(ImageNet16Validation, self).__init__(img_size = 16, root_dir = root, train = False, transform_fns = transform_fns)