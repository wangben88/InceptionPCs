import os
import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_img_fnames(data_root, name, data_paths = list(), visited = set()):
    if name in visited:
        return

    visited.add(name)

    full_name = os.path.join(data_root, name)
    for ch_name in os.listdir(full_name):
        if os.path.isfile(os.path.join(full_name, ch_name)):
            data_paths.append(os.path.join(name, ch_name))
        elif os.path.isdir(os.path.join(full_name, ch_name)):
            get_img_fnames(data_root, os.path.join(name, ch_name), data_paths, visited)

    print(len(data_paths))

    return


class LSUNBase(Dataset):
    def __init__(self, data_root, train = True, size=None,
                 interpolation="bicubic",
                 flip_p=0.0
                 ):
        if train:
            self.data_root = os.path.join(data_root, "train/")
            self.index_fname = os.path.join(data_root, "train_fnames.txt")
        else:
            self.data_root = os.path.join(data_root, "val/")
            self.index_fname = os.path.join(data_root, "val_fnames.txt")
        
        if os.path.exists(self.index_fname):
            with open(self.index_fname, "r") as f:
                self.data_paths = f.read().splitlines()
        else:
            self.data_paths = list()
            get_img_fnames(self.data_root, name = "./", data_paths = self.data_paths)

            with open(self.index_fname, "w") as f:
                for data_path in self.data_paths:
                    f.write(f"{data_path}\n")

        self.image_paths = self.data_paths

        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        return example


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, root, **kwargs):
        super().__init__(data_root=os.path.append(root, "bedrooms/"), train = True, **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, root, **kwargs):
        super().__init__(data_root=os.path.append(root, "bedrooms/"), train = False, **kwargs)