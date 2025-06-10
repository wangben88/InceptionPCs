import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
import collections
import sys
import os

from data.image.utils import instantiate_from_config

from typing import Dict, Tuple, Optional, NamedTuple, Union
from PIL.Image import Image as pil_image
from torch import Tensor
from torchvision.transforms import v2
from torch.utils.data import RandomSampler

from cookbook.data.datasets import NumpyLoader

try:
  from typing import Literal
except ImportError:
  from typing_extensions import Literal


Image = Union[Tensor, pil_image]
BoundingBox = Tuple[float, float, float, float]  # x0, y0, w, h
CropMethodType = Literal['none', 'random', 'center', 'random-2d']
SplitType = Literal['train', 'validation', 'test']


class ImageDescription(NamedTuple):
    id: int
    file_name: str
    original_size: Tuple[int, int]  # w, h
    url: Optional[str] = None
    license: Optional[int] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None
    flickr_url: Optional[str] = None
    flickr_id: Optional[str] = None
    coco_id: Optional[str] = None


class Category(NamedTuple):
    id: str
    super_category: Optional[str]
    name: str


class Annotation(NamedTuple):
    area: float
    image_id: str
    bbox: BoundingBox
    category_no: int
    category_id: str
    id: Optional[int] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    is_group_of: Optional[bool] = None
    is_truncated: Optional[bool] = None
    is_occluded: Optional[bool] = None
    is_depiction: Optional[bool] = None
    is_inside: Optional[bool] = None
    segmentation: Optional[Dict] = None


def custom_collate(batch, batched_item = False):
    r"""source: pytorch 1.9.0, only one modification to original code. NB: added additional modification for batched items """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        # out = None
        # if torch.utils.data.get_worker_info() is not None:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        #     numel = sum([x.numel() for x in batch])
        #     storage = elem.storage()._new_shared(numel)
        #     out = elem.new(storage)
        # return torch.stack(batch, 0, out=out)
        if batched_item:
            return torch.cat(batch, dim = 0)
        return torch.stack(batch, dim = 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    if isinstance(elem, collections.abc.Sequence) and isinstance(elem[0], Annotation):  # added
        return batch  # added
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, drop_last=False, shuffle_train=True, 
                 shuffle_validation=False, transform_fns=None, sampler=None, pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train
        self.shuffle_validation = shuffle_validation

        self.pin_memory = pin_memory

        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = None

        if transform_fns is not None:
            raise NotImplementedError("Transforms now must be included in the train/val/test specific params.")

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    # Modified to NumpyLoader
    def _train_dataloader(self, sampler = None):
        if sampler is None and self.sampler is not None:
            sampler = self.sampler

        if isinstance(sampler, str) and sampler.startswith("__subset__:"):
            sampler = RandomSampler(self.datasets["train"], False, int(sampler.split(":")[1]))

        if self.dataset_configs["train"].get("batched_item", False):
            collate_fn = lambda x: custom_collate(x, batched_item = True)
            prebatch_size = next(iter(self.datasets["train"])).shape[0]
            batch_size = self.batch_size // prebatch_size
        else:
            collate_fn = custom_collate
            batch_size = self.batch_size
        if sampler is None:
            return NumpyLoader(self.datasets["train"], batch_size=batch_size,
                            num_workers=self.num_workers, shuffle=self.shuffle_train, 
                            collate_fn=collate_fn, persistent_workers = (self.num_workers > 0),
                            drop_last=self.drop_last, pin_memory = self.pin_memory)
        else:
            return NumpyLoader(self.datasets["train"], batch_size=batch_size,
                            num_workers=self.num_workers, sampler = sampler, 
                            collate_fn=collate_fn, persistent_workers = (self.num_workers > 0),
                            drop_last=self.drop_last, pin_memory = self.pin_memory)


    def _val_dataloader(self):
        if self.dataset_configs["validation"].get("batched_item", False):
            collate_fn = lambda x: custom_collate(x, batched_item = True)
            prebatch_size = next(iter(self.datasets["validation"])).shape[0]
            batch_size = self.batch_size // prebatch_size
        else:
            collate_fn = custom_collate
            batch_size = self.batch_size
        return NumpyLoader(self.datasets["validation"],
                          batch_size=batch_size,
                          num_workers=0, collate_fn=collate_fn,
                          drop_last=self.drop_last, shuffle=self.shuffle_validation,
                          pin_memory = self.pin_memory)

    def _test_dataloader(self):
        if self.dataset_configs["test"].get("batched_item", False):
            collate_fn = lambda x: custom_collate(x, batched_item = True)
            prebatch_size = next(iter(self.datasets["test"])).shape[0]
            batch_size = self.batch_size // prebatch_size
        else:
            collate_fn = custom_collate
            batch_size = self.batch_size
        return NumpyLoader(self.datasets["test"], batch_size=batch_size,
                          num_workers=self.num_workers, collate_fn=collate_fn,
                          drop_last=self.drop_last, shuffle=False, 
                          persistent_workers = (self.num_workers > 0),
                          pin_memory = self.pin_memory)