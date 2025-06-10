import random
import torch

class Patchify():
    def __init__(self, patch_size, aligned = False, channel_last = False, all_patches = False):
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) or isinstance(patch_size, list)
        self.aligned = aligned
        self.channel_last = channel_last
        self.all_patches = all_patches

    def __call__(self, x):
        if self.channel_last:
            H, W = x.size(0), x.size(1)
        else:
            H, W = x.size(1), x.size(2)

        if self.all_patches:
            if self.channel_last:
                x_patched = x.unfold(0, self.patch_size[0], self.patch_size[0]).unfold(1, self.patch_size[1], self.patch_size[1]) # shape = (num_patches_H, num_patches_W, C, patch_size_H, patch_size_W)
                return x_patched.permute(0, 1, 3, 4, 2).reshape((-1,) + x_patched.shape[-3:])
            else:
                x_patched = x.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1]) # shape = (C, num_patches_H, num_patches_W, patch_size_H, patch_size_W)
                return x_patched.permute(1, 2, 0, 3, 4).reshape((-1,) + x_patched.shape[-3:])
        else:
            if not self.aligned:
                Hs, Ws = random.randint(0, H - self.patch_size[0]), random.randint(0, W - self.patch_size[1])
                He, We = Hs + self.patch_size[0], Ws + self.patch_size[1]
            else:
                Hs, Ws = random.randint(0, H // self.patch_size[0] - 1) * self.patch_size[0], random.randint(0, W // self.patch_size[1] - 1) * self.patch_size[1]
                He, We = Hs + self.patch_size[0], Ws + self.patch_size[1]

            if self.channel_last:
                return x[Hs:He,Ws:We,:]
            else:
                return x[:,Hs:He,Ws:We]

class PatchifyRotated():
    def __init__(self, patch_size, aligned = False, channel_last = False):
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            assert isinstance(patch_size, tuple) or isinstance(patch_size, list)
        self.aligned = aligned
        assert aligned
        self.channel_last = channel_last

    def __call__(self, x):
        if self.channel_last:
            H, W = x.size(0), x.size(1)
        else:
            H, W = x.size(1), x.size(2)

        assert H // self.patch_size[0] == 2 and W // self.patch_size[1] == 2
        Hs, Ws = random.randint(0, H // self.patch_size[0] - 1) * self.patch_size[0], random.randint(0, W // self.patch_size[1] - 1) * self.patch_size[1]
        He, We = Hs + self.patch_size[0], Ws + self.patch_size[1]

        if self.channel_last:
            patch = x[Hs:He,Ws:We,:]
            axes = [0, 1]
        else:
            patch = x[:,Hs:He,Ws:We]
            axes = [1, 2]
        if Hs == 0 and Ws == 0:
            return patch
        elif Hs == 0 and Ws == self.patch_size[1]:
            return torch.rot90(patch, 1, axes)
        elif Hs == self.patch_size[0] and Ws == self.patch_size[1]:
            return torch.rot90(patch, 2, axes)
        elif Hs == self.patch_size[0] and Ws == 0:
            return torch.rot90(patch, 3, axes)