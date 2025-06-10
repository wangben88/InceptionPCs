import torch


class Quantize():
    def __init__(self, num_levels = 256, input_range = (-1, 1)):
        self.num_levels = num_levels
        self.input_range = input_range

    def __call__(self, x):
        x = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

        return torch.floor(x * self.num_levels).long().clip(0, self.num_levels - 1)