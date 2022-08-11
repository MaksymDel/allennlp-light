# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
from typing import List

import torch
from tango.common.registrable import Registrable
from torch import FloatTensor, IntTensor, nn


class Image2ImageModule(nn.Module, Registrable):
    """
    An `Image2ImageModule` takes a batch of images as a tensor with the dimensions
    `(batch_size, color_channels, height, width)`, and returns a tensor in the same format,
    after applying some transformation on the images.
    """

    def forward(self, images: FloatTensor, sizes: IntTensor):
        raise NotImplementedError()


@Image2ImageModule.register("normalize")
class NormalizeImage(Image2ImageModule):
    """
    Normalizes an image by subtracting the mean and dividing by the
    standard deviation, separately for each channel.
    """

    def __init__(self, means: List[float], stds: List[float]):
        super().__init__()
        assert len(means) == len(stds)
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

    def forward(self, images: FloatTensor, sizes: IntTensor):
        assert images.size(1) == self.means.size(0)
        self.means = self.means.to(images.device)
        self.stds = self.stds.to(images.device)
        images = images.transpose(1, -1)
        images = images - self.means
        images = images / self.stds
        return images.transpose(-1, 1)
