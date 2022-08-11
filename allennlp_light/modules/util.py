# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import torch


def replicate_layers(layer: torch.nn.Module, num_copies: int):
    """
    # Parameters
            layer (torch.nn.Module) - The torch layer that needs to be replicated.
            num_copies (int) - Number of copies to create.

    # Returns
            A ModuleList that contains `num_copies` of the `layer`.
    """
    return torch.nn.ModuleList([deepcopy(layer) for _ in range(num_copies)])
