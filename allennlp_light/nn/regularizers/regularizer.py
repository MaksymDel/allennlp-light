# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
import torch
from tango.common import Registrable


class Regularizer(Registrable):
    """
    An abstract class representing a regularizer. It must implement
    call, returning a scalar tensor.
    """

    default_implementation = "l2"

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
