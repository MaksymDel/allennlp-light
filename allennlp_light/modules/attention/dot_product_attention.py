# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
import torch

from allennlp_light.modules.attention.attention import Attention


@Attention.register("dot_product")
class DotProductAttention(Attention):
    """
    Computes attention between a vector and a matrix using dot product.

    Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "dot_product".
    """

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)
