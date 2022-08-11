# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
import math

import torch

from allennlp_light.modules.matrix_attention.dot_product_matrix_attention import (
    DotProductMatrixAttention,
)
from allennlp_light.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("scaled_dot_product")
class ScaledDotProductMatrixAttention(DotProductMatrixAttention):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using a dot
    product. Scales the result by the size of the embeddings.

    Registered as a `MatrixAttention` with name "scaled_dot_product".
    """

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        return super().forward(matrix_1, matrix_2) / math.sqrt(matrix_1.size(-1))
