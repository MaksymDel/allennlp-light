# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
import torch

from allennlp_light.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp_light.nn import util


@MatrixAttention.register("cosine")
class CosineMatrixAttention(MatrixAttention):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using cosine
    similarity.

    Registered as a `MatrixAttention` with name "cosine".
    """

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        a_norm = matrix_1 / (
            matrix_1.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix_1.dtype)
        )
        b_norm = matrix_2 / (
            matrix_2.norm(p=2, dim=-1, keepdim=True) + util.tiny_value_of_dtype(matrix_2.dtype)
        )
        return torch.bmm(a_norm, b_norm.transpose(-1, -2))
