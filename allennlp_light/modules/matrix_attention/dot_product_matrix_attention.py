# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
import torch

from allennlp_light.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("dot_product")
class DotProductMatrixAttention(MatrixAttention):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using a dot
    product.

    Registered as a `MatrixAttention` with name "dot_product".
    """

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        return matrix_1.matmul(matrix_2.transpose(-1, -2))
