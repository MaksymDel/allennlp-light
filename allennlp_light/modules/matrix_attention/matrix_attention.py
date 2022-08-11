# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
import torch
from tango.common.registrable import Registrable


class MatrixAttention(torch.nn.Module, Registrable):
    """
    `MatrixAttention` takes two matrices as input and returns a matrix of attentions.

    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores. Because these scores are unnormalized, we don't take a mask as input; it's up to the
    caller to deal with masking properly when this output is used.

    Input:
        - matrix_1 : `(batch_size, num_rows_1, embedding_dim_1)`
        - matrix_2 : `(batch_size, num_rows_2, embedding_dim_2)`

    Output:
        - `(batch_size, num_rows_1, num_rows_2)`
    """

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
