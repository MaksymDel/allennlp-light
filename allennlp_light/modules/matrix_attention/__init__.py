# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
from allennlp_light.modules.matrix_attention.bilinear_matrix_attention import (
    BilinearMatrixAttention,
)
from allennlp_light.modules.matrix_attention.cosine_matrix_attention import (
    CosineMatrixAttention,
)
from allennlp_light.modules.matrix_attention.dot_product_matrix_attention import (
    DotProductMatrixAttention,
)
from allennlp_light.modules.matrix_attention.linear_matrix_attention import (
    LinearMatrixAttention,
)
from allennlp_light.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp_light.modules.matrix_attention.scaled_dot_product_matrix_attention import (
    ScaledDotProductMatrixAttention,
)
