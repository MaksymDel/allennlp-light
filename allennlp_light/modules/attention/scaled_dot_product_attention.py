# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
import math
from typing import Optional

import torch

from allennlp_light.modules.attention.attention import Attention
from allennlp_light.modules.attention.dot_product_attention import DotProductAttention


@Attention.register("scaled_dot_product")
class ScaledDotProductAttention(DotProductAttention):
    """
    Computes attention between two tensors using scaled dot product.
    # Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    # (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "scaled_dot_product".

    # Parameters

    scaling_factor : `int`, required
        The similarity score is scaled down by the `scaling_factor`.
    normalize : `bool`, optional (default=`True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, scaling_factor: Optional[int] = None, normalize: bool = True) -> None:
        super().__init__(normalize)
        self.scaling_factor = scaling_factor

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        scores = super()._forward_internal(vector, matrix)
        scaling_factor = self.scaling_factor or matrix.size(-1)
        scores = scores / math.sqrt(scaling_factor)
        return scores
