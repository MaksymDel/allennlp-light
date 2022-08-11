# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
"""
Custom PyTorch
`Module <https://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ s
that are used as components in AllenNLP `Model` s.
"""

from allennlp_light.modules.attention import Attention
from allennlp_light.modules.bimpm_matching import BiMpmMatching
from allennlp_light.modules.conditional_random_field import ConditionalRandomField
from allennlp_light.modules.feedforward import FeedForward
from allennlp_light.modules.gated_sum import GatedSum
from allennlp_light.modules.highway import Highway
from allennlp_light.modules.input_variational_dropout import InputVariationalDropout
from allennlp_light.modules.layer_norm import LayerNorm
from allennlp_light.modules.matrix_attention import MatrixAttention
from allennlp_light.modules.maxout import Maxout
from allennlp_light.modules.residual_with_layer_dropout import ResidualWithLayerDropout
from allennlp_light.modules.scalar_mix import ScalarMix
from allennlp_light.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp_light.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp_light.modules.softmax_loss import SoftmaxLoss
from allennlp_light.modules.span_extractors import SpanExtractor
from allennlp_light.modules.time_distributed import TimeDistributed
