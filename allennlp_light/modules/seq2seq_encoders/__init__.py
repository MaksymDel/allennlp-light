# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
"""
Modules that transform a sequence of input vectors
into a sequence of output vectors.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.

The available Seq2Seq encoders are

- `"gru"` : allennlp_light.modules.seq2seq_encoders.GruSeq2SeqEncoder
- `"lstm"` : allennlp_light.modules.seq2seq_encoders.LstmSeq2SeqEncoder
- `"rnn"` : allennlp_light.modules.seq2seq_encoders.RnnSeq2SeqEncoder
- `"augmented_lstm"` : allennlp_light.modules.seq2seq_encoders.AugmentedLstmSeq2SeqEncoder
- `"alternating_lstm"` : allennlp_light.modules.seq2seq_encoders.StackedAlternatingLstmSeq2SeqEncoder
- `"pass_through"` : allennlp_light.modules.seq2seq_encoders.PassThroughEncoder
- `"feedforward"` : allennlp_light.modules.seq2seq_encoders.FeedForwardEncoder
- `"pytorch_transformer"` : allennlp_light.modules.seq2seq_encoders.PytorchTransformer
- `"compose"` : allennlp_light.modules.seq2seq_encoders.ComposeEncoder
- `"gated-cnn-encoder"` : allennlp_light.momdules.seq2seq_encoders.GatedCnnEncoder
- `"stacked_bidirectional_lstm"`: allennlp_light.modules.seq2seq_encoders.StackedBidirectionalLstmSeq2SeqEncoder
"""

from allennlp_light.modules.seq2seq_encoders.compose_encoder import ComposeEncoder
from allennlp_light.modules.seq2seq_encoders.feedforward_encoder import (
    FeedForwardEncoder,
)
from allennlp_light.modules.seq2seq_encoders.gated_cnn_encoder import GatedCnnEncoder
from allennlp_light.modules.seq2seq_encoders.pass_through_encoder import (
    PassThroughEncoder,
)
from allennlp_light.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import (
    AugmentedLstmSeq2SeqEncoder,
    GruSeq2SeqEncoder,
    LstmSeq2SeqEncoder,
    PytorchSeq2SeqWrapper,
    RnnSeq2SeqEncoder,
    StackedAlternatingLstmSeq2SeqEncoder,
    StackedBidirectionalLstmSeq2SeqEncoder,
)
from allennlp_light.modules.seq2seq_encoders.pytorch_transformer_wrapper import (
    PytorchTransformer,
)
from allennlp_light.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
