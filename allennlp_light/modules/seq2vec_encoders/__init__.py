# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
"""
Modules that transform a sequence of input vectors
into a single output vector.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.

The available Seq2Vec encoders are

* `"gru"` https://pytorch.org/docs/master/nn.html#torch.nn.GRU
* `"lstm"` https://pytorch.org/docs/master/nn.html#torch.nn.LSTM
* `"rnn"` https://pytorch.org/docs/master/nn.html#torch.nn.RNN
* `"cnn"` allennlp_light.modules.seq2vec_encoders.cnn_encoder.CnnEncoder
* `"augmented_lstm"` allennlp_light.modules.augmented_lstm.AugmentedLstm
* `"alternating_lstm"` allennlp_light.modules.stacked_alternating_lstm.StackedAlternatingLstm
* `"stacked_bidirectional_lstm"` allennlp_light.modules.stacked_bidirectional_lstm.StackedBidirectionalLstm
"""

from allennlp_light.modules.seq2vec_encoders.bert_pooler import BertPooler
from allennlp_light.modules.seq2vec_encoders.boe_encoder import BagOfEmbeddingsEncoder
from allennlp_light.modules.seq2vec_encoders.cls_pooler import ClsPooler
from allennlp_light.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp_light.modules.seq2vec_encoders.cnn_highway_encoder import (
    CnnHighwayEncoder,
)
from allennlp_light.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import (
    AugmentedLstmSeq2VecEncoder,
    GruSeq2VecEncoder,
    LstmSeq2VecEncoder,
    PytorchSeq2VecWrapper,
    RnnSeq2VecEncoder,
    StackedAlternatingLstmSeq2VecEncoder,
    StackedBidirectionalLstmSeq2VecEncoder,
)
from allennlp_light.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
