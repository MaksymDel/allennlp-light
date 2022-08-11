# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
import torch

from allennlp_light.modules.feedforward import FeedForward
from allennlp_light.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("feedforward")
class FeedForwardEncoder(Seq2SeqEncoder):
    """
    This class applies the `FeedForward` to each item in sequences.

    Registered as a `Seq2SeqEncoder` with name "feedforward".
    """

    def __init__(self, feedforward: FeedForward) -> None:
        super().__init__()
        self._feedforward = feedforward

    def get_input_dim(self) -> int:
        return self._feedforward.get_input_dim()

    def get_output_dim(self) -> int:
        return self._feedforward.get_output_dim()

    def is_bidirectional(self) -> bool:
        return False

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        # Parameters

        inputs : `torch.Tensor`, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, timesteps).

        # Returns

        A tensor of shape (batch_size, timesteps, output_dim).
        """
        if mask is None:
            return self._feedforward(inputs)
        else:
            outputs = self._feedforward(inputs)
            return outputs * mask.unsqueeze(dim=-1)
