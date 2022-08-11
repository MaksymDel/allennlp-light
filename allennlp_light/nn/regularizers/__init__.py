# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
"""
This module contains classes representing regularization schemes
as well as a class for applying regularization to parameters.
"""

from allennlp_light.nn.regularizers.regularizer import Regularizer
from allennlp_light.nn.regularizers.regularizer_applicator import RegularizerApplicator
from allennlp_light.nn.regularizers.regularizers import L1Regularizer, L2Regularizer
