# allennlp-light

## About 

As [AllenNLP framework](https://github.com/allenai/allennlp) honorably retires and will not update dependencies, *allennlp-light* is a port of AllenNLP's awesome `modules` and `nn` portions into a standalone package with minimum dependencies.\
*allennlp-light* natively integrates with [Tango](https://github.com/allenai/tango) (check it out!) by using its `FromParams/Registrable` so you get allennlp's components for free, registered, and ready to use. \

The modules are thoroughly [documented](https://docs.allennlp.org/main/) [and](https://github.com/allenai/allennlp/tree/main/tests/nn) [tested](https://github.com/allenai/allennlp/tree/main/tests/modules) in the original [AllenNLP repository](https://github.com/allenai/allennlp).

To learn how to use them, check the relevan section in the [AllenNLP guide](https://guide.allennlp.org/common-architectures).

AllenNLP is licensed under Apache 2 Licence, so please see below the *copyright* notice and the *list of changes*.

## Installation

1) Install PyTorch: [pytorch.org](https://pytorch.org/)
2) `pip install allennlp-light`

## Example
    
```python
>>> from allennlp_light import Seq2SeqEncoder
>>> Seq2SeqEncoder.list_available()
['compose', 'feedforward', 'gated-cnn-encoder', 'pass_through', 'gru', 'lstm', 'rnn', 'augmented_lstm', 'alternating_lstm', 'stacked_bidirectional_lstm', 'pytorch_transformer']
```

## Copyright

Below is the copyright notice that applies to all source codes.

```
Copyright 2017 The Allen Institute for Artificial Intelligence
Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
SPDX-License-Identifier: Apache-2.0
```

## List of changes

I kept the log of how I got from allennlp to allennlp-light.

```
Copied with changes from 
   
    https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f

Only codes from allennlp/modules and allennlp/nn folders are copied.

The purpose is to integrate AllenNLP modules with the Tango project (https://github.com/allenai/tango).

The following is the list of the changes made to the AllenNLP original (allennlp/modules and allennlp/nn) files:

Removed files and folders:
- allennlp/modules/transformer
- allennlp/modules/token_embedders
- allennlp/modules/text_field_embedders
- allennlp/modules/backbones
- allennlp/modules/elmo.py
- allennlp/modules/elmo_lstm.py
- allennlp/nn/parallel
- allennlp/nn/checkpoint
- allennlp/nn/beam_search.py
- allennlp/nn/module.py

Removed from the nn/util.py file:
- line: from itertools import chain
- line: import torch.distributed as dist
- line: from allennlp.common.util import int_to_device, is_distributed, is_global_primary
- func: find_text_field_embedder
- func: find_embedding_layer
- func: move_to_device
- func: distributed_device
- line: _V = TypeVar("_V", int, float, torch.Tensor)
- func: dist_reduce
- func: dist_reduce_sum
- func: _collect_state_dict
- func: load_state_dict_distributed
- func: _broadcast_params
- class: _IncompatibleKeys
- func: _check_incompatible_keys 

Removed from the nn/__init__.py file:
- line: from allennlp.nn.module import Module

Removed/added from/to the modules/__init__.py file:
- line: from allennlp.modules.backbones import Backbone
- line: from allennlp.modules.elmo import Elmo
- line: from allennlp.modules.text_field_embedders import TextFieldEmbedder
- line: from allennlp.modules.token_embedders import TokenEmbedder, Embedding
+ line: from allennlp.modules.span_extractors import SpanExtractor

Removed/added from/to the modules/span_extractors/span_extractor_with_span_width_embedding.py file:
- from allennlp.modules.token_embedders.embedding import Embedding
+ from torch.nn import Embedding

Removed from /nn/initializers.py file:
- class: PretrainedModelInitializer

Renamed across all files and folders:
* from allennlp.common.checks import ConfigurationError -> from tango.common.exceptions import ConfigurationError 
* from allennlp.common -> from tango.common // this line redirects imports of Registrable and FromParams classes to Tango versions
* allennlp -> allennlp-light
```