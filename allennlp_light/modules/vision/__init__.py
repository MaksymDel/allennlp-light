# Copyright 2017 The Allen Institute for Artificial Intelligence
# Adapted by Maksym Del from https://github.com/allenai/allennlp/tree/8571d930fe6dc6291c6351c6e599576b007cf22f
# SPDX-License-Identifier: Apache-2.0
from allennlp_light.modules.vision.grid_embedder import GridEmbedder, ResnetBackbone
from allennlp_light.modules.vision.image2image import Image2ImageModule, NormalizeImage
from allennlp_light.modules.vision.region_detector import (
    FasterRcnnRegionDetector,
    RegionDetector,
)
