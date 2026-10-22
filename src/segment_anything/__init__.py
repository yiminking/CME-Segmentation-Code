# Packaging change (2026-10-22): added this attribution notice only.
# This file is from the study-supplied SAM tree, which includes adaptations.
# Historical base revision unknown; see THIRD_PARTY_NOTICES.txt.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator
