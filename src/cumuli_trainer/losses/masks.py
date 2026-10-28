# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Opacity-mask term: penalise rendered opacity where the ground-truth alpha
says background, whatever its colour.

    lambda * mean(-(1 - gt_alpha) * log(1 - rendered_alpha + eps))

This is a binary cross-entropy restricted to the background side. It is a
Solipsist specification (cumuli comfyui pack, ``lambda_opa_mask``, measured
production weight 0.005). Unlike matching the background colour, it removes
floaters of any shade.
"""

from __future__ import annotations

import torch

from ..registry import register
from .base import LossInputs, LossTerm


@register("loss", "opacity_mask")
class OpacityMaskLoss(LossTerm):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def __call__(self, inp: LossInputs) -> torch.Tensor:
        a = inp.pred_alpha.clamp(0.0, 1.0)
        return (-(1.0 - inp.gt_alpha) * torch.log(1.0 - a + self.eps)).mean()
