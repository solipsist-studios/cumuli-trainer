# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Loss-term plugin interface.

Each term is one plugin. The config lists terms with weights:

    losses: {l1: 0.8, ssim: 0.2, opacity_mask: 0.005}

and the loop adds ``weight * term(inputs)`` over all terms. A term sees the
whole ``LossInputs`` so regularisers can read parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch


@dataclass
class LossInputs:
    pred_rgb: torch.Tensor  # [C, H, W, 3] composited onto the background
    pred_alpha: torch.Tensor  # [C, H, W, 1]
    gt_rgb: torch.Tensor  # [C, H, W, 3] composited onto the same background
    gt_alpha: torch.Tensor  # [C, H, W, 1]
    params: Dict[str, torch.nn.Parameter]
    gaussians: Any  # SlicedGaussians
    info: Dict[str, Any]
    step: int
    extras: Dict[str, Any] = field(default_factory=dict)


class LossTerm:
    """Base class. Subclasses register with ``@register("loss", name)``."""

    def __init__(self, **params):
        if params:
            raise TypeError(f"{type(self).__name__}: unknown parameter(s) {sorted(params)}")

    def __call__(self, inputs: LossInputs) -> torch.Tensor:
        raise NotImplementedError
