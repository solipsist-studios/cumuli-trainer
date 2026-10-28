# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Strategy plugin interface.

A strategy grows, prunes and relocates Gaussians. It edits the
representation's parameter dict and the matching Adam optimizers in place
(see ``ops.py``), so it works for any representation that uses gsplat's key
names, whatever extra per-Gaussian tensors it carries.

The training loop calls, every step:

    strategy.step_pre_backward(params, optimizers, step, info)
    loss.backward()
    strategy.step_post_backward(params, optimizers, step, info, lr)

``info`` is the renderer's info dict. ``lr`` is the current position learning rate. Nothing else about the
strategy is visible to the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch


@dataclass
class StrategyContext:
    scene_scale: float
    iterations: int
    device: torch.device
    #: [N] each Gaussian's opacity (the activated opacity parameter).
    opacity: Callable[[], torch.Tensor]
    #: [N] bool, True where a Gaussian's position lies inside the scene
    #: bounds. None when the bounds are disabled (scene_bound_factor 0).
    in_bounds: Optional[Callable[[], torch.Tensor]] = None


class Strategy:
    """Base class. Subclasses register with ``@register("strategy", name)``."""

    #: Loss terms this strategy expects, merged under the config's own
    #: ``losses`` (the config wins; weight 0 removes a term).
    default_losses: Dict[str, float] = {}

    def __init__(self, **params):
        if params:
            raise TypeError(f"{type(self).__name__}: unknown parameter(s) {sorted(params)}")

    def initialize(self, params, optimizers, ctx: StrategyContext) -> None:
        self.ctx = ctx

    def step_pre_backward(self, params, optimizers, step: int, info: Dict[str, Any]) -> None:
        pass

    def step_post_backward(self, params, optimizers, step: int, info: Dict[str, Any], lr: float) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass

    def describe(self) -> str:
        return type(self).__name__
