# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Representation plugin interface.

A representation owns a dict of ``torch.nn.Parameter`` tensors, one row per
Gaussian. It uses gsplat's key names (``means``, ``quats``, ``scales``,
``opacities``, ``sh0``, ``shN``) so gsplat-style strategy ops can duplicate,
split, remove and relocate rows; any extra per-Gaussian tensors in the dict
follow those ops automatically.

The training loop only calls the methods below. Time is passed in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class RepresentationContext:
    time_min: float
    time_max: float
    fps: float
    scene_scale: float
    sh_degree: int
    t_init_div: float
    device: torch.device


@dataclass
class SlicedGaussians:
    """What a renderer needs for one instant: 3D Gaussians, activated."""
    means: torch.Tensor  # [N, 3]
    quats: torch.Tensor  # [N, 4] wxyz, unnormalised is fine
    scales: torch.Tensor  # [N, 3] linear
    opacities: torch.Tensor  # [N] in [0, 1], already sliced in time
    colors: torch.Tensor  # [N, K, 3] SH coefficients
    sh_degree: int  # active SH degree
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)


class Representation:
    """Base class. Subclasses register with ``@register("representation", name)``."""

    def __init__(self, ctx: RepresentationContext, **params):
        self.ctx = ctx
        self.params: Dict[str, torch.nn.Parameter] = {}

    # -- construction -----------------------------------------------------
    def init_from_points(self, points: dict) -> None:
        """Seed from ``{"xyz", "rgb", "time"}`` arrays (time may be None)."""
        raise NotImplementedError

    def init_from_interchange(self, header: dict, fields: dict) -> None:
        """Seed from an interchange PLY's fields."""
        raise NotImplementedError

    def init_empty(self, shapes: Dict[str, tuple]) -> None:
        """Allocate parameters of the given shapes (before loading a checkpoint)."""
        self._set_params({
            k: torch.nn.Parameter(torch.zeros(s, device=self.ctx.device)) for k, s in shapes.items()
        })

    # -- training ---------------------------------------------------------
    def learning_rates(self, lr_cfg: dict) -> Dict[str, float]:
        """Initial Adam learning rate for every parameter."""
        raise NotImplementedError

    def decayed_params(self) -> tuple:
        """Parameter names whose LR follows the position decay schedule."""
        return ("means",)

    def slice(self, t: float, sh_degree: int) -> SlicedGaussians:
        """Differentiable Gaussians at time ``t`` (seconds)."""
        raise NotImplementedError

    def peak_opacity(self) -> torch.Tensor:
        """[N] highest opacity each Gaussian reaches inside the training window.
        Export uses this to leave out Gaussians never visible in the clip."""
        return torch.sigmoid(self.params["opacities"]).detach()

    # -- output -----------------------------------------------------------
    def to_interchange(self, min_peak_opacity: float = 0.0, keep=None) -> dict:
        """Interchange PLY fields (numpy), cumuli docs/sogst-format.md section 7,
        leaving out Gaussians whose peak opacity inside the clip is below
        ``min_peak_opacity`` and, when ``keep`` ([N] bool tensor) is given,
        those where it is False."""
        raise NotImplementedError

    @property
    def num_gaussians(self) -> int:
        return int(self.params["means"].shape[0])

    def state_shapes(self) -> Dict[str, tuple]:
        return {k: tuple(v.shape) for k, v in self.params.items()}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach() for k, v in self.params.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self._set_params({
            k: torch.nn.Parameter(v.to(self.ctx.device).clone()) for k, v in state.items()
        })

    def _set_params(self, new: Dict[str, torch.nn.Parameter]) -> None:
        """Replace the contents of ``self.params`` in place. Strategies hold a
        reference to this dict, so its identity must never change."""
        self.params.clear()
        self.params.update(new)

    def extra_state(self) -> dict:
        return {}

    def load_extra_state(self, state: Optional[dict]) -> None:
        pass
