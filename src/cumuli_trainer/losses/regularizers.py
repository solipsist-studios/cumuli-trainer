# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Parameter regularisers used with MCMC densification (Kheradmand et al.
2024, section 5): mean opacity and mean scale, which push unused Gaussians
toward transparency and small size so relocation can recycle them.

opacity_reg uses the peak opacity (sigmoid of the logit). scale_reg uses the
linear scale in scene units.
"""

from __future__ import annotations

import torch

from ..registry import register
from .base import LossInputs, LossTerm


@register("loss", "opacity_reg")
class OpacityReg(LossTerm):
    def __call__(self, inp: LossInputs) -> torch.Tensor:
        return torch.sigmoid(inp.params["opacities"]).mean()


@register("loss", "scale_reg")
class ScaleReg(LossTerm):
    def __call__(self, inp: LossInputs) -> torch.Tensor:
        return torch.exp(inp.params["scales"]).mean()


@register("loss", "aniso_reg")
class AnisoReg(LossTerm):
    """Hinge on each Gaussian's axis ratio: mean(max(0, log(s_max / s_min) -
    log(max_ratio))). Only the part of the ratio above ``max_ratio`` costs,
    so ordinary elongation is free and needles are not.

    Why: with the MCMC opacity regulariser off, faint needles turn opaque.
    On Heidi 29.6% of Gaussians were opaque with an axis ratio above 20
    (3.3% with the regulariser on), and a sub-pixel-thin opaque needle
    draws as a bright line that sparkles as the view or time changes.
    The hinge on the axis ratio follows the anisotropy regulariser in Xie et
    al. 2024, "PhysGaussian"; taking it on the log ratio, so the cost grows
    with orders of magnitude rather than raw ratio, is this module's own."""

    def __init__(self, max_ratio: float = 10.0):
        super().__init__()
        if max_ratio <= 1.0:
            raise ValueError("aniso_reg max_ratio must be > 1")
        self.log_max_ratio = float(torch.log(torch.tensor(float(max_ratio))))

    def __call__(self, inp: LossInputs) -> torch.Tensor:
        s = inp.params["scales"]  # log scales
        excess = s.max(dim=1).values - s.min(dim=1).values - self.log_max_ratio
        return torch.relu(excess).mean()


@register("loss", "sh_reg")
class SHReg(LossTerm):
    """Mean squared view-dependent SH energy: mean over Gaussians of the sum
    of squared higher-band coefficients (shN). The DC colour is free.

    Why: ten cameras pin a Gaussian's colour down in about ten directions.
    Between them the fitted SH lobes extrapolate into hues the subject does
    not have, which shows as coloured glints once the Gaussians are opaque
    (Heidi close-up orbit: 0.94 glint pixels per 10k against 0.12 with SH
    off). Charging for the energy keeps view-dependence where the data
    demand it."""

    def __call__(self, inp: LossInputs) -> torch.Tensor:
        sh_n = inp.params.get("shN")
        if sh_n is None or sh_n.numel() == 0:
            return inp.params["means"].new_zeros(())
        return sh_n.pow(2).sum(dim=(1, 2)).mean()
