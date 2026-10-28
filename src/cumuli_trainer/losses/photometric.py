# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Photometric terms on the composited RGB: L1 and D-SSIM (1 - SSIM).
The 3DGS paper (Kerbl et al. 2023, section 7) weights them 0.8 and 0.2."""

from __future__ import annotations

import torch

from ..registry import register
from .base import LossInputs, LossTerm
from .ssim import ssim, ssim_map


@register("loss", "l1")
class L1Loss(LossTerm):
    def __call__(self, inp: LossInputs) -> torch.Tensor:
        return (inp.pred_rgb - inp.gt_rgb).abs().mean()


@register("loss", "ssim")
class SSIMLoss(LossTerm):
    """1 - SSIM over the full frame, so a perfect match costs 0.

    With ``crop: true`` (default) SSIM is computed only inside each view's
    foreground crop window (``LossInputs.extras["crop"]``, measured from the
    GT alpha at load time) and every pixel outside counts as SSIM = 1. That
    is exact wherever render and GT are both the plain background, because
    SSIM of two identical windows is 1. A floater outside the window is still
    charged by L1 and the opacity-mask term. On the 12-camera parity set the
    windows cover about a tenth of the frame and this cuts the step time by
    roughly two thirds.
    """

    def __init__(self, crop: bool = True):
        super().__init__()
        self.crop = bool(crop)

    def __call__(self, inp: LossInputs) -> torch.Tensor:
        crop = inp.extras.get("crop") if self.crop else None
        if crop is None:
            p = inp.pred_rgb.permute(0, 3, 1, 2)
            g = inp.gt_rgb.permute(0, 3, 1, 2)
            return 1.0 - ssim(p, g)
        (hc, wc), origins = crop
        c, h, w, _ = inp.pred_rgb.shape
        p = torch.stack([inp.pred_rgb[i, y:y + hc, x:x + wc] for i, (y, x) in enumerate(origins)])
        g = torch.stack([inp.gt_rgb[i, y:y + hc, x:x + wc] for i, (y, x) in enumerate(origins)])
        smap = ssim_map(p.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2))
        return (1.0 - smap).sum() / float(c * 3 * h * w)
