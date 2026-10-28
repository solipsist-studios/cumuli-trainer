# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Test-camera scoring during training.

Renders each test camera at its own timestamp, composites onto the test
background, clamps to [0, 1] and scores PSNR, SSIM and LPIPS with
``cumuli_core.metrics``, which matches cumuli ``scripts/eval_render.py``.
Ground truth is the test frame's RGBA composited onto the same background.
The external evaluator renders with no background, which is black, so with
``background: black`` the two should agree to rounding.
"""

from __future__ import annotations

import numpy as np
import torch

from .core import metrics as core_metrics
from .core import render as core_render
from .data import _load_uint8


class EvalSet:
    def __init__(self, dataset, cams, device, background):
        self.cams = cams
        self.device = device
        self.bg = torch.tensor(background, dtype=torch.float32, device=device)
        self.images = [torch.from_numpy(np.ascontiguousarray(_load_uint8(dataset, c))) for c in cams]
        self.viewmats, self.Ks = core_render.camera_tensors(cams, device) if cams else (None, None)
        self._lpips = None

    def lpips(self):
        if self._lpips is None:
            self._lpips = core_metrics.LPIPS(device=self.device)
        return self._lpips

    @torch.no_grad()
    def evaluate(self, representation, renderer, sh_degree):
        rows = []
        lp = self.lpips()
        for i, cam in enumerate(self.cams):
            g = representation.slice(cam.time, sh_degree)
            rgb, alpha, _ = renderer.render(g, self.viewmats[i:i + 1], self.Ks[i:i + 1],
                                            cam.width, cam.height)
            img = (rgb[0] + (1.0 - alpha[0]) * self.bg).clamp(0.0, 1.0)
            gt = self.images[i].to(self.device).float() / 255.0
            a = gt[..., 3:4]
            gt = gt[..., :3] * a + (1.0 - a) * self.bg
            rows.append({
                "name": cam.name, "time": float(cam.time),
                "psnr_db": float(core_metrics.psnr(img, gt)),
                "ssim": float(core_metrics.ssim(img, gt)),
                "lpips": float(lp(img, gt)),
            })
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr_db", "ssim", "lpips")} if rows else {}
        return {"mean": mean, "views": rows}
