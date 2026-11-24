# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""The two motion regularisers of the TRiGS paper (arXiv 2604.00538, Eqs. 12-14), written from the paper.

trigs_motion   the second difference of the three Bezier control twists, mean over Gaussians of
               ||zeta0 - 2 zeta1 + zeta2||^2 (rotation and translation parts): a curve that bends is
               paid for, a straight one is free.  Paper weight 1e-4.

trigs_rigid    neighbouring Gaussians that look alike should move alike: for each Gaussian and its
               k nearest neighbours (by canonical mean), K_ij ||zeta_base_i - zeta_base_j||^2 with
               K_ij = exp(-lambda_c ||c_i - c_j||^2) on the DC colour, for the rotation and translation
               halves of the base twist.  Paper: k = 3, lambda_c = 50, weight 1.0.

Ours, where the paper is silent: both are averaged over Gaussians (the paper's sums would grow with the
population, and its weights were set for a population of 0.5M); colour is the DC colour in [0, 1]
(0.28209479 * sh0 + 0.5); the neighbour table is rebuilt on the CPU every ``refresh`` steps, and
whenever the population changes in size, rather than every step.
"""

from __future__ import annotations

import numpy as np
import torch

from ..registry import register
from .base import LossInputs, LossTerm

SH_C0 = 0.28209479177387814


@register("loss", "trigs_motion")
class TrigsMotion(LossTerm):
    def __call__(self, inp: LossInputs) -> torch.Tensor:
        p = inp.params
        total = p["means"].new_zeros(())
        for half in ("r", "t"):
            bend = p[f"zeta_c0_{half}"] - 2.0 * p[f"zeta_c1_{half}"] + p[f"zeta_c2_{half}"]
            total = total + bend.pow(2).sum(dim=1).mean()
        return total


@register("loss", "trigs_rigid")
class TrigsRigid(LossTerm):
    def __init__(self, k: int = 3, color_lambda: float = 50.0, refresh: int = 200, max_points: int = 600_000):
        super().__init__()
        self.k, self.color_lambda, self.refresh, self.max_points = int(k), float(color_lambda), int(refresh), int(max_points)
        self._neighbors = None     # [N, k] long, on the device of the parameters
        self._built_for = (-1, -1)  # (step bucket, population)

    @torch.no_grad()
    def _build(self, means: torch.Tensor) -> torch.Tensor:
        from scipy.spatial import cKDTree

        pts = means.detach().float().cpu().numpy()
        k = min(self.k, max(1, len(pts) - 1))
        _, idx = cKDTree(pts).query(pts, k=k + 1)
        return torch.as_tensor(np.asarray(idx)[:, 1:], dtype=torch.long, device=means.device)

    def __call__(self, inp: LossInputs) -> torch.Tensor:
        p = inp.params
        n = p["means"].shape[0]
        if n < 2:
            return p["means"].new_zeros(())
        key = (inp.step // self.refresh, n)
        if self._neighbors is None or key != self._built_for:
            self._neighbors, self._built_for = self._build(p["means"]), key
        nb = self._neighbors
        colour = (p["sh0"][:, 0, :] * SH_C0 + 0.5).detach()
        weight = torch.exp(-self.color_lambda * (colour[:, None, :] - colour[nb]).pow(2).sum(-1))      # [N, k]
        total = p["means"].new_zeros(())
        for key_ in ("zeta_base_r", "zeta_base_t"):
            z = p[key_]
            total = total + (weight * (z[:, None, :] - z[nb]).pow(2).sum(-1)).mean()
        return total
