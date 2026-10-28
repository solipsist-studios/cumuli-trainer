# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""MCMC densification (Kheradmand et al. 2024, "3D Gaussian Splatting as
Markov Chain Monte Carlo").

Every ``refine_every`` steps between ``refine_start`` and ``refine_stop``:

1. Gaussians whose opacity is at or below ``min_opacity``, or that sit
   outside the scene bounds, are dead. Each is moved onto a live Gaussian
   drawn with probability proportional to its opacity (paper section 4,
   equation 9 for the opacity/scale correction).
2. The population grows by ``grow_rate`` (5%) up to ``cap_max``, with new
   Gaussians drawn the same way.

Every step, positions get exploration noise shaped by each Gaussian's
covariance and gated to nearly transparent ones (equation 8), scaled by the
current position learning rate times ``noise_lr``.

Parameters mirror gsplat's MCMCStrategy, which this follows; the ops are
local (``ops.py``) so every per-Gaussian tensor of the representation moves
with its row.
"""

from __future__ import annotations

import torch

from ..registry import register
from . import ops
from .base import Strategy, StrategyContext


@register("strategy", "mcmc")
class MCMCStrategy(Strategy):
    # The paper's opacity and scale regularisers default OFF. On real
    # capture they starve the model: on Heidi (10 cameras, 61 frames,
    # held-out cam05) weights 0 / 0.0003 / 0.001 gave 36.39 / 36.23 / 36.07
    # dB and LPIPS 0.0093 / 0.0101 / 0.0123, with 297k / 205k / 110k of 300k
    # Gaussians left visible. They help on the 12-frame synthetic dur_12
    # (39.90 dB off, 40.52 at 0.001), but off still beats OMG4 there.
    # A 300k cap beat 500k on dur_12 (40.52 vs 40.15 dB) and 1M on Heidi
    # (36.39 vs 36.07 dB).
    default_losses = {"opacity_reg": 0.0, "scale_reg": 0.0}

    def __init__(self, cap_max=300_000, noise_lr=5e5, refine_start=500, refine_stop=25_000,
                 refine_every=100, min_opacity=0.005, grow_rate=1.05, verbose=False):
        super().__init__()
        self.cap_max = int(cap_max)
        self.noise_lr = float(noise_lr)
        self.refine_start = int(refine_start)
        self.refine_stop = int(refine_stop)
        self.refine_every = int(refine_every)
        self.min_opacity = float(min_opacity)
        self.grow_rate = float(grow_rate)
        self.verbose = bool(verbose)
        self.last = {"relocated": 0, "added": 0}

    def initialize(self, params, optimizers, ctx: StrategyContext) -> None:
        super().initialize(params, optimizers, ctx)
        self.binoms = ops.binomial_table().to(ctx.device)

    def is_refine_step(self, step: int) -> bool:
        return (self.refine_start < step < self.refine_stop) and step % self.refine_every == 0

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, step, info, lr):
        if self.is_refine_step(step):
            dead, weights = self._dead_and_weights()
            n_reloc = ops.relocate(params, optimizers, dead, weights, self.binoms, self.min_opacity)
            n = params["means"].shape[0]
            target = min(self.cap_max, int(self.grow_rate * n))
            n_add = ops.sample_add(params, optimizers, max(0, target - n),
                                   self._dead_and_weights()[1], self.binoms, self.min_opacity)
            self.last = {"relocated": n_reloc, "added": n_add}
            if self.verbose:
                print(f"step {step}: relocated {n_reloc}, added {n_add}, total {params['means'].shape[0]}")
        if self.noise_lr > 0:
            ops.inject_position_noise(params, self.ctx.opacity(), lr * self.noise_lr)

    def _dead_and_weights(self):
        """Dead: opacity at or below min_opacity, or outside the
        scene bounds. Position noise grows as opacity falls, so a Gaussian
        idling just above min_opacity random-walks away; on Heidi with the
        regularisers off, 2,349 of 297k ended 10 to 800 units out (the
        subject spans about 3) and wrecked viewer framing. Out-of-bounds
        Gaussians are recycled here and never chosen as copy sources."""
        peak = self.ctx.opacity()
        dead = peak <= self.min_opacity
        if self.ctx.in_bounds is None:
            return dead, peak
        out = ~self.ctx.in_bounds()
        return dead | out, peak.masked_fill(out, 0.0)

    def state_dict(self):
        return {"last": dict(self.last)}

    def load_state_dict(self, state):
        self.last = dict(state.get("last", self.last))

    def describe(self):
        return (f"mcmc(cap_max={self.cap_max}, noise_lr={self.noise_lr:g}, "
                f"refine {self.refine_start}..{self.refine_stop} every {self.refine_every}, "
                f"min_opacity={self.min_opacity})")
