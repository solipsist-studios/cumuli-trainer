# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Adaptive density control (Kerbl et al. 2023, section 5.2).

Every step, for each Gaussian, the norm of the screen-space position gradient
is accumulated over the cameras that saw it. Every ``refine_every`` steps
between ``refine_start`` and ``refine_stop``, Gaussians whose mean
gradient exceeds ``grow_grad2d`` are cloned (small) or split (large), and
Gaussians with low opacity, or too large after the first opacity reset, are
removed. Every ``reset_every`` steps opacities are clamped to
``2 * prune_opa``.

Pruning uses each Gaussian's opacity.

Statistics are accumulated densely (no index lists), so the per-step update
has no host synchronisation. The gradient normalisation follows gsplat's
DefaultStrategy so its thresholds carry over: screen-space gradients are
scaled to [-1, 1] NDC and multiplied by the number of cameras in the batch,
which undoes the batch mean in the loss.

``cap_max`` limits growth: when the candidates would exceed it, only the
highest-gradient ones grow.
"""

from __future__ import annotations

import torch

from ..registry import register
from . import ops
from .base import Strategy, StrategyContext


@register("strategy", "default")
class DefaultStrategy(Strategy):
    default_losses = {}

    def __init__(self, prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01, prune_scale3d=0.1,
                 refine_start=500, refine_stop=15_000, refine_every=100, reset_every=3000,
                 pause_refine_after_reset=0, absgrad=False, cap_max=3_000_000,
                 verbose=False):
        super().__init__()
        self.prune_opa = float(prune_opa)
        self.grow_grad2d = float(grow_grad2d)
        self.grow_scale3d = float(grow_scale3d)
        self.prune_scale3d = float(prune_scale3d)
        self.refine_start = int(refine_start)
        self.refine_stop = int(refine_stop)
        self.refine_every = int(refine_every)
        self.reset_every = int(reset_every)
        self.pause_refine_after_reset = int(pause_refine_after_reset)
        self.absgrad = bool(absgrad)
        self.cap_max = int(cap_max)
        self.verbose = bool(verbose)
        self.state = {"grad2d": None, "count": None}
        self.last = {}

    @property
    def wants_absgrad(self) -> bool:
        return self.absgrad

    def initialize(self, params, optimizers, ctx: StrategyContext) -> None:
        super().initialize(params, optimizers, ctx)
        n = params["means"].shape[0]
        if self.state["grad2d"] is None or self.state["grad2d"].shape[0] != n:
            self.state = {"grad2d": torch.zeros(n, device=ctx.device),
                          "count": torch.zeros(n, device=ctx.device)}

    def step_pre_backward(self, params, optimizers, step, info):
        if step < self.refine_stop:
            info["means2d"].retain_grad()

    @torch.no_grad()
    def visibility(self, info) -> torch.Tensor:
        """[C, N] bool: camera c counts toward Gaussian n's statistics."""
        radii = info["radii"]
        vis = (radii > 0).all(dim=-1) if radii.dim() == 3 else radii > 0
        return vis

    @torch.no_grad()
    def accumulate(self, info) -> None:
        m2d = info["means2d"]
        g = m2d.absgrad if self.absgrad else m2d.grad
        if g is None:
            return
        g = g.clone()
        g[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        g[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        vis = self.visibility(info).float()  # [C, N]
        self.state["grad2d"] += (g.norm(dim=-1) * vis).sum(dim=0)
        self.state["count"] += vis.sum(dim=0)

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, step, info, lr):
        if step >= self.refine_stop:
            return
        self.accumulate(info)
        if (step > self.refine_start and step % self.refine_every == 0
                and step % self.reset_every >= self.pause_refine_after_reset):
            n_dup, n_split = self._grow(params, optimizers)
            n_prune = self._prune(params, optimizers, step)
            self.state["grad2d"].zero_()
            self.state["count"].zero_()
            self.last = {"duplicated": n_dup, "split": n_split, "pruned": n_prune}
            if self.verbose:
                print(f"step {step}: +{n_dup} cloned, +{n_split} split, -{n_prune} pruned, "
                      f"total {params['means'].shape[0]}")
        if step > 0 and step % self.reset_every == 0:
            ops.clamp_opacity(params, optimizers, self.prune_opa * 2.0)

    def _grow(self, params, optimizers):
        grads = self.state["grad2d"] / self.state["count"].clamp(min=1)
        high = grads > self.grow_grad2d
        n = params["means"].shape[0]
        budget = self.cap_max - n
        if budget <= 0:
            return 0, 0
        small = torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * self.ctx.scene_scale
        # a split adds one net Gaussian, a clone adds one
        n_cand = int(high.sum())
        if n_cand > budget:
            thresh = torch.topk(grads[high], budget).values.min()
            high = high & (grads >= thresh)
        dup = high & small
        spl = high & ~small
        n_dup = int(dup.sum())
        n_split = int(spl.sum())
        if n_dup:
            ops.duplicate(params, optimizers, dup, self.state)
            spl = torch.cat([spl, torch.zeros(n_dup, dtype=torch.bool, device=spl.device)])
        if n_split:
            ops.split(params, optimizers, spl, self.state)
        return n_dup, n_split

    def _prune(self, params, optimizers, step):
        prune = self.ctx.opacity() < self.prune_opa
        if step > self.reset_every:
            too_big = torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d * self.ctx.scene_scale
            prune = prune | too_big
        n = int(prune.sum())
        if n:
            ops.remove(params, optimizers, prune, self.state)
        return n

    def state_dict(self):
        return {"grad2d": self.state["grad2d"], "count": self.state["count"], "last": self.last}

    def load_state_dict(self, state):
        self.state = {"grad2d": state.get("grad2d"), "count": state.get("count")}
        self.last = state.get("last", {})

    def describe(self):
        return (f"default(grow_grad2d={self.grow_grad2d:g}, refine {self.refine_start}..{self.refine_stop}, "
                f"reset_every={self.reset_every}, cap_max={self.cap_max})")
