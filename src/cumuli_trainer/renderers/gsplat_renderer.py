# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""gsplat rasteriser.

Uses gsplat's ``rasterization`` with the settings cumuli's
``scripts/eval_render.py`` uses (classic mode, eps2d 0.3, near 0.01), so a
training render and an evaluation render agree. ``packed`` is off: with
every camera of a batch sharing one set of Gaussians, the dense [C, N] layout
lets the strategy gate visibility without index lists.
"""

from __future__ import annotations

from ..core import render as core_render
from ..registry import register
from .base import Renderer


@register("renderer", "gsplat")
class GsplatRenderer(Renderer):
    def __init__(self, rasterize_mode="classic", eps2d=0.3, near_plane=0.01, far_plane=1e10,
                 packed=False, tile_size=16):
        super().__init__()
        self.kw = dict(rasterize_mode=rasterize_mode, eps2d=float(eps2d),
                       near_plane=float(near_plane), far_plane=float(far_plane),
                       packed=bool(packed), tile_size=int(tile_size))
        self.absgrad = False

    def render(self, g, viewmats, Ks, width, height):
        rgb, alpha, info = core_render.render(
            g.means, g.quats, g.scales, g.opacities, g.colors, (viewmats, Ks),
            int(width), int(height), g.sh_degree, render_mode="RGB",
            absgrad=self.absgrad, **self.kw)
        return rgb, alpha, info

    def render_each(self, gs, viewmats, Ks, width, height):
        import torch

        first = gs[0]
        stack = lambda name: torch.stack([getattr(g, name) for g in gs])  # noqa: E731
        rgb, alpha, info = core_render.render(
            stack("means"), stack("quats"), stack("scales"), stack("opacities"), stack("colors"),
            (viewmats[:, None], Ks[:, None]), int(width), int(height), first.sh_degree,
            render_mode="RGB", absgrad=self.absgrad, **self.kw)
        info["n_cameras"] = int(info.get("n_batches", len(gs))) * int(info.get("n_cameras", 1))
        return rgb[:, 0], alpha[:, 0], info
