# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Renderer plugin interface.

``render(gaussians, viewmats, Ks, width, height)`` rasterises one sliced set
of Gaussians into C cameras at once and returns

    rgb    [C, H, W, 3]  premultiplied colour (no background)
    alpha  [C, H, W, 1]
    info   dict that strategies read (gsplat keys: means2d, radii, width,
           height, n_cameras, ...)

The loop composites the background itself, so every renderer and every loss
sees the same compositing rule.
"""

from __future__ import annotations


class Renderer:
    def __init__(self, **params):
        if params:
            raise TypeError(f"{type(self).__name__}: unknown parameter(s) {sorted(params)}")

    def configure(self, absgrad: bool = False) -> None:
        """Called once after the strategy is built."""
        self.absgrad = absgrad

    def render(self, gaussians, viewmats, Ks, width, height):
        raise NotImplementedError

    def render_each(self, gaussians_per_view, viewmats, Ks, width, height):
        """Render view c from its own sliced Gaussians ``gaussians_per_view[c]``
        (cameras of one instant that took their pictures at slightly different
        times).  Returns rgb [C, H, W, 3], alpha [C, H, W, 1] and an info whose
        per-view tensors carry an extra batch dimension ([C, 1, N, ...]);
        ``n_cameras`` counts every view."""
        raise NotImplementedError(f"{type(self).__name__} cannot render views at different times")
