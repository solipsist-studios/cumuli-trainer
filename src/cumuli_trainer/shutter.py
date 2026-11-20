# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Rolling-shutter views: the Gaussians a view saw, each at its own time.

A camera with a readout time (``readout_s`` in its transforms row) read its
raw sensor line by line, so the top and the bottom of its photo show the
subject at different moments. ``slice_view`` gives each Gaussian the time of
the raw row its centre falls on in that view (``cumuli_core.shutter``,
shared with ``scripts/eval_render.py`` so training and scoring agree) and
slices the representation at those per-Gaussian times. The view is then
rendered once, through the batched per-view path the trainer already uses
for cameras with their own capture times.

Method and its trade-offs: ``cumuli_core.shutter.rolling_shutter_offsets``.
In short: per-row-band rendering is exact but costs a full render (forward,
backward and buffers) per band; per-Gaussian times cost one projection of
the means per view and nothing per pixel, and are exact up to the shear
inside a single Gaussian's footprint.

A view with no readout slices exactly as before, so a dataset without
``readout_s`` trains bit-identically.
"""

from __future__ import annotations

import torch

try:
    from cumuli_core import shutter as core_shutter
except ImportError:  # an older cumuli_core: every camera is a global shutter
    core_shutter = None


def readout_of(cam) -> float:
    """Seconds from the first raw line to the last; 0 for a global shutter."""
    return float(getattr(cam, "readout_s", 0.0) or 0.0) if core_shutter is not None else 0.0


def is_rolling(cam) -> bool:
    return readout_of(cam) != 0.0


def gaussian_offsets(rep, t, cam, viewmat) -> torch.Tensor:
    """[N] seconds after ``t`` at which the rolling-shutter view ``cam``
    (world-to-camera ``viewmat`` [4, 4]) saw each Gaussian. No gradient."""
    return core_shutter.rolling_shutter_offsets(rep.means_at, viewmat, core_shutter.sensor_of(cam), t,
                                                readout_of(cam))


def slice_view(rep, t, sh_degree: int, cam, viewmat):
    """The Gaussians view ``cam`` saw, at capture time ``t`` (seconds, a float
    or a tensor carrying learned corrections). A global-shutter view is
    ``rep.slice(t, sh_degree)``; a rolling-shutter one slices each Gaussian at
    ``t`` plus its row's offset, and the gradient still reaches ``t``."""
    if not is_rolling(cam):
        return rep.slice(t, sh_degree)
    return rep.slice(t + gaussian_offsets(rep, t, cam, viewmat), sh_degree)
