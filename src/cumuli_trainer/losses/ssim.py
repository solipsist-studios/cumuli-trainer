# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Differentiable SSIM (Wang et al. 2004) with an 11x11 Gaussian window,
sigma 1.5, constants C1 = 0.01^2 and C2 = 0.03^2, zero padding so the map has
the input's size. Separable convolution: the five local moments are blurred
together as one grouped 1-D convolution along each axis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_WINDOWS = {}


def _window(size: int, sigma: float, device, dtype):
    key = (size, sigma, device, dtype)
    w = _WINDOWS.get(key)
    if w is None:
        x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
        w = torch.exp(-(x * x) / (2 * sigma * sigma))
        w = w / w.sum()
        _WINDOWS[key] = w
    return w


def ssim_map(x: torch.Tensor, y: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM map for [B, C, H, W] images in [0, 1]."""
    b, c, h, w = x.shape
    win = _window(size, sigma, x.device, x.dtype)
    stack = torch.cat([x, y, x * x, y * y, x * y], dim=1)  # [B, 5C, H, W]
    ch = stack.shape[1]
    kx = win.view(1, 1, 1, size).expand(ch, 1, 1, size)
    ky = win.view(1, 1, size, 1).expand(ch, 1, size, 1)
    pad = size // 2
    blur = F.conv2d(stack, kx, padding=(0, pad), groups=ch)
    blur = F.conv2d(blur, ky, padding=(pad, 0), groups=ch)
    mx, my, sxx, syy, sxy = blur.split(c, dim=1)
    vx = sxx - mx * mx
    vy = syy - my * my
    cxy = sxy - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mx * my + c1) * (2 * cxy + c2)
    den = (mx * mx + my * my + c1) * (vx + vy + c2)
    return num / den


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ssim_map(x, y).mean()
