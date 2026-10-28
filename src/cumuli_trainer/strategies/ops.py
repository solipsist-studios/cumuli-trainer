# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Row operations on a parameter dict and its Adam state.

Every per-Gaussian tensor in ``params`` (gsplat's names plus anything a
representation adds) is edited together with its optimizer moments, so extra
tensors such as velocities and temporal windows follow automatically.

Written from the densification descriptions in Kerbl et al. 2023 (section
5.2: clone small, split large with scale / 1.6, prune transparent) and
Kheradmand et al. 2024 (section 4: relocate dead Gaussians onto live ones
sampled by opacity, with the opacity/scale correction of their equation 9),
using gsplat's public ``compute_relocation`` and ``quat_scale_to_covar_preci``.
Unlike gsplat's ops, the opacity used to choose dead Gaussians and to weight
the sampling is passed in, so a temporal representation can supply its peak
opacity inside the training window.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

Params = Dict[str, torch.nn.Parameter]
Optims = Dict[str, torch.optim.Optimizer]


def binomial_table(n_max: int = 51) -> torch.Tensor:
    t = torch.zeros((n_max, n_max))
    for n in range(n_max):
        for k in range(n + 1):
            t[n, k] = math.comb(n, k)
    return t


@torch.no_grad()
def _apply(params: Params, optimizers: Optims,
           param_fn: Callable[[str, torch.Tensor], torch.Tensor],
           state_fn: Callable[[str, torch.Tensor], torch.Tensor],
           names=None) -> None:
    """Replace each parameter by ``param_fn(name, p)`` and each optimizer
    moment by ``state_fn(key, v)``, re-pointing the optimizer at the new
    tensor. The scalar ``step`` entry is left alone."""
    for name in list(params.keys()) if names is None else names:
        old = params[name]
        new = torch.nn.Parameter(param_fn(name, old.data), requires_grad=old.requires_grad)
        params[name] = new
        opt = optimizers.get(name)
        if opt is None:
            continue
        st = opt.state.pop(old, None)
        if st is not None:
            for key in list(st.keys()):
                if key != "step" and isinstance(st[key], torch.Tensor) and st[key].dim() > 0:
                    st[key] = state_fn(key, st[key])
            opt.state[new] = st
        for group in opt.param_groups:
            group["params"] = [new]


def _extend_state(state: Optional[dict], fn) -> None:
    if not state:
        return
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor) and v.dim() > 0:
            state[k] = fn(v)


@torch.no_grad()
def duplicate(params: Params, optimizers: Optims, mask: torch.Tensor, state: dict = None) -> None:
    sel = mask.nonzero(as_tuple=True)[0]
    _apply(params, optimizers,
           lambda n, p: torch.cat([p, p[sel]]),
           lambda k, v: torch.cat([v, torch.zeros_like(v[sel])]))
    _extend_state(state, lambda v: torch.cat([v, v[sel]]))


@torch.no_grad()
def split(params: Params, optimizers: Optims, mask: torch.Tensor, state: dict = None,
          scale_div: float = 1.6) -> None:
    """Replace each selected Gaussian by two, positioned by sampling its own
    spatial Gaussian, with scales divided by ``scale_div``. Every other key,
    including temporal ones, is copied to both children."""
    sel = mask.nonzero(as_tuple=True)[0]
    rest = (~mask).nonzero(as_tuple=True)[0]
    scales = torch.exp(params["scales"].data[sel])
    q = F.normalize(params["quats"].data[sel], dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    offsets = torch.einsum("nij,bnj->bni", R, torch.randn(2, len(sel), 3, device=sel.device) * scales)

    def pfn(name, p):
        if name == "means":
            child = (p[sel][None] + offsets).reshape(-1, 3)
        elif name == "scales":
            child = torch.log(scales / scale_div).repeat(2, 1)
        else:
            child = p[sel].repeat(2, *([1] * (p.dim() - 1)))
        return torch.cat([p[rest], child])

    def sfn(key, v):
        return torch.cat([v[rest], torch.zeros((2 * len(sel), *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _apply(params, optimizers, pfn, sfn)
    _extend_state(state, lambda v: torch.cat([v[rest], v[sel].repeat(2, *([1] * (v.dim() - 1)))]))


@torch.no_grad()
def remove(params: Params, optimizers: Optims, mask: torch.Tensor, state: dict = None) -> None:
    keep = (~mask).nonzero(as_tuple=True)[0]
    _apply(params, optimizers, lambda n, p: p[keep], lambda k, v: v[keep])
    _extend_state(state, lambda v: v[keep])


@torch.no_grad()
def clamp_opacity(params: Params, optimizers: Optims, max_value: float) -> None:
    """Opacity reset (Kerbl et al. 2023, section 5.2): clamp every opacity to
    at most ``max_value`` and clear its Adam moments."""
    cap = math.log(max_value / (1 - max_value))
    _apply(params, optimizers, lambda n, p: p.clamp(max=cap), lambda k, v: torch.zeros_like(v),
           names=["opacities"])


def _split_opacity_and_scale(params, idx, binoms, min_opacity):
    from gsplat.relocation import compute_relocation
    eps = torch.finfo(torch.float32).eps
    op = torch.sigmoid(params["opacities"].data[idx])
    counts = torch.bincount(idx)[idx] + 1
    new_op, new_scale = compute_relocation(
        opacities=op, scales=torch.exp(params["scales"].data[idx]),
        ratios=counts.clamp(max=binoms.shape[0] - 1).int(), binoms=binoms)
    new_op = new_op.clamp(min=min_opacity, max=1.0 - eps)
    return torch.logit(new_op), torch.log(new_scale)


def _sample(weights: torch.Tensor, n: int) -> torch.Tensor:
    w = weights.float().clamp(min=0)
    if float(w.sum()) <= 0:
        w = torch.ones_like(w)
    if w.numel() <= 2 ** 24:
        return torch.multinomial(w, n, replacement=True)
    import numpy as np
    p = (w / w.sum()).cpu().double().numpy()
    return torch.from_numpy(np.random.choice(len(p), n, p=p / p.sum())).to(w.device)


@torch.no_grad()
def relocate(params: Params, optimizers: Optims, dead: torch.Tensor, weights: torch.Tensor,
             binoms: torch.Tensor, min_opacity: float = 0.005, state: dict = None) -> int:
    """Move every ``dead`` Gaussian onto a live one drawn with probability
    proportional to ``weights``. Source and copies share the source's
    opacity and scale after the correction, so the rendered result is
    approximately unchanged. Every other key is copied from the source."""
    dead_idx = dead.nonzero(as_tuple=True)[0]
    alive_idx = (~dead).nonzero(as_tuple=True)[0]
    n = len(dead_idx)
    if n == 0 or len(alive_idx) == 0:
        return 0
    src = alive_idx[_sample(weights[alive_idx], n)]
    new_logit, new_log_scale = _split_opacity_and_scale(params, src, binoms, min_opacity)

    def pfn(name, p):
        p = p.clone()
        if name == "opacities":
            p[src] = new_logit
        elif name == "scales":
            p[src] = new_log_scale
        p[dead_idx] = p[src]
        return p

    def sfn(key, v):
        v = v.clone()
        v[src] = 0
        v[dead_idx] = 0
        return v

    _apply(params, optimizers, pfn, sfn)
    if state:
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                v[src] = 0
                v[dead_idx] = 0
    return n


@torch.no_grad()
def sample_add(params: Params, optimizers: Optims, n: int, weights: torch.Tensor,
               binoms: torch.Tensor, min_opacity: float = 0.005, state: dict = None) -> int:
    """Append ``n`` Gaussians copied from sources drawn by ``weights``."""
    if n <= 0:
        return 0
    src = _sample(weights, n)
    new_logit, new_log_scale = _split_opacity_and_scale(params, src, binoms, min_opacity)

    def pfn(name, p):
        p = p.clone()
        if name == "opacities":
            p[src] = new_logit
        elif name == "scales":
            p[src] = new_log_scale
        return torch.cat([p, p[src]])

    def sfn(key, v):
        return torch.cat([v, torch.zeros((n, *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _apply(params, optimizers, pfn, sfn)
    _extend_state(state, lambda v: torch.cat([v, torch.zeros((n, *v.shape[1:]), device=v.device, dtype=v.dtype)]))
    return n


@torch.no_grad()
def inject_position_noise(params: Params, opacity: torch.Tensor, scaler: float) -> None:
    """MCMC exploration noise (Kheradmand et al. 2024, equation 8): Gaussian
    noise shaped by each Gaussian's own covariance, gated so only nearly
    transparent Gaussians move noticeably."""
    from gsplat import quat_scale_to_covar_preci
    covars, _ = quat_scale_to_covar_preci(
        params["quats"].data, torch.exp(params["scales"].data),
        compute_covar=True, compute_preci=False, triu=False)
    gate = torch.sigmoid(100.0 * ((1.0 - opacity) - 0.995))
    noise = torch.randn_like(params["means"].data) * (gate * scaler)[:, None]
    params["means"].data.add_(torch.einsum("nij,nj->ni", covars, noise))
