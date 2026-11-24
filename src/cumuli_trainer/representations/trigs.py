# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""TRiGS: 3D Gaussians whose motion is a Bezier twist about an anchor, with a temporal window.

The model is the one of the TRiGS paper (arXiv 2604.00538, Sec. 3), evaluated by
``cumuli_core.trigs.slice_at`` (the same function the evaluator and the viewer's parity test use).
It is written from the paper and from the tensor names the MIT Open4DGS-Core packager reads; no
trainer code of Open4DGS was read.  Where the paper says nothing the choice is ours and is marked.

Parameters, one row per Gaussian, with gsplat's key names so the strategy ops move whole rows:

    means, quats (wxyz), scales (log), opacities (logit), sh0, shN     as in ``spacetime``
    t_centers       [N]     time centre, normalised time
    t_log_sigmas    [N]     log of the temporal sigma, normalised time
    anchor_offsets  [N, 3]  anchor minus mean (ours: a relative anchor keeps the pivot with its Gaussian when
                            positions move, relocate or take MCMC noise; the exported anchor is absolute)
    zeta_base_r/t, zeta_c0..c2_r/t   [N, 3]   the base twist and three Bezier control twists, (omega, nu)

Time is normalised to [0, 1] over the clip, as in ``spacetime``, so learning rates do not depend on
clip length.  The model equations are unit-free, so slicing in normalised time is exact; export divides
the twists by the clip duration to give radians and scene units per second.

Ours: the initial state is no motion with a small random rotation axis per Gaussian
(``init_omega`` radians per unit of normalised time).  The gauge-fixed translation of the paper's
Eq. 10 keeps only the part of nu along the rotation axis, so an axis of exactly zero would give no
axis at all; a small random one lets the first gradients move a Gaussian along it and turn the axis
toward the motion the images want.  Position noise, relocation and the opacity floor are the
strategy's (``mcmc``: the paper's fixed budget with relocation of faint Gaussians).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..registry import register
from .base import Representation, RepresentationContext, SlicedGaussians
from .spacetime import SH_C0, knn_mean_distance, rgb_to_sh0

ZETA_KEYS = ("zeta_base_r", "zeta_base_t", "zeta_c0_r", "zeta_c0_t", "zeta_c1_r", "zeta_c1_t", "zeta_c2_r", "zeta_c2_t")
#: Twist parameters that are scene units per time (translation); the others are radians per time.
TRANSLATION_KEYS = tuple(k for k in ZETA_KEYS if k.endswith("_t"))
ROTATION_KEYS = tuple(k for k in ZETA_KEYS if k.endswith("_r"))


def _trigs():
    from cumuli_core import trigs   # lazy: a cumuli_core without it still runs every other representation
    return trigs


@register("representation", "trigs")
class TrigsRepresentation(Representation):
    """Parameters (config ``representation:`` section):

    init_opacity        initial peak opacity (0.1)
    init_scale          multiplier on the k-NN distance (1.0)
    init_knn            neighbours for the initial scale (3)
    knn_per_time        compute k-NN within each timestamp (true)
    max_init_points     random subset of points3d.ply to start from (all)
    max_t_sigma         ceiling on t_sigma in clip durations (10)
    min_t_sigma_frames  floor on t_sigma in frames (0 = off)
    init_omega          size of the initial rotation axis, radians per normalised time (0.05)
    """

    def __init__(self, ctx: RepresentationContext, init_opacity=0.1, init_scale=1.0, init_knn=3, knn_per_time=True,
                 max_init_points=None, min_t_sigma_frames=0.0, max_t_sigma=10.0, init_omega=0.05, seed=0):
        super().__init__(ctx)
        self.init_opacity = float(init_opacity)
        self.init_scale = float(init_scale)
        self.init_knn = int(init_knn)
        self.knn_per_time = bool(knn_per_time)
        self.max_init_points = None if max_init_points is None else int(max_init_points)
        self.init_omega = float(init_omega)
        self.seed = int(seed)
        self.duration = max(ctx.time_max - ctx.time_min, 1e-6)
        frame = 1.0 / ctx.fps if ctx.fps else 0.0
        self.min_log_sigma = (math.log(min_t_sigma_frames * frame / self.duration)
                              if min_t_sigma_frames > 0 and frame > 0 else None)
        self.max_log_sigma = math.log(max_t_sigma) if max_t_sigma and max_t_sigma > 0 else None

    # -- time -------------------------------------------------------------
    def to_tau(self, t_seconds):
        return (t_seconds - self.ctx.time_min) / self.duration

    def initial_log_sigma(self) -> float:
        return math.log(math.sqrt(self.duration / self.ctx.t_init_div) / self.duration)

    # -- construction -----------------------------------------------------
    def _alloc(self, xyz, rgb, tau, log_scales, log_sigma, quats=None, opacity_logit=None, shN=None, zeta=None, anchor_offsets=None):
        dev = self.ctx.device
        n = len(xyz)
        k = (self.ctx.sh_degree + 1) ** 2
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        if quats is None:
            quats = torch.nn.functional.normalize(torch.randn((n, 4), generator=g), dim=-1).numpy()
        if opacity_logit is None:
            p = self.init_opacity
            opacity_logit = np.full(n, math.log(p / (1 - p)), dtype=np.float32)
        sh0 = rgb_to_sh0(np.asarray(rgb, dtype=np.float32))[:, None, :]
        shn = np.zeros((n, k - 1, 3), dtype=np.float32)
        if shN is not None and k > 1:
            m = min(shN.shape[1], k - 1)
            shn[:, :m] = shN[:, :m]
        if np.ndim(log_sigma) == 0:
            log_sigma = np.full(n, float(log_sigma), dtype=np.float32)
        P = torch.nn.Parameter
        params = {"means": P(t(xyz)), "quats": P(t(quats)), "scales": P(t(log_scales)), "opacities": P(t(opacity_logit)),
                  "sh0": P(t(sh0)), "shN": P(t(shn)), "t_centers": P(t(tau)), "t_log_sigmas": P(t(log_sigma)),
                  "anchor_offsets": P(t(np.zeros((n, 3), dtype=np.float32) if anchor_offsets is None else anchor_offsets))}
        for key in ZETA_KEYS:
            params[key] = P(t(np.zeros((n, 3), dtype=np.float32) if zeta is None else zeta[key]))
        if zeta is None and self.init_omega > 0:
            axis = torch.nn.functional.normalize(torch.randn((n, 3), generator=g), dim=-1).numpy()
            params["zeta_base_r"] = P(t(axis * self.init_omega))
        self._set_params(params)

    def init_from_points(self, points: dict) -> None:
        xyz = np.asarray(points["xyz"], dtype=np.float32)
        rgb = np.asarray(points["rgb"], dtype=np.float32)
        time = points.get("time")
        rng = np.random.default_rng(self.seed)
        if time is None:
            time = rng.uniform(self.ctx.time_min, self.ctx.time_max, size=len(xyz)).astype(np.float32)
        time = np.asarray(time, dtype=np.float32)
        if self.max_init_points and len(xyz) > self.max_init_points:
            keep = np.sort(rng.choice(len(xyz), self.max_init_points, replace=False))
            xyz, rgb, time = xyz[keep], rgb[keep], time[keep]
        dist = np.empty(len(xyz), dtype=np.float32)
        uniq = np.unique(time)
        if self.knn_per_time and 1 < len(uniq) <= 10000:
            for value in uniq:
                sel = np.nonzero(time == value)[0]
                dist[sel] = knn_mean_distance(xyz[sel], self.init_knn)
        else:
            dist = knn_mean_distance(xyz, self.init_knn)
        log_scales = np.log(dist * self.init_scale)[:, None].repeat(3, axis=1)
        tau = self.to_tau(time.astype(np.float64)).astype(np.float32)
        self._alloc(xyz, rgb, tau, log_scales, self.initial_log_sigma())

    def init_from_interchange(self, header: dict, fields: dict) -> None:
        """Seed from a spacetime interchange PLY: positions, colours and windows carry over; the twists start
        from the PLY's velocity as a translation along a rotation axis pointing the same way (so the motion
        starts as the PLY had it), with no rotation to speak of."""
        f = {k: np.asarray(v) for k, v in fields.items()}
        xyz = np.stack([f["x"], f["y"], f["z"]], 1)
        quats = np.stack([f[f"rot_{i}"] for i in range(4)], 1)
        log_scales = np.stack([f[f"scale_{i}"] for i in range(3)], 1)
        dc = np.stack([f[f"f_dc_{i}"] for i in range(3)], 1)
        rgb = dc * SH_C0 + 0.5
        tau = self.to_tau(f["t_center"].astype(np.float64)).astype(np.float32)
        log_sigma = np.log(np.maximum(f["t_sigma"], 1e-8) / self.duration).astype(np.float32)
        shN = None
        if "f_rest" in f:
            c = f["f_rest"].shape[1] // 3
            shN = f["f_rest"].reshape(-1, 3, c).transpose(0, 2, 1)
        vel = np.stack([f["vx"], f["vy"], f["vz"]], 1) * self.duration       # per normalised time
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        axis = np.where(speed > 1e-12, vel / np.maximum(speed, 1e-12), np.array([[0.0, 0.0, 1.0]]))
        zeta = {k: np.zeros_like(xyz) for k in ZETA_KEYS}
        zeta["zeta_base_r"] = (axis * max(self.init_omega, 1e-3)).astype(np.float32)
        zeta["zeta_base_t"] = vel.astype(np.float32)
        self._alloc(xyz, rgb, tau, log_scales, log_sigma, quats=quats, opacity_logit=f["opacity"], shN=shN, zeta=zeta)

    # -- training ---------------------------------------------------------
    def learning_rates(self, lr_cfg: dict) -> dict:
        s = self.ctx.scene_scale
        lrs = {"means": lr_cfg["means"] * s, "quats": lr_cfg["quats"], "scales": lr_cfg["scales"],
               "opacities": lr_cfg["opacities"], "sh0": lr_cfg["sh0"], "shN": lr_cfg["shN"],
               "t_centers": lr_cfg["t_centers"], "t_log_sigmas": lr_cfg["t_log_sigmas"],
               "anchor_offsets": lr_cfg["anchors"] * s}
        for key in ROTATION_KEYS:
            lrs[key] = lr_cfg["zeta_r"]
        for key in TRANSLATION_KEYS:
            lrs[key] = lr_cfg["zeta_t"] * s
        return lrs

    def decayed_params(self) -> tuple:
        return ("means", "anchor_offsets", *TRANSLATION_KEYS)

    def _log_sigmas(self):
        ls = self.params["t_log_sigmas"]
        if self.min_log_sigma is not None or self.max_log_sigma is not None:
            ls = ls.clamp(min=self.min_log_sigma, max=self.max_log_sigma)
        return ls

    def _tensors(self):
        p = self.params
        return _trigs().TrigsTensors(
            means=p["means"], quats_wxyz=p["quats"], log_scales=p["scales"], opacity_logit=p["opacities"], sh=p["sh0"],
            times=p["t_centers"], log_sigma=self._log_sigmas(), anchors=p["means"] + p["anchor_offsets"],
            **{k: p[k] for k in ZETA_KEYS})

    def slice(self, t: float, sh_degree: int) -> SlicedGaussians:
        # A tensor time stays a tensor, so learned capture-time corrections get a gradient.
        tau = self.to_tau(t if torch.is_tensor(t) else float(t))
        means, quats, alpha = _trigs().slice_at(self._tensors(), tau)
        p = self.params
        return SlicedGaussians(means=means, quats=quats, scales=torch.exp(p["scales"]), opacities=alpha,
                               colors=torch.cat([p["sh0"], p["shN"]], dim=1), sh_degree=sh_degree)

    def means_at(self, t) -> torch.Tensor:
        tau = self.to_tau(t if torch.is_tensor(t) else float(t))
        return _trigs().slice_at(self._tensors(), tau)[0]

    @torch.no_grad()
    def peak_opacity(self) -> torch.Tensor:
        """Peak opacity reached inside [time_min, time_max] (tau in [0, 1])."""
        tc = self.params["t_centers"]
        gap = torch.clamp(torch.maximum(-tc, tc - 1.0), min=0.0)
        return torch.sigmoid(self.params["opacities"]) * torch.exp(-0.5 * (gap / torch.exp(self._log_sigmas())) ** 2)

    # -- output -----------------------------------------------------------
    def to_interchange(self, min_peak_opacity: float = 0.0, keep=None) -> dict:
        raise NotImplementedError("the trigs representation exports trigs_ckpt.pt (export_checkpoint), not an interchange PLY")

    @torch.no_grad()
    def export_checkpoint(self, model_path: str, step: int, final: bool, min_peak_opacity: float = 0.0, keep=None) -> tuple[str, int]:
        """Write the model in the layout Open4DGS-Core's packager reads (``trigs_ckpt.pt``), or, before the last
        step, a snapshot ``trigs_<step>.pt`` of the same layout.  Seconds, not normalised time; quaternions xyzw.
        Gaussians whose peak opacity inside the clip is under ``min_peak_opacity``, or where ``keep`` is False,
        are left out.  Returns (path, how many Gaussians were written)."""
        import os

        if min_peak_opacity > 0:
            visible = self.peak_opacity() >= min_peak_opacity
            keep = visible if keep is None else (keep & visible)
        sel = (lambda v: v.detach()[keep]) if keep is not None else (lambda v: v.detach())
        d = self.duration
        p = {k: sel(v).float().cpu() for k, v in self.params.items()}
        ls = sel(self._log_sigmas()).float().cpu()
        q = torch.nn.functional.normalize(p["quats"], dim=-1)
        params = {"means": p["means"], "scales": p["scales"], "quats": torch.cat([q[:, 1:], q[:, :1]], dim=1),   # wxyz -> xyzw
                  "opacities": p["opacities"], "sh0": p["sh0"], "shN": p["shN"],
                  "times": p["t_centers"] * d, "log_durations": ls + math.log(d),
                  "anchors": p["means"] + p["anchor_offsets"]}
        for key in ZETA_KEYS:
            params[key] = p[key] / d
        n = int(params["means"].shape[0])
        n_frames = int(round(d * self.ctx.fps)) + 1 if self.ctx.fps else None
        ckpt = {"params": {k: v.contiguous() for k, v in params.items()}, "step": int(step), "fps": float(self.ctx.fps),
                "n_frames": n_frames,
                "metrics": {"windows": [[0.0, d]], "ownership_bounds": [0.0, d], "time_origin": float(self.ctx.time_min)}}
        name = "trigs_ckpt.pt" if final else f"trigs_{step}.pt"
        path = os.path.join(model_path, name)
        torch.save(ckpt, path)
        return path, n
