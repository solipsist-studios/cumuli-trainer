# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Spacetime Gaussians: 3D Gaussians with linear motion and a temporal window.

Model (cumuli docs/sogst-format.md section 1; the polynomial-motion and
temporal radial-basis opacity of Li et al. CVPR 2024, "Spacetime Gaussian
Feature Splatting", section 3, restricted to linear motion and constant
rotation/scale):

    dt       = t - t_center
    mean(t)  = xyz + v * dt
    alpha(t) = sigmoid(opacity) * exp(-0.5 * (dt / t_sigma)^2)

Parameters, one row per Gaussian, in a dict with gsplat's key names:

    means         [N, 3]   position at t_center
    quats         [N, 4]   rotation, wxyz
    scales        [N, 3]   log scale
    opacities     [N]      logit of the peak opacity
    sh0           [N, 1, 3] SH DC
    shN           [N, K-1, 3] higher SH bands
    velocities    [N, 3]   scene units per unit of normalised time
    t_centers     [N]      normalised time
    t_log_sigmas  [N]      log of the normalised temporal std-dev

Time is normalised internally, tau = (t - time_min) / duration, so learning
rates do not depend on clip length. ``to_interchange`` converts back to
seconds. The slice itself is ``cumuli_core.spacetime.slice_at``, the same
function the viewer and evaluator use, applied in normalised units (the
formula is unit-invariant).

How gsplat's strategy ops treat the temporal tensors: duplicate copies every
row, so the copy shares the parent's motion and window. split samples new
positions from the parent's spatial Gaussian, shrinks spatial scales and
repeats every other key, so both children keep the parent's velocity,
t_center and t_sigma; the sampled offset is applied at t_center, which is
where ``means`` lives, so the children stay consistent. relocate and
sample_add copy every key from the source row and only rewrite opacity and
scale, so a relocated Gaussian inherits the source's motion and window and
covers the same span of time. All of these are sensible defaults for a
temporal model and need no special handling.

Initialisation follows the brief: positions, colours and times from the
dataset's points3d.ply, spatial scale from the mean distance to the 3 nearest
neighbours (scipy cKDTree, computed within each timestamp so twelve copies of
a static region do not shrink each other), velocity 0, and
t_sigma = sqrt(duration_seconds / t_init_div).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..core import spacetime as core_spacetime
from ..registry import register
from .base import Representation, RepresentationContext, SlicedGaussians

SH_C0 = 0.28209479177387814


def rgb_to_sh0(rgb):
    return (rgb - 0.5) / SH_C0


def knn_mean_distance(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance from each point to its ``k`` nearest neighbours."""
    from scipy.spatial import cKDTree
    n = len(xyz)
    if n <= 1:
        return np.full(n, 0.01, dtype=np.float32)
    kk = min(k, n - 1)
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz, k=kk + 1)
    d = dist[:, 1:].mean(axis=1)
    pos = d[d > 0]
    floor = float(np.percentile(pos, 1)) if len(pos) else 1e-4
    return np.maximum(d, max(floor, 1e-7)).astype(np.float32)


@register("representation", "spacetime")
class SpacetimeRepresentation(Representation):
    """Parameters (config ``representation:`` section):

    init_opacity      initial peak opacity (0.1)
    init_scale        multiplier on the k-NN distance (1.0)
    init_knn          neighbours for the initial scale (3)
    knn_per_time      compute k-NN within each timestamp (true)
    max_init_points   random subset of points3d.ply to start from (all)
    min_t_sigma_frames  floor on t_sigma in frames, applied in the slice
                      (0 = off). Keeps a window from collapsing between frames.
    max_t_sigma       ceiling on t_sigma in clip durations (10). A static
                      Gaussian keeps widening its window for as long as it
                      trains; at 10 durations the fade across the clip is
                      under 0.5%, so the cap changes nothing visible and keeps
                      the exported sigmas in a range packers quantise well.
    """

    def __init__(self, ctx: RepresentationContext, init_opacity=0.1, init_scale=1.0,
                 init_knn=3, knn_per_time=True, max_init_points=None, min_t_sigma_frames=0.0,
                 max_t_sigma=10.0, seed=0):
        super().__init__(ctx)
        self.init_opacity = float(init_opacity)
        self.init_scale = float(init_scale)
        self.init_knn = int(init_knn)
        self.knn_per_time = bool(knn_per_time)
        self.max_init_points = None if max_init_points is None else int(max_init_points)
        self.seed = int(seed)
        self.duration = max(ctx.time_max - ctx.time_min, 1e-6)
        frame = 1.0 / ctx.fps if ctx.fps else 0.0
        self.min_log_sigma = (math.log(min_t_sigma_frames * frame / self.duration)
                              if min_t_sigma_frames > 0 and frame > 0 else None)
        self.max_log_sigma = math.log(max_t_sigma) if max_t_sigma and max_t_sigma > 0 else None

    # -- time conversion --------------------------------------------------
    def to_tau(self, t_seconds):
        return (t_seconds - self.ctx.time_min) / self.duration

    def initial_log_sigma(self) -> float:
        sigma_seconds = math.sqrt(self.duration / self.ctx.t_init_div)
        return math.log(sigma_seconds / self.duration)

    # -- construction -----------------------------------------------------
    def _alloc(self, xyz, rgb, tau, log_scales, log_sigma, quats=None, opacity_logit=None,
               velocities=None, shN=None):
        dev = self.ctx.device
        n = len(xyz)
        k = (self.ctx.sh_degree + 1) ** 2
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
        if quats is None:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            q = torch.randn((n, 4), generator=g)
            quats = torch.nn.functional.normalize(q, dim=-1).numpy()
        if opacity_logit is None:
            p = self.init_opacity
            opacity_logit = np.full(n, math.log(p / (1 - p)), dtype=np.float32)
        if velocities is None:
            velocities = np.zeros((n, 3), dtype=np.float32)
        sh0 = rgb_to_sh0(np.asarray(rgb, dtype=np.float32))[:, None, :]
        shn = np.zeros((n, k - 1, 3), dtype=np.float32)
        if shN is not None and k > 1:
            m = min(shN.shape[1], k - 1)
            shn[:, :m] = shN[:, :m]
        if np.ndim(log_sigma) == 0:
            log_sigma = np.full(n, float(log_sigma), dtype=np.float32)
        P = torch.nn.Parameter
        self._set_params({
            "means": P(t(xyz)),
            "quats": P(t(quats)),
            "scales": P(t(log_scales)),
            "opacities": P(t(opacity_logit)),
            "sh0": P(t(sh0)),
            "shN": P(t(shn)),
            "velocities": P(t(velocities)),
            "t_centers": P(t(tau)),
            "t_log_sigmas": P(t(log_sigma)),
        })

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
        """Seed from a previous interchange PLY. Times are re-expressed in this
        run's window; positions stay at their own t_center."""
        f = {k: np.asarray(v) for k, v in fields.items()}
        xyz = np.stack([f["x"], f["y"], f["z"]], 1)
        quats = np.stack([f[f"rot_{i}"] for i in range(4)], 1)
        log_scales = np.stack([f[f"scale_{i}"] for i in range(3)], 1)
        dc = np.stack([f[f"f_dc_{i}"] for i in range(3)], 1)
        rgb = dc * SH_C0 + 0.5
        vel_sec = np.stack([f["vx"], f["vy"], f["vz"]], 1)
        tau = self.to_tau(f["t_center"].astype(np.float64)).astype(np.float32)
        log_sigma = np.log(np.maximum(f["t_sigma"], 1e-8) / self.duration).astype(np.float32)
        shN = None
        if "f_rest" in f:
            width = f["f_rest"].shape[1]
            c = width // 3
            shN = f["f_rest"].reshape(-1, 3, c).transpose(0, 2, 1)
        self._alloc(xyz, rgb, tau, log_scales, log_sigma, quats=quats,
                    opacity_logit=f["opacity"], velocities=vel_sec * self.duration, shN=shN)

    # -- training ---------------------------------------------------------
    def learning_rates(self, lr_cfg: dict) -> dict:
        s = self.ctx.scene_scale
        return {
            "means": lr_cfg["means"] * s,
            "quats": lr_cfg["quats"],
            "scales": lr_cfg["scales"],
            "opacities": lr_cfg["opacities"],
            "sh0": lr_cfg["sh0"],
            "shN": lr_cfg["shN"],
            "velocities": lr_cfg["velocities"] * s,
            "t_centers": lr_cfg["t_centers"],
            "t_log_sigmas": lr_cfg["t_log_sigmas"],
        }

    def decayed_params(self) -> tuple:
        return ("means", "velocities")

    def _sigmas(self):
        ls = self.params["t_log_sigmas"]
        if self.min_log_sigma is not None or self.max_log_sigma is not None:
            ls = ls.clamp(min=self.min_log_sigma, max=self.max_log_sigma)
        return torch.exp(ls)

    def slice(self, t: float, sh_degree: int) -> SlicedGaussians:
        p = self.params
        tau = self.to_tau(float(t))
        means, alpha = core_spacetime.slice_at(
            p["means"], p["velocities"], p["opacities"], p["t_centers"], self._sigmas(), tau)
        colors = torch.cat([p["sh0"], p["shN"]], dim=1)
        return SlicedGaussians(
            means=means, quats=p["quats"], scales=torch.exp(p["scales"]),
            opacities=alpha, colors=colors, sh_degree=sh_degree)

    @torch.no_grad()
    def peak_opacity(self) -> torch.Tensor:
        """Peak opacity reached inside [time_min, time_max] (tau in [0, 1])."""
        tc = self.params["t_centers"]
        gap = torch.clamp(torch.maximum(-tc, tc - 1.0), min=0.0)
        return torch.sigmoid(self.params["opacities"]) * torch.exp(-0.5 * (gap / self._sigmas()) ** 2)

    # -- output -----------------------------------------------------------
    @torch.no_grad()
    def to_interchange(self, min_peak_opacity: float = 0.0, keep=None) -> dict:
        """Interchange fields in seconds. Gaussians whose peak opacity inside
        the window is below ``min_peak_opacity``, or where ``keep`` is False,
        are left out."""
        if min_peak_opacity > 0:
            visible = self.peak_opacity() >= min_peak_opacity
            keep = visible if keep is None else (keep & visible)
        p = {k: (v.detach()[keep] if keep is not None else v.detach()).float().cpu().numpy()
             for k, v in self.params.items()}
        d = self.duration
        fields = {}
        for i, c in enumerate("xyz"):
            fields[c] = p["means"][:, i]
        q = p["quats"] / np.maximum(np.linalg.norm(p["quats"], axis=1, keepdims=True), 1e-12)
        for i in range(4):
            fields[f"rot_{i}"] = q[:, i]
        for i in range(3):
            fields[f"scale_{i}"] = p["scales"][:, i]
        fields["opacity"] = p["opacities"]
        for i in range(3):
            fields[f"f_dc_{i}"] = p["sh0"][:, 0, i]
        vel = p["velocities"] / d
        fields["vx"], fields["vy"], fields["vz"] = vel[:, 0], vel[:, 1], vel[:, 2]
        fields["t_center"] = self.ctx.time_min + p["t_centers"].astype(np.float64) * d
        ls = p["t_log_sigmas"]
        if self.min_log_sigma is not None:
            ls = np.maximum(ls, self.min_log_sigma)
        if self.max_log_sigma is not None:
            ls = np.minimum(ls, self.max_log_sigma)
        fields["t_sigma"] = np.exp(ls.astype(np.float64)) * d
        shn = p["shN"]
        if shn.shape[1] > 0:
            # [N, K-1, 3] -> channel-major [N, 3*(K-1)]: column j*c + k
            fields["f_rest"] = shn.transpose(0, 2, 1).reshape(len(shn), -1)
        return fields
