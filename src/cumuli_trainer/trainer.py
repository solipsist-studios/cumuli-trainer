# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""The training loop.

The loop knows four plugin interfaces and nothing else: it asks the
representation for Gaussians at an instant, the renderer for images, the loss
terms for a scalar and the strategy to edit the population. Which
representation, strategy, losses or renderer run is decided by the config
through the registry.

One step:

    t, cams, gt = data.next()                       one instant, C cameras
    g = representation.slice(t, sh_degree)          slice once
    rgb, alpha, info = renderer.render(g, cams)     one gsplat call for C views
    composite rgb and gt onto the background
    loss = sum(weight * term(inputs))
    loss.backward(); optimizers step
    strategy.step_post_backward(...)                densify / prune / relocate

There are no host synchronisations in a step except every ``log_every``
steps (progress bar) and on the strategy's refine steps.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

from . import load_builtin_plugins, progress, registry
from .config import dump, fps_from_times, loss_specs
from .core import CORE_VERSION
from .core import dataset as core_dataset
from .core import ply as core_ply
from .data import TrainData
from .evaluate import EvalSet
from .losses.base import LossInputs
from .representations.base import RepresentationContext
from .strategies.base import StrategyContext

BACKGROUNDS = {"black": [0.0, 0.0, 0.0], "white": [1.0, 1.0, 1.0]}


def scene_scale_from_cameras(cams) -> float:
    """1.1 x the largest distance of a camera centre from their mean, as in
    the 3DGS reference setup (Kerbl et al. 2023, section 5.2, 'extent')."""
    centres = np.stack([np.linalg.inv(np.asarray(c.w2c))[:3, 3] for c in cams])
    d = np.linalg.norm(centres - centres.mean(axis=0), axis=1)
    return float(max(d.max(), 1e-6) * 1.1)


def view_contribution(means, quats, scales, opacities, viewmat, K, width, height) -> torch.Tensor:
    """[N] each Gaussian's total blend weight in one view, in pixels: the
    gradient of the rendered sum with respect to a per-Gaussian colour of 1."""
    from gsplat import rasterization  # the default renderer's backend
    with torch.enable_grad():
        w = torch.ones(means.shape[0], 1, device=means.device, requires_grad=True)
        img, _, _ = rasterization(means, quats, scales, opacities, w, viewmat, K,
                                  width, height, sh_degree=None)
        img.sum().backward()
    return w.grad[:, 0]


def camera_centroid(cams) -> np.ndarray:
    """Mean camera centre: the centre of the scene bounds."""
    return np.stack([np.linalg.inv(np.asarray(c.w2c))[:3, 3] for c in cams]).mean(axis=0)


class Trainer:
    def __init__(self, cfg: dict, resume: str | None = None, log=None):
        load_builtin_plugins()
        self.cfg = cfg
        self.log = log or (lambda msg: print(msg, flush=True))
        self.resume_path = resume or cfg.get("resume")
        if not torch.cuda.is_available():
            raise RuntimeError("cumuli-train needs a CUDA device (gsplat rasterisation)")
        self.device = torch.device("cuda")
        self._seed(cfg["seed"])
        os.makedirs(cfg["model_path"], exist_ok=True)

        self.log(f"Found transforms_train.json in {cfg['source_path']}")
        self.dataset = core_dataset.Dataset4D(cfg["source_path"])
        train_cams = self.dataset.train_cameras
        if cfg["time_duration"] is None:
            cfg["time_duration"] = [self.dataset.time_min, self.dataset.time_max]
        if cfg["fps"] is None:
            cfg["fps"] = fps_from_times([c.time for c in train_cams])
        self.time_min, self.time_max = cfg["time_duration"]
        self.scene_scale = scene_scale_from_cameras(train_cams)
        self.scene_center = camera_centroid(train_cams)
        self.bound_radius = float(cfg["scene_bound_factor"]) * self.scene_scale
        self.log(f"cumuli-core {CORE_VERSION}")
        self.log(f"time window [{self.time_min}, {self.time_max}] s at {cfg['fps']} fps; "
                 f"scene scale {self.scene_scale:.4f}; bounds "
                 + (f"radius {self.bound_radius:.3f} around the camera centroid"
                    if self.bound_radius > 0 else "off"))

        # -- representation
        rep_cfg = dict(cfg["representation"])
        rep_name = rep_cfg.pop("name")
        rep_cls = registry.get("representation", rep_name)
        ctx = RepresentationContext(
            time_min=self.time_min, time_max=self.time_max, fps=cfg["fps"],
            scene_scale=self.scene_scale, sh_degree=cfg["sh_degree"],
            t_init_div=cfg["t_init_div"], device=self.device)
        if "seed" not in rep_cfg and "seed" in rep_cls.__init__.__code__.co_varnames:
            rep_cfg["seed"] = cfg["seed"]
        self.rep = rep_cls(ctx, **rep_cfg)

        ckpt = None
        self.start_step = 1
        if self.resume_path:
            ckpt = torch.load(self.resume_path, map_location=self.device, weights_only=False)
            self.rep.load_state_dict(ckpt["params"])
            self.rep.load_extra_state(ckpt.get("representation_extra"))
            self.start_step = int(ckpt["iteration"]) + 1
            self.log(f"Resuming from {self.resume_path} at iteration {ckpt['iteration']} "
                     f"with {self.rep.num_gaussians:,} Gaussians")
        elif cfg["init_ply"]:
            header, fields = core_ply.read_sogst_ply(cfg["init_ply"])
            self.rep.init_from_interchange(header, fields)
            self.log(f"Number of points at initialisation: {self.rep.num_gaussians:,} "
                     f"(from {cfg['init_ply']})")
        else:
            self.rep.init_from_points(self.dataset.load_points())
            self.log(f"Number of points at initialisation: {self.rep.num_gaussians:,}")

        # -- optimisers: one Adam per parameter (gsplat's convention, which its
        # strategy ops rely on)
        self.lr_init = self.rep.learning_rates(cfg["lr"])
        self.decayed = set(self.rep.decayed_params())
        self.optimizers = {
            name: torch.optim.Adam([{"params": [p], "lr": self.lr_init[name], "name": name}],
                                   eps=1e-15, betas=(0.9, 0.999), fused=True)
            for name, p in self.rep.params.items()
        }
        if ckpt is not None:
            for name, opt in self.optimizers.items():
                if name in ckpt["optimizers"]:
                    opt.load_state_dict(ckpt["optimizers"][name])

        # -- strategy
        st_cfg = dict(cfg["strategy"])
        st_name = st_cfg.pop("name")
        self.strategy = registry.create("strategy", st_name, **st_cfg)
        if ckpt is not None and "strategy" in ckpt:
            self.strategy.load_state_dict(ckpt["strategy"])
        self.strategy.initialize(self.rep.params, self.optimizers, StrategyContext(
            scene_scale=self.scene_scale, iterations=cfg["iterations"], device=self.device,
            opacity=lambda: torch.sigmoid(self.rep.params["opacities"]).detach(),
            in_bounds=self.in_bounds if self.bound_radius > 0 else None))
        self.log(f"strategy: {self.strategy.describe()}")

        # -- renderer
        rd_cfg = dict(cfg["renderer"])
        self.renderer = registry.create("renderer", rd_cfg.pop("name"), **rd_cfg)
        self.renderer.configure(absgrad=bool(getattr(self.strategy, "wants_absgrad", False)))

        # -- losses
        self.losses = []
        for term, weight, params in loss_specs(cfg, getattr(self.strategy, "default_losses", {})):
            self.losses.append((term, weight, registry.create("loss", term, **params)))
        self.log("losses: " + ", ".join(f"{t}={w:g}" for t, w, _ in self.losses))

        # -- data
        self.data = TrainData(self.dataset, train_cams, self.device, cfg["cameras_per_step"],
                              cfg["data_device"], seed=cfg["seed"], log=self.log)
        if ckpt is not None:
            self.data.load_state_dict(ckpt.get("data"))
            if "rng" in ckpt:
                self._restore_rng(ckpt["rng"])
        bg = cfg["background"]
        self.random_bg = bg == "random"
        self.bg = None if self.random_bg else torch.tensor(
            BACKGROUNDS.get(bg, bg) if isinstance(bg, str) else bg, dtype=torch.float32,
            device=self.device)
        test_bg = [0.0, 0.0, 0.0] if self.random_bg else (BACKGROUNDS.get(bg, bg) if isinstance(bg, str) else bg)
        test_cams = self.dataset.test_cameras[::max(1, int(cfg["test_every"]))]
        self.evalset = EvalSet(self.dataset, test_cams, self.device, test_bg) if test_cams else None
        if self.evalset is None:
            self.log("no transforms_test.json cameras: test metrics are skipped")

        with open(os.path.join(cfg["model_path"], "config.json"), "w") as fp:
            fp.write(dump(cfg))
        self.history = []

    # ------------------------------------------------------------------
    def _seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _rng_state(self) -> dict:
        return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                "numpy": np.random.get_state(), "python": random.getstate()}

    def _restore_rng(self, st: dict) -> None:
        torch.set_rng_state(st["torch"].cpu())
        torch.cuda.set_rng_state(st["cuda"].cpu())
        np.random.set_state(st["numpy"])
        random.setstate(st["python"])

    def lr_at(self, name: str, step: int) -> float:
        lr = self.lr_init[name]
        if name in self.decayed:
            ratio = self.cfg["lr"]["means_final_ratio"]
            lr = lr * ratio ** (min(step, self.cfg["iterations"]) / self.cfg["iterations"])
        return lr

    def sh_degree_at(self, step: int) -> int:
        return min(self.cfg["sh_degree"], step // max(1, int(self.cfg["sh_degree_interval"])))

    # ------------------------------------------------------------------
    def train_step(self, step: int):
        cfg = self.cfg
        for name in self.decayed:
            if name in self.optimizers:
                self.optimizers[name].param_groups[0]["lr"] = self.lr_at(name, step)

        batch = self.data.next()
        idx, imgs = batch.index, batch.images
        g = self.rep.slice(batch.time, self.sh_degree_at(step))
        rgb, alpha, info = self.renderer.render(
            g, self.data.viewmats[idx], self.data.Ks[idx], self.data.width, self.data.height)
        gt = imgs.float().div_(255.0)
        gt_alpha = gt[..., 3:4]
        gt_rgb = gt[..., :3] * gt_alpha
        if self.random_bg:
            bg = torch.rand(3, device=self.device)
        else:
            bg = self.bg
        if self.random_bg or bool(cfg["background"] != "black"):
            rgb = rgb + (1.0 - alpha) * bg
            gt_rgb = gt_rgb + (1.0 - gt_alpha) * bg

        self.strategy.step_pre_backward(self.rep.params, self.optimizers, step, info)
        inputs = LossInputs(pred_rgb=rgb, pred_alpha=alpha, gt_rgb=gt_rgb, gt_alpha=gt_alpha,
                            params=self.rep.params, gaussians=g, info=info, step=step,
                            extras={"crop": (batch.crop_size, batch.crop_origins)})
        loss = None
        for _, weight, term in self.losses:
            value = weight * term(inputs)
            loss = value if loss is None else loss + value
        loss.backward()
        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        self.strategy.step_post_backward(self.rep.params, self.optimizers, step, info,
                                         lr=self.lr_at("means", step))
        with torch.no_grad():
            mse = ((rgb.detach() - gt_rgb) ** 2).mean()
        return loss.detach(), mse

    # ------------------------------------------------------------------
    @torch.no_grad()
    def in_bounds(self) -> torch.Tensor:
        """[N] True where a Gaussian's position (at its t_center) lies within
        bound_radius of the camera centroid."""
        means = self.rep.params["means"]
        centre = torch.as_tensor(self.scene_center, dtype=means.dtype, device=means.device)
        return (means - centre).norm(dim=1) <= self.bound_radius

    def max_contribution(self) -> torch.Tensor:
        """[N] the largest total blend weight, in pixels, that any single
        training view gives each Gaussian: sum over that view's pixels of
        transmittance x alpha. It is the gradient of the rendered image sum
        with respect to a per-Gaussian colour fixed at 1, so one backward pass
        per view gives every Gaussian's figure at once.

        Gaussians near zero are hidden from every training camera, inside the
        body or behind the surface, so their colour never receives a
        correction. On Heidi 23.6% of Gaussians stayed under 0.1 pixel in
        every view, including 11.7k opaque ones with saturated off-palette
        colours that the viewer showed as coloured glints from close up."""
        n = self.rep.num_gaussians
        best = torch.zeros(n, device=self.device)
        sliced = {}
        with torch.enable_grad():
            for i, cam in enumerate(self.data.cams):
                if cam.time not in sliced:
                    sliced.clear()
                    g = self.rep.slice(cam.time, 0)
                    sliced[cam.time] = (g.means.detach(), g.quats.detach(),
                                        g.scales.detach(), g.opacities.detach())
                means, quats, scales, opac = sliced[cam.time]
                best = torch.maximum(best, view_contribution(
                    means, quats, scales, opac, self.data.viewmats[i:i + 1],
                    self.data.Ks[i:i + 1], self.data.width, self.data.height))
        return best

    def save_ply(self, step: int) -> str:
        path = os.path.join(self.cfg["model_path"], "splat_4d.ply")
        print(progress.saving_line(step), flush=True)
        keep = self.in_bounds() if self.bound_radius > 0 else None
        min_contrib = float(self.cfg["export_min_contribution"])
        if min_contrib > 0:
            seen = self.max_contribution() >= min_contrib
            self.log(f"visibility: {int((~seen).sum()):,} of {self.rep.num_gaussians:,} Gaussians "
                     f"contribute under {min_contrib:g} px to every training view; dropped")
            keep = seen if keep is None else (keep & seen)
        fields = self.rep.to_interchange(float(self.cfg["export_min_opacity"]), keep=keep)
        n = core_ply.write_sogst_ply(path, fields, self.time_min, self.time_max,
                                     self.cfg["fps"], generator="cumuli-trainer")
        self.log(f"wrote {path}: {n:,} of {self.rep.num_gaussians:,} Gaussians "
                 f"(peak opacity >= {float(self.cfg['export_min_opacity']):.4g}, inside bounds, seen)")
        self.exported_count = n
        if step != self.cfg["iterations"]:
            snap = os.path.join(self.cfg["model_path"], f"splat_4d_{step}.ply")
            core_ply.write_sogst_ply(snap, fields, self.time_min, self.time_max,
                                     self.cfg["fps"], generator="cumuli-trainer")
        return path

    def save_checkpoint(self, step: int) -> str:
        path = os.path.join(self.cfg["model_path"], f"ckpt_{step}.pt")
        torch.save({
            "iteration": step,
            "config": self.cfg,
            "params": self.rep.state_dict(),
            "representation_extra": self.rep.extra_state(),
            "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
            "strategy": self.strategy.state_dict(),
            "data": self.data.state_dict(),
            "rng": self._rng_state(),
        }, path)
        return path

    def evaluate(self, step: int, bar=None) -> dict | None:
        if self.evalset is None:
            return None
        res = self.evalset.evaluate(self.rep, self.renderer, self.cfg["sh_degree"])
        m = res["mean"]
        line = progress.test_line(step, m["psnr_db"], m["ssim"], m["lpips"], len(res["views"]))
        (bar.write if bar is not None else print)(line)
        sys.stdout.flush()
        res["iteration"] = step
        res["num_gaussians"] = self.rep.num_gaussians
        self.history.append({"iteration": step, **m, "num_gaussians": self.rep.num_gaussians})
        with open(os.path.join(self.cfg["model_path"], f"test_{step}.json"), "w") as fp:
            json.dump(res, fp, indent=2)
        return res

    # ------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        total = cfg["iterations"]
        log_every = max(1, int(cfg["log_every"]))
        bar = progress.make_bar(total, initial=self.start_step - 1)
        ema_loss = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        eval_seconds = 0.0
        last_logged = self.start_step - 1
        for step in range(self.start_step, total + 1):
            loss, mse = self.train_step(step)
            if step % log_every == 0 or step == total:
                lv = float(loss)
                psnr = -10.0 * math.log10(max(float(mse), 1e-12))
                ema_loss = lv if ema_loss is None else 0.6 * lv + 0.4 * ema_loss
                bar.set_postfix(progress.postfix(ema_loss, psnr), refresh=False)
                bar.update(step - last_logged)
                last_logged = step
            if step in cfg["test_iterations"]:
                torch.cuda.synchronize()
                te = time.perf_counter()
                self.evaluate(step, bar)
                eval_seconds += time.perf_counter() - te
            if step in cfg["save_iterations"]:
                bar.clear()
                self.save_ply(step)
            if step in cfg["checkpoint_iterations"]:
                self.save_checkpoint(step)
        bar.close()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        summary = {
            "iterations": total,
            "train_seconds": wall - eval_seconds,
            "eval_seconds": eval_seconds,
            "num_gaussians": self.rep.num_gaussians,
            "exported_gaussians": getattr(self, "exported_count", None),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "tests": self.history,
            "time_duration": [self.time_min, self.time_max],
            "fps": cfg["fps"],
        }
        with open(os.path.join(cfg["model_path"], "training_summary.json"), "w") as fp:
            json.dump(summary, fp, indent=2)
        self.log(f"Training complete: {total} iterations in {summary['train_seconds']:.1f} s "
                 f"(+{eval_seconds:.1f} s eval), {self.rep.num_gaussians:,} Gaussians, "
                 f"peak {summary['peak_memory_gib']:.2f} GiB")
        return summary
