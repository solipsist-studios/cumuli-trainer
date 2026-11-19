# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Learned per-camera capture-time corrections.

Cameras that are not genlocked fire at slightly different moments. The
dataset already carries each camera's measured offset (``time_offset``, from
the sync step); this learns what is left, one constant per physical camera,
from the photometric loss, so a camera whose measured offset is a little off
is rendered at the moment its photo was actually taken.

    correction_k = c * tanh(raw_k / c)        c = clamp_frames / fps
    effective_k  = correction_k - mean(correction)

The tanh keeps each correction within +-c (half a frame by default) with a
smooth gradient, and removing the mean keeps the corrections relative: a
shift of every camera at once is the same as a shift of the whole scene in
time, which the photos cannot tell apart, so it is not learned. Learning
starts after ``warmup`` steps, when the scene has settled enough to tell
which way time should move. A quadratic penalty (``reg``, in units of the
clamp) pulls the corrections back toward the measured offsets.

Held-out test cameras are never trained, so they get no correction.
"""

from __future__ import annotations

import torch

DEFAULTS = {"learn": False, "warmup": 8000, "lr": 1e-4, "reg": 0.01, "clamp_frames": 0.5}


def validate(section) -> dict:
    """The ``time_offsets`` config section, defaults filled in and checked."""
    from .config import ConfigError

    if section is None:
        section = {}
    if isinstance(section, bool):
        section = {"learn": section}
    if not isinstance(section, dict):
        raise ConfigError("time_offsets must be a mapping (learn, warmup, lr, reg, clamp_frames) or true/false")
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"unknown time_offsets key(s) {unknown}; valid: {sorted(DEFAULTS)}")
    out = {**DEFAULTS, **section}
    out["learn"] = bool(out["learn"])
    out["warmup"] = int(out["warmup"])
    for key in ("lr", "reg", "clamp_frames"):
        out[key] = float(out[key])
    if out["warmup"] < 0 or out["lr"] <= 0 or out["reg"] < 0 or out["clamp_frames"] <= 0:
        raise ConfigError("time_offsets: warmup and reg must be 0 or more, lr and clamp_frames above 0")
    return out


class CameraTimeOffsets:
    """One learnable capture-time correction per physical camera (by camera label)."""

    def __init__(self, labels, fps: float, cfg: dict, device):
        self.labels = sorted({str(l) for l in labels})
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.clamp_s = float(cfg["clamp_frames"]) / float(fps)
        self.warmup = int(cfg["warmup"])
        self.reg = float(cfg["reg"])
        self.raw = torch.zeros(len(self.labels), device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.raw], lr=float(cfg["lr"]), eps=1e-15)

    def view_index(self, cams, device) -> torch.Tensor:
        """[V] the correction index of each training view."""
        return torch.tensor([self.index[str(c.label)] for c in cams], dtype=torch.long, device=device)

    def active(self, step: int) -> bool:
        return step > self.warmup and len(self.labels) > 1

    def effective(self) -> torch.Tensor:
        """[K] seconds, zero mean."""
        c = self.clamp_s
        corr = c * torch.tanh(self.raw / c)
        return corr - corr.mean()

    def penalty(self) -> torch.Tensor:
        return self.reg * (self.effective() / self.clamp_s).pow(2).mean()

    def step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def seconds(self) -> dict:
        """{label: seconds} of the corrections in effect."""
        return {l: float(v) for l, v in zip(self.labels, self.effective().tolist())}

    def state_dict(self) -> dict:
        return {"labels": self.labels, "raw": self.raw.detach().clone(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state) -> None:
        if not state:
            return
        if list(state["labels"]) != self.labels:
            raise ValueError(f"checkpoint time offsets are for cameras {state['labels']}, this dataset has {self.labels}")
        with torch.no_grad():
            self.raw.copy_(state["raw"].to(self.raw.device))
        self.optimizer.load_state_dict(state["optimizer"])
