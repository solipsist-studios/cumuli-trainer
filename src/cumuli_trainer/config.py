# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Run configuration.

A run is one YAML (or JSON, which is valid YAML) file plus optional
``--key value`` overrides on the command line. Dotted keys reach into nested
sections: ``--strategy.cap_max 400000``. Override values are parsed as YAML,
so ``--test_iterations "[7000, 30000]"`` gives a list.

Top-level keys (anything else is an error, so a typo cannot be silently
ignored):

    source_path        dataset_4dgs directory (required)
    model_path         output directory (required)
    time_duration      [min, max] clip window in seconds; default: the span of
                       the training camera times
    fps                clip frame rate; default: worked out from the training
                       camera timestamps
    iterations         30000
    sh_degree          2 (measured better than 3 on 12-camera rigs)
    sh_degree_interval 1000: one more SH band every this many steps
    t_init_div         initial t_sigma = sqrt(duration / t_init_div); 100
    representation     name or {name, ...params}; default spacetime
    strategy           name or {name, ...params}; default mcmc
    renderer           name or {name, ...params}; default gsplat
    losses             {term: weight} or {term: {weight, ...params}}, merged
                       over the defaults (l1 0.8, ssim 0.2, opacity_mask
                       0.005); weight 0 removes a term
    lr                 per-parameter learning rates, see DEFAULT_LR
    background         black | white | random | [r, g, b]
    cameras_per_step   cameras rendered per step, all at one instant
    test_iterations    list of steps at which to score the test cameras
    save_iterations    list of steps at which to write splat_4d.ply
    checkpoint_iterations  list of steps at which to write ckpt_<iter>.pt
    seed               0
    init_ply           seed from an interchange PLY instead of points3d.ply
    resume             continue from a ckpt_<iter>.pt
    log_every          steps between progress-bar refreshes (host syncs)
    test_every         score every Nth test frame (1 = all)
    data_device        auto | cuda | cpu: where decoded training images live
    export_min_opacity leave Gaussians whose peak opacity inside the window
                       is below this out of splat_4d.ply. Default 1/255: gsplat
                       culls those at every instant anyway, so the render is
                       unchanged. 0 keeps everything. Checkpoints keep all.
    export_min_contribution  drop Gaussians whose largest contribution to any
                       training view (sum of transmittance x alpha over its
                       pixels) is below this many pixels. Hidden Gaussians get
                       no colour correction and glint from new viewpoints.
                       Default 2; 0 keeps them. Checkpoints keep all.
    scene_bound_factor bounds radius as a multiple of the scene scale (1.1 x
                       the furthest camera from the camera centroid), centred
                       on that centroid. MCMC recycles Gaussians outside it
                       and export drops them. Default 2; 0 disables.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import yaml

DEFAULT_LR = {
    # From the 3DGS paper (Kerbl et al. 2023, section 7.1) and gsplat's
    # simple_trainer. means is multiplied by the scene scale and decays
    # exponentially to means * means_final_ratio.
    "means": 1.6e-4,
    "means_final_ratio": 0.01,
    "scales": 5e-3,
    "quats": 1e-3,
    "opacities": 5e-2,
    "sh0": 2.5e-3,
    "shN": 2.5e-3 / 20,
    # Temporal parameters. Time is normalised to [0, 1] over time_duration
    # inside the trainer, so these do not depend on clip length. velocities
    # are scene units per normalised time, multiplied by the scene scale and
    # decayed like means. See README for how they were chosen.
    "velocities": 1.6e-4,
    "t_centers": 1e-3,
    "t_log_sigmas": 5e-3,
}

DEFAULTS: dict[str, Any] = {
    "source_path": None,
    "model_path": None,
    "time_duration": None,
    "fps": None,
    "iterations": 30000,
    "sh_degree": 2,
    "sh_degree_interval": 1000,
    "t_init_div": 100,
    "representation": {"name": "spacetime"},
    "strategy": {"name": "mcmc"},
    "renderer": {"name": "gsplat"},
    # aniso_reg: without the MCMC opacity regulariser, needles turn opaque
    # and draw as sparkling scratches in the viewer (Heidi: 29.6% of
    # Gaussians opaque with an axis ratio above 20). Weight 0.1 at ratio 10
    # left 0.0% for 0.07 dB and LPIPS 0.0094 -> 0.0096 on held-out cam05.
    "losses": {"l1": 0.8, "ssim": 0.2, "opacity_mask": 0.005,
               "aniso_reg": {"weight": 0.1, "max_ratio": 10.0}},
    "lr": dict(DEFAULT_LR),
    "background": "black",
    # 2 beat 4 and 8 on dur_12 at half the time: 40.04 / 39.93 / 39.59 dB,
    # 5.4 / 11.0 / 23.1 min (sh_degree 3, no MCMC regularisers)
    "cameras_per_step": 2,
    "test_iterations": None,
    "save_iterations": None,
    "checkpoint_iterations": None,
    "seed": 0,
    "init_ply": None,
    "resume": None,
    "log_every": 10,
    "test_every": 1,
    "data_device": "auto",
    "export_min_opacity": 1.0 / 255.0,
    "scene_bound_factor": 2.0,
    # Heidi: dropping Gaussians under 2 px in every training view removed
    # 51% of them, 86% of the off-palette opaque ones, and moved held-out
    # cam05 from 36.07 to 36.12 dB (LPIPS 0.0096 both).
    "export_min_contribution": 2.0,
}

# Sections whose contents belong to a plugin and are validated by it.
PLUGIN_SECTIONS = ("representation", "strategy", "renderer")


class ConfigError(ValueError):
    pass


def _as_plugin(value, section):
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, dict):
        if "name" not in value:
            raise ConfigError(f"{section} needs a 'name'")
        return dict(value)
    raise ConfigError(f"{section} must be a name or a mapping, got {value!r}")


def _set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or node[part] is None:
            node[part] = {}
        elif isinstance(node[part], str) and part in PLUGIN_SECTIONS:
            node[part] = {"name": node[part]}
        elif not isinstance(node[part], dict):
            raise ConfigError(f"cannot set {key}: {part} is not a section")
        node = node[part]
    node[parts[-1]] = value


def parse_overrides(argv: list[str]) -> dict[str, Any]:
    """``["--a.b", "1", "--c", "x"]`` -> ``{"a.b": 1, "c": "x"}``."""
    out = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise ConfigError(f"expected --key, got {tok!r}")
        key = tok[2:]
        if "=" in key:
            key, raw = key.split("=", 1)
            i += 1
        else:
            if i + 1 >= len(argv):
                raise ConfigError(f"--{key} needs a value")
            raw = argv[i + 1]
            i += 2
        out[key.replace("-", "_") if "." not in key else key] = yaml.safe_load(raw)
    return out


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Merge defaults, the file at ``path`` and dotted ``overrides``."""
    raw = {}
    if path is not None:
        with open(path) as fp:
            raw = yaml.safe_load(fp) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
    for key, value in (overrides or {}).items():
        _set_dotted(raw, key, value)
    return build_config(raw)


def build_config(raw: dict) -> dict:
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"unknown config key(s): {unknown}; valid keys: {sorted(DEFAULTS)}")
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key in PLUGIN_SECTIONS:
            cfg[key] = _as_plugin(value, key)
        elif key == "lr":
            if not isinstance(value, dict):
                raise ConfigError("lr must be a mapping")
            bad = sorted(set(value) - set(DEFAULT_LR))
            if bad:
                raise ConfigError(f"unknown lr key(s) {bad}; valid: {sorted(DEFAULT_LR)}")
            cfg["lr"].update(value)
        elif key == "losses":
            if not isinstance(value, dict):
                raise ConfigError("losses must be a mapping of term -> weight")
            # merged over the defaults; weight 0 removes a default term
            cfg["losses"].update(value)
        else:
            cfg[key] = value
    validate(cfg)
    return cfg


def _int_list(value, name):
    if value is None:
        return None
    if isinstance(value, int):
        value = [value]
    try:
        return sorted({int(v) for v in value})
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a list of integers, got {value!r}") from None


def validate(cfg: dict) -> None:
    for key in ("source_path", "model_path"):
        if not cfg.get(key):
            raise ConfigError(f"{key} is required")
    cfg["iterations"] = int(cfg["iterations"])
    if cfg["iterations"] < 1:
        raise ConfigError("iterations must be positive")
    cfg["sh_degree"] = int(cfg["sh_degree"])
    if not 0 <= cfg["sh_degree"] <= 3:
        raise ConfigError("sh_degree must be 0..3")
    td = cfg["time_duration"]
    if td is not None:
        if not (isinstance(td, (list, tuple)) and len(td) == 2):
            raise ConfigError("time_duration must be [min, max] in seconds")
        cfg["time_duration"] = [float(td[0]), float(td[1])]
        if cfg["time_duration"][1] < cfg["time_duration"][0]:
            raise ConfigError("time_duration max is below min")
    if cfg["fps"] is not None:
        cfg["fps"] = float(cfg["fps"])
        if cfg["fps"] <= 0:
            raise ConfigError("fps must be positive")
    cfg["t_init_div"] = float(cfg["t_init_div"])
    if cfg["t_init_div"] <= 0:
        raise ConfigError("t_init_div must be positive")
    cfg["cameras_per_step"] = int(cfg["cameras_per_step"])
    if cfg["cameras_per_step"] < 1:
        raise ConfigError("cameras_per_step must be at least 1")
    bg = cfg["background"]
    if isinstance(bg, (list, tuple)):
        if len(bg) != 3:
            raise ConfigError("background colour must have three components")
        cfg["background"] = [float(c) for c in bg]
    elif bg not in ("black", "white", "random"):
        raise ConfigError(f"background must be black, white, random or [r, g, b]; got {bg!r}")
    for key in ("test_iterations", "save_iterations", "checkpoint_iterations"):
        cfg[key] = _int_list(cfg[key], key)
    if cfg["test_iterations"] is None:
        cfg["test_iterations"] = [cfg["iterations"]]
    if cfg["save_iterations"] is None:
        cfg["save_iterations"] = [cfg["iterations"]]
    if cfg["checkpoint_iterations"] is None:
        cfg["checkpoint_iterations"] = [cfg["iterations"]]
    cfg["export_min_opacity"] = float(cfg["export_min_opacity"])
    if not 0.0 <= cfg["export_min_opacity"] < 1.0:
        raise ConfigError("export_min_opacity must be in [0, 1)")
    cfg["export_min_contribution"] = float(cfg["export_min_contribution"])
    if cfg["export_min_contribution"] < 0:
        raise ConfigError("export_min_contribution must be >= 0 (0 keeps hidden Gaussians)")
    cfg["scene_bound_factor"] = float(cfg["scene_bound_factor"])
    if cfg["scene_bound_factor"] < 0:
        raise ConfigError("scene_bound_factor must be >= 0 (0 disables the bounds)")
    if cfg["data_device"] not in ("auto", "cuda", "cpu"):
        raise ConfigError("data_device must be auto, cuda or cpu")
    for term, spec in cfg["losses"].items():
        if isinstance(spec, dict):
            if "weight" not in spec:
                raise ConfigError(f"loss {term} needs a weight")
        elif not isinstance(spec, (int, float)):
            raise ConfigError(f"loss {term}: weight must be a number or a mapping")
    cfg["source_path"] = os.path.expanduser(str(cfg["source_path"]))
    cfg["model_path"] = os.path.expanduser(str(cfg["model_path"]))


def loss_specs(cfg: dict, strategy_defaults: dict | None = None) -> list[tuple[str, float, dict]]:
    """``[(term, weight, params)]``. A strategy's default terms apply unless
    the config names the term itself (weight 0 switches a term off)."""
    merged = dict(strategy_defaults or {})
    merged.update(cfg["losses"])
    out = []
    for term, spec in merged.items():
        if isinstance(spec, dict):
            params = {k: v for k, v in spec.items() if k != "weight"}
            weight = float(spec["weight"])
        else:
            params, weight = {}, float(spec)
        if weight != 0.0:
            out.append((term, weight, params))
    return out


def dump(cfg: dict) -> str:
    return json.dumps(cfg, indent=2, sort_keys=True)


def fps_from_times(times) -> float:
    """Frame rate from camera timestamps (seconds): 1 / median frame step."""
    uniq = sorted({round(float(t), 9) for t in times})
    if len(uniq) < 2:
        raise ConfigError("the dataset has a single timestamp, so fps cannot be worked out; "
                          "set fps in the config")
    steps = [b - a for a, b in zip(uniq, uniq[1:])]
    steps.sort()
    step = steps[len(steps) // 2]
    return round(1.0 / step, 3)
