# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
import json

import pytest

from cumuli_trainer.config import (ConfigError, build_config, fps_from_times, load_config,
                                   loss_specs, parse_overrides)

BASE = {"source_path": "/data/ds", "model_path": "/out"}


def test_defaults():
    cfg = build_config(dict(BASE))
    assert cfg["iterations"] == 30000
    assert cfg["sh_degree"] == 2
    assert cfg["strategy"]["name"] == "mcmc"
    assert cfg["representation"]["name"] == "spacetime"
    assert cfg["background"] == "black"
    assert cfg["test_iterations"] == [30000]
    assert cfg["save_iterations"] == [30000]
    assert cfg["t_init_div"] == 100
    assert cfg["scene_bound_factor"] == 2.0


def test_negative_scene_bound_factor_is_an_error():
    with pytest.raises(ConfigError, match="scene_bound_factor"):
        build_config({**BASE, "scene_bound_factor": -1})


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigError, match="unknown config key"):
        build_config({**BASE, "itertions": 10})


def test_unknown_lr_key_is_an_error():
    with pytest.raises(ConfigError):
        build_config({**BASE, "lr": {"velocity": 1e-3}})


def test_required_keys():
    with pytest.raises(ConfigError):
        build_config({"source_path": "/x"})


def test_plugin_sections_accept_names_and_mappings():
    cfg = build_config({**BASE, "strategy": "default", "renderer": {"name": "gsplat", "eps2d": 0.3}})
    assert cfg["strategy"] == {"name": "default"}
    assert cfg["renderer"]["eps2d"] == 0.3


def test_overrides(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("source_path: /d\nmodel_path: /m\nstrategy: mcmc\n")
    ov = parse_overrides(["--iterations", "500", "--strategy.cap_max", "1000",
                          "--test_iterations", "[100, 500]", "--background=white"])
    cfg = load_config(str(p), ov)
    assert cfg["iterations"] == 500
    assert cfg["strategy"] == {"name": "mcmc", "cap_max": 1000}
    assert cfg["test_iterations"] == [100, 500]
    assert cfg["background"] == "white"


def test_orchestrator_json_config(tmp_path):
    # the shape cumuli's run_unified_pipeline writes (JSON is valid YAML)
    p = tmp_path / "gs4d_config.yaml"
    p.write_text(json.dumps({
        "source_path": "/d", "model_path": "/m", "time_duration": [0, 0.458333], "fps": 24,
        "iterations": 30000, "test_iterations": [30000], "save_iterations": [30000],
        "t_init_div": 100}))
    cfg = load_config(str(p))
    assert cfg["time_duration"] == [0.0, 0.458333]
    assert cfg["fps"] == 24.0
    assert cfg["t_init_div"] == 100.0


def test_bad_values():
    for bad in ({"time_duration": [1, 0]}, {"background": "grey"}, {"sh_degree": 4},
                {"cameras_per_step": 0}, {"losses": {"l1": "heavy"}}):
        with pytest.raises(ConfigError):
            build_config({**BASE, **bad})


def test_fps_from_times():
    assert fps_from_times([i / 24 for i in range(12)] * 12) == 24.0
    assert fps_from_times([0.0, 1 / 30, 2 / 30]) == 30.0
    with pytest.raises(ConfigError):
        fps_from_times([0.5, 0.5])


def test_loss_specs_merge_strategy_defaults():
    cfg = build_config({**BASE, "losses": {"l1": 0.8, "ssim": 0.2, "opacity_reg": 0}})
    specs = loss_specs(cfg, {"opacity_reg": 0.01, "scale_reg": 0.01})
    names = {n: w for n, w, _ in specs}
    assert names == {"l1": 0.8, "ssim": 0.2, "opacity_mask": 0.005, "scale_reg": 0.01,
                     "aniso_reg": 0.1}
    cfg = build_config({**BASE, "losses": {"opacity_mask": {"weight": 0.005, "eps": 1e-5},
                                           "l1": 0, "ssim": 0, "aniso_reg": 0}})
    (name, weight, params), = loss_specs(cfg)
    assert (name, weight, params) == ("opacity_mask", 0.005, {"eps": 1e-5})


def test_loss_override_merges_with_defaults():
    cfg = load_config(None, {"source_path": "/d", "model_path": "/m", "losses.opacity_reg": 0.001})
    assert cfg["losses"] == {"l1": 0.8, "ssim": 0.2, "opacity_mask": 0.005,
                             "aniso_reg": {"weight": 0.1, "max_ratio": 10.0}, "opacity_reg": 0.001}
