# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""End-to-end training on a tiny synthetic dataset (GPU)."""

import json
import math
import os

import numpy as np
import pytest
import torch

from cumuli_trainer.config import build_config
from cumuli_trainer.core import ply as core_ply


def _look_at_c2w_gl(eye, target=(0, 0, 0)):
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -f, eye  # OpenGL: camera looks down -z
    return m


def make_dataset(root, n_cams=4, times=(0.0, 0.1), w=64, h=48):
    from PIL import Image
    os.makedirs(root, exist_ok=True)
    frames, test = [], []
    for split, cams, out in (("train", range(n_cams), frames), ("test", [0.5], test)):
        for ci in cams:
            ang = 2 * math.pi * ci / n_cams
            c2w = _look_at_c2w_gl([3 * math.sin(ang), 0.3, 3 * math.cos(ang)])
            for ti, t in enumerate(times):
                rel = f"{split}/c{ci}_t{ti}"
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                img = np.zeros((h, w, 4), np.uint8)
                img[h // 4: 3 * h // 4, w // 3: 2 * w // 3] = (200, 80 + 100 * ti, 40, 255)
                Image.fromarray(img).save(os.path.join(root, rel + ".png"))
                out.append({"file_path": rel, "time": t, "fl_x": 60.0, "fl_y": 60.0,
                            "cx": w / 2, "cy": h / 2, "w": w, "h": h,
                            "camera_label": f"{ci}", "transform_matrix": c2w.tolist()})
    for name, fr in (("transforms_train.json", frames), ("transforms_test.json", test)):
        with open(os.path.join(root, name), "w") as fp:
            json.dump({"camera_model": "OPENCV", "frames": fr}, fp)
    rng = np.random.default_rng(0)
    n = 400
    pts = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"),
                             ("green", "u1"), ("blue", "u1"), ("time", "<f4")])
    xyz = rng.normal(scale=0.3, size=(n, 3))
    pts["x"], pts["y"], pts["z"] = xyz.T
    pts["red"], pts["green"], pts["blue"] = 200, 120, 40
    pts["time"] = rng.choice(times, n)
    with open(os.path.join(root, "points3d.ply"), "wb") as fp:
        fp.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "property float time\nend_header\n" % n).encode())
        fp.write(pts.tobytes())


@pytest.mark.gpu
@pytest.mark.parametrize("strategy", ["mcmc", "default"])
def test_train_save_resume(tmp_path, strategy, capsys):
    from cumuli_trainer.trainer import Trainer
    ds = str(tmp_path / "ds")
    make_dataset(ds)
    out = str(tmp_path / "out")
    strat = {"name": strategy, "refine_start": 10, "refine_every": 10, "refine_stop": 1000}
    if strategy == "mcmc":
        strat["cap_max"] = 600
        strat["grow_rate"] = 1.5
    cfg = build_config({"source_path": ds, "model_path": out, "iterations": 60,
                        "sh_degree": 1, "sh_degree_interval": 20, "cameras_per_step": 2,
                        "strategy": strat, "test_iterations": [60], "save_iterations": [30, 60],
                        "checkpoint_iterations": [30, 60], "log_every": 5})
    summary = Trainer(cfg, log=lambda m: None).run()
    printed = capsys.readouterr().out
    assert "[ITER 60] Saving Gaussians" in printed
    assert "Training progress:" in printed and "Evaluating test" in printed
    header, fields = core_ply.read_sogst_ply(os.path.join(out, "splat_4d.ply"))
    assert header["fps"] == pytest.approx(10.0)
    assert (header["time_min"], header["time_max"]) == (0.0, pytest.approx(0.1))
    assert fields["f_rest"].shape[1] == 9
    assert np.all(fields["t_sigma"] > 0)
    assert os.path.exists(os.path.join(out, "splat_4d_30.ply"))
    assert summary["tests"][-1]["psnr_db"] > 10
    if strategy == "mcmc":
        assert summary["num_gaussians"] == 600  # grew to the cap

    # resume from step 30 with a larger cap: densification is still live
    cfg2 = dict(cfg, model_path=str(tmp_path / "resumed"))
    if strategy == "mcmc":
        cfg2["strategy"] = dict(strat, cap_max=800)
    tr = Trainer(cfg2, resume=os.path.join(out, "ckpt_30.pt"), log=lambda m: None)
    assert tr.start_step == 31
    n_before = tr.rep.num_gaussians
    s2 = tr.run()
    if strategy == "mcmc":
        assert n_before == 600 and s2["num_gaussians"] > 600

    # seed a fresh run from the PLY: starts at step 1 with that many Gaussians
    cfg3 = dict(cfg, model_path=str(tmp_path / "seeded"), init_ply=os.path.join(out, "splat_4d.ply"),
                iterations=5, test_iterations=[5], save_iterations=[5], checkpoint_iterations=[])
    tr = Trainer(cfg3, log=lambda m: None)
    assert tr.start_step == 1 and tr.rep.num_gaussians == header["count"]
    tr.run()


@pytest.mark.gpu
def test_view_contribution_is_near_zero_for_a_hidden_gaussian():
    from cumuli_trainer.trainer import view_contribution
    dev = "cuda"
    # camera at the origin looking down +z; a big opaque Gaussian at z=2 hides
    # a small one straight behind it at z=4
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], device=dev)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2, device=dev)
    scales = torch.tensor([[0.5, 0.5, 0.05], [0.05, 0.05, 0.05]], device=dev)
    opac = torch.tensor([0.999, 0.999], device=dev)
    K = torch.tensor([[[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]]], device=dev)
    contrib = view_contribution(means, quats, scales, opac, torch.eye(4, device=dev)[None], K, 128, 128)
    alone = view_contribution(means[1:], quats[1:], scales[1:], opac[1:],
                              torch.eye(4, device=dev)[None], K, 128, 128)
    assert contrib[0] > 100                    # the front Gaussian covers many pixels
    # gsplat caps alpha at 0.99, so 1% leaks through: far below the 2 px cut
    assert contrib[1] < 0.02 * alone[0]
    assert contrib[1] < 0.5
