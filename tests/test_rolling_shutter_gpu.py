# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Rolling-shutter views (GPU): renders against brute-force ground truth, and
capture-time learning with a readout."""

import numpy as np
import pytest
import torch

from cumuli_core.shutter import pixel_offsets
from cumuli_trainer.config import build_config
from cumuli_trainer.data import Batch
from test_train_gpu import make_dataset

READOUT = 0.05  # s: long, so the shear is several pixels in a 96-row image
# A fisheye raw sensor wider than the undistorted crop (fl 60): its rows
# are curves in the image.
FISHEYE = {"model": "OPENCV_FISHEYE", "K": [[60.0, 0.0, 128.0], [0.0, 60.0, 100.0], [0.0, 0.0, 1.0]],
           "dist": [0.1, -0.05, 0.01, 0.0], "w": 256, "h": 200}


def moving_scene(tr, speed=5.0, seed=1):
    """Small opaque Gaussians, always visible, moving fast (scene units per second)."""
    p = tr.rep.params
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        p["velocities"].copy_(torch.randn(p["velocities"].shape, generator=g, device="cuda")
                              * speed * tr.rep.duration)
        p["scales"].fill_(float(np.log(0.015)))
        p["opacities"].fill_(3.0)
        p["t_log_sigmas"].fill_(3.0)
        p["sh0"].copy_(torch.rand(p["sh0"].shape, generator=g, device="cuda") * 2 - 1)


def brute_force(tr, i, t_capture, samples=241):
    """Ground truth of view i: the scene rendered at `samples` times across the
    readout; each pixel takes the render at its own raw row's time."""
    cam = tr.data.cams[i]
    w, h = tr.data.width, tr.data.height
    v, u = np.meshgrid(np.arange(h) + 0.5, np.arange(w) + 0.5, indexing="ij")  # gsplat pixel centres
    off = pixel_offsets(cam, u, v)                                            # [H, W] seconds
    span = np.linspace(-0.55 * abs(READOUT), 0.55 * abs(READOUT), samples)
    assert off.min() >= span[0] and off.max() <= span[-1]
    nearest = torch.as_tensor(np.abs(off[..., None] - span).argmin(-1), device="cuda")
    out = torch.zeros(h, w, 3, device="cuda")
    for k, dt in enumerate(span):
        sel = nearest == k
        if not sel.any():
            continue
        g = tr.rep.slice(t_capture + float(dt), 0)
        rgb, _, _ = tr.renderer.render(g, tr.data.viewmats[i:i + 1], tr.data.Ks[i:i + 1], w, h)
        out[sel] = rgb[0][sel]
    return out, off


@pytest.mark.gpu
@pytest.mark.parametrize("raw", [None, FISHEYE], ids=["pinhole", "fisheye"])
def test_rolling_shutter_render_matches_brute_force(tmp_path, raw):
    from cumuli_trainer.trainer import Trainer
    ds = str(tmp_path / "ds")
    make_dataset(ds, readout=READOUT, raw_camera=raw, offsets=(0.0, 0.004, -0.006, 0.008), w=128, h=96)
    cfg = build_config({"source_path": ds, "model_path": str(tmp_path / "out"), "iterations": 1,
                        "sh_degree": 0, "cameras_per_step": 4, "test_iterations": [],
                        "save_iterations": [], "checkpoint_iterations": []})
    tr = Trainer(cfg, log=lambda m: None)
    assert all(tr.data.rolling)
    moving_scene(tr)
    views = [0, 1, 2, 3]
    times = [tr.data.cams[i].capture_time for i in views]
    batch = Batch(time=tr.data.cams[0].time, index=torch.tensor(views, device="cuda"), images=None,
                  crop_size=(tr.data.height, tr.data.width), crop_origins=[(0, 0)] * 4, times=times,
                  views=views)
    errs_rs, errs_global = [], []
    with torch.no_grad():
        rs, _, info, _ = tr.render_batch(batch, 0)
        assert info["n_cameras"] == 4 and info["means2d"].shape[:2] == (4, 1)
        for c, i in enumerate(views):
            gt, off = brute_force(tr, i, times[c])
            if raw is not None:   # fisheye rows are curves: one image row spans several times
                assert np.ptp(off, axis=1).max() > 0.02 * READOUT
            g = tr.rep.slice(times[c], 0)
            glob, _, _ = tr.renderer.render(g, tr.data.viewmats[i:i + 1], tr.data.Ks[i:i + 1],
                                            tr.data.width, tr.data.height)
            # over the pixels the subject covers in either picture
            seen = (gt.max(-1).values > 0.02) | (glob[0].max(-1).values > 0.02)
            errs_rs.append(float((rs[c] - gt).abs()[seen].mean()))
            errs_global.append(float((glob[0] - gt).abs()[seen].mean()))
    rs_err, global_err = float(np.mean(errs_rs)), float(np.mean(errs_global))
    print(f"\n[{'fisheye' if raw else 'pinhole'}] mean |render - brute force| on the subject: "
          f"rolling {rs_err:.5f}, global at mid time {global_err:.5f} (ratio {rs_err / global_err:.3f})")
    assert global_err > 0.03           # the readout matters in this scene
    assert rs_err < 0.15 * global_err


@pytest.mark.gpu
def test_zero_readout_renders_as_global_shutter(tmp_path):
    """readout_s 0 in the file is a global shutter: the single-slice path."""
    from cumuli_trainer.trainer import Trainer
    ds = str(tmp_path / "ds")
    make_dataset(ds, readout=0.0, raw_camera=FISHEYE)
    cfg = build_config({"source_path": ds, "model_path": str(tmp_path / "out"), "iterations": 1,
                        "sh_degree": 0, "cameras_per_step": 4, "test_iterations": [],
                        "save_iterations": [], "checkpoint_iterations": []})
    tr = Trainer(cfg, log=lambda m: None)
    assert not any(tr.data.rolling)
    _, _, info, _ = tr.render_batch(tr.data.next(), 0)
    assert info["means2d"].dim() == 3   # [C, N, 2]: one render of one slice


@pytest.mark.gpu
def test_train_and_evaluate_with_rolling_shutter(tmp_path):
    from cumuli_trainer.trainer import Trainer
    ds = str(tmp_path / "ds")
    make_dataset(ds, readout=0.02, raw_camera=FISHEYE, offsets=(0.0, 0.004, -0.006, 0.008))
    cfg = build_config({"source_path": ds, "model_path": str(tmp_path / "out"), "iterations": 40,
                        "sh_degree": 0, "cameras_per_step": 2,
                        "strategy": {"name": "default", "refine_start": 10, "refine_every": 10,
                                     "refine_stop": 1000},
                        "test_iterations": [40], "save_iterations": [40], "checkpoint_iterations": []})
    tr = Trainer(cfg, log=lambda m: None)
    summary = tr.run()
    assert summary["tests"][-1]["psnr_db"] > 10
    assert summary["num_gaussians"] != 400  # densification still sees the per-view layout
    res = tr.evaluate(40)
    assert res["views"][0]["readout_s"] == pytest.approx(0.02)


@pytest.mark.gpu
def test_learned_time_offsets_with_rolling_shutter(tmp_path):
    """Photos of a moving scene taken by rolling-shutter cameras that fired at
    time + true offset: learning the corrections through the rolling-shutter
    render recovers the offsets (zero-meaned)."""
    from PIL import Image

    from cumuli_trainer.shutter import slice_view
    from cumuli_trainer.trainer import Trainer

    true = torch.tensor([0.0, 0.02, -0.03, 0.01])
    ds = str(tmp_path / "ds")
    make_dataset(ds, readout=0.03, raw_camera=FISHEYE)
    cfg = build_config({"source_path": ds, "model_path": str(tmp_path / "gt"), "iterations": 400, "sh_degree": 0,
                        "cameras_per_step": 2, "strategy": {"name": "default", "refine_stop": 0},
                        "losses": {"opacity_mask": 0, "aniso_reg": 0},
                        "test_iterations": [], "save_iterations": [], "checkpoint_iterations": []})
    scene = Trainer(cfg, log=lambda m: None)
    with torch.no_grad():
        g = torch.Generator(device="cuda").manual_seed(1)
        scene.rep.params["velocities"].copy_(torch.randn(scene.rep.params["velocities"].shape, generator=g,
                                                         device="cuda") * 3)
        scene.rep.params["opacities"].fill_(3.0)
        scene.rep.params["t_log_sigmas"].fill_(3.0)
        scene.rep.params["sh0"].copy_(torch.rand(scene.rep.params["sh0"].shape, generator=g, device="cuda") * 2 - 1)
        for i, cam in enumerate(scene.data.cams):
            sl = slice_view(scene.rep, cam.time + float(true[int(cam.label)]), 0, cam, scene.data.viewmats[i])
            rgb, alpha, _ = scene.renderer.render(sl, scene.data.viewmats[i:i + 1], scene.data.Ks[i:i + 1],
                                                  scene.data.width, scene.data.height)
            a = alpha[0].clamp(1e-6, 1)
            rgba = torch.cat([(rgb[0] / a).clamp(0, 1), alpha[0]], dim=-1)
            Image.fromarray((rgba.cpu().numpy() * 255).round().astype(np.uint8)).save(cam.image_path + ".png")

    cfg2 = build_config({**{k: v for k, v in cfg.items() if k not in ("time_offsets",)},
                         "model_path": str(tmp_path / "learn"),
                         "time_offsets": {"learn": True, "warmup": 0, "lr": 2e-3, "reg": 0.0, "clamp_frames": 0.6}})
    tr = Trainer(cfg2, log=lambda m: None)
    assert all(tr.data.rolling)
    with torch.no_grad():
        for k, p in tr.rep.params.items():
            p.copy_(scene.rep.params[k])
    for opt in tr.optimizers.values():
        for group in opt.param_groups:
            group["lr"] = 0.0
    tr.lr_init = {k: 0.0 for k in tr.lr_init}
    for step in range(1, 401):
        tr.train_step(step)
    learned = torch.tensor([tr.time_offsets.seconds()[str(i)] for i in range(4)])
    want = true - true.mean()
    assert torch.allclose(learned, want, atol=0.004), (learned, want)
