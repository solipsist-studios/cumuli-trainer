# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""The trigs representation and its two losses."""

import math
import os

import numpy as np
import pytest
import torch

pytest.importorskip("cumuli_core.trigs", reason="needs a cumuli_core with the TRiGS slice")

from cumuli_core import trigs as core_trigs  # noqa: E402

from cumuli_trainer import registry  # noqa: E402
from cumuli_trainer.losses.base import LossInputs  # noqa: E402
from cumuli_trainer.representations.trigs import ROTATION_KEYS, TRANSLATION_KEYS, ZETA_KEYS, TrigsRepresentation  # noqa: E402

from test_representation import make_ctx, make_points  # noqa: E402


def make_rep(**kw):
    rep = TrigsRepresentation(make_ctx(sh_degree=2, t1=2.0, fps=10.0), **kw)
    rep.init_from_points(make_points(n_per=30, times=(0.0, 1.0, 2.0)))
    return rep


def randomise(rep, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for key in ZETA_KEYS:
            rep.params[key].copy_(torch.randn(rep.params[key].shape, generator=g) * 0.3)
        rep.params["anchor_offsets"].copy_(torch.randn(rep.params["anchor_offsets"].shape, generator=g) * 0.2)
        rep.params["t_log_sigmas"].copy_(torch.randn(rep.params["t_log_sigmas"].shape, generator=g) * 0.1 - 1.0)
        rep.params["t_centers"].copy_(torch.rand(rep.params["t_centers"].shape, generator=g))


def test_the_registry_knows_trigs_and_its_losses():
    assert registry.get("representation", "trigs") is TrigsRepresentation
    for name in ("trigs_motion", "trigs_rigid"):
        assert registry.get("loss", name)


def test_init_has_every_parameter_with_a_small_axis_and_no_motion():
    rep = make_rep(init_omega=0.05)
    p = rep.params
    n = rep.num_gaussians
    for key in ("means", "quats", "scales", "opacities", "sh0", "shN", "t_centers", "t_log_sigmas", "anchor_offsets", *ZETA_KEYS):
        assert key in p and p[key].shape[0] == n, key
    assert torch.allclose(p["zeta_base_r"].norm(dim=1), torch.full((n,), 0.05), atol=1e-6)
    for key in ZETA_KEYS:
        if key != "zeta_base_r":
            assert torch.all(p[key] == 0), key
    assert torch.all(p["anchor_offsets"] == 0)
    # no motion at all while nu is zero: every Gaussian stays at its canonical position
    g = rep.slice(0.7, 2)
    assert torch.allclose(g.means, p["means"], atol=1e-6)


def test_learning_rates_cover_every_parameter():
    from cumuli_trainer.config import DEFAULT_LR
    rep = make_rep()
    lrs = rep.learning_rates(dict(DEFAULT_LR))
    assert set(lrs) == set(rep.params)
    assert set(rep.decayed_params()) <= set(rep.params)


def test_slice_is_the_core_slice_in_normalised_time():
    rep = make_rep()
    randomise(rep)
    t = 0.9    # seconds, in a 2 s clip
    g = rep.slice(t, 2)
    want = core_trigs.slice_at(rep._tensors(), t / 2.0)
    assert torch.allclose(g.means, want[0]) and torch.allclose(g.quats, want[1]) and torch.allclose(g.opacities, want[2])
    assert torch.allclose(rep.means_at(t), g.means)
    assert g.colors.shape == (rep.num_gaussians, 9, 3) and g.sh_degree == 2


def test_a_per_gaussian_time_tensor_slices_each_gaussian_at_its_own_time():
    rep = make_rep()
    randomise(rep)
    n = rep.num_gaussians
    times = torch.linspace(0.2, 1.8, n)
    g = rep.slice(times, 2)
    for i in (0, n // 2, n - 1):
        one = rep.slice(float(times[i]), 2)
        assert torch.allclose(g.means[i], one.means[i], atol=1e-5)


def test_gradients_reach_every_parameter():
    rep = make_rep()
    randomise(rep)
    g = rep.slice(0.9, 2)
    loss = g.means.pow(2).sum() + g.quats.sum() + g.opacities.sum() + g.colors.sum() + g.scales.sum()
    loss.backward()
    for key, value in rep.params.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), key
        if key not in ("shN",):   # shN is a colour: summed above but with zero init the gradient is still 1
            assert value.grad.abs().sum() > 0, key


def test_export_is_in_seconds_and_slices_the_same_as_the_trainer(tmp_path):
    rep = make_rep()
    randomise(rep)
    path, n = rep.export_checkpoint(str(tmp_path), step=10, final=True)
    assert os.path.basename(path) == "trigs_ckpt.pt" and n == rep.num_gaussians
    ck = torch.load(path, weights_only=False)
    assert ck["fps"] == 10.0 and ck["n_frames"] == 21 and ck["metrics"]["windows"] == [[0.0, 2.0]]
    prm = ck["params"]
    for key in ("means", "scales", "quats", "opacities", "sh0", "shN", "times", "log_durations", "anchors", *ZETA_KEYS):
        assert key in prm and prm[key].shape[0] == n, key
    assert prm["quats"].shape == (n, 4) and prm["sh0"].shape == (n, 1, 3)
    # the file's own slice, in seconds, is the trainer's, in normalised time
    q = prm["quats"]
    tt = core_trigs.TrigsTensors(
        means=prm["means"], quats_wxyz=torch.cat([q[:, 3:4], q[:, :3]], 1), log_scales=prm["scales"], opacity_logit=prm["opacities"],
        sh=prm["sh0"], times=prm["times"], log_sigma=prm["log_durations"], anchors=prm["anchors"], **{k: prm[k] for k in ZETA_KEYS})
    for t in (0.0, 0.5, 1.3, 2.0):
        mine = rep.slice(t, 2)
        means, quats, opac = core_trigs.slice_at(tt, t)
        assert torch.allclose(means, mine.means, atol=2e-4), t
        assert torch.allclose(opac, mine.opacities, atol=1e-5), t
        qa = torch.nn.functional.normalize(quats, dim=1)
        qb = torch.nn.functional.normalize(mine.quats, dim=1)
        assert torch.allclose((qa * qb).sum(1).abs(), torch.ones(n), atol=1e-4), t   # the same rotation, whichever sign


def test_a_snapshot_is_not_the_final_file_and_dropped_rows_stay_out(tmp_path):
    rep = make_rep()
    keep = torch.arange(rep.num_gaussians) % 2 == 0
    path, n = rep.export_checkpoint(str(tmp_path), step=7, final=False, keep=keep)
    assert os.path.basename(path) == "trigs_7.pt" and n == int(keep.sum())
    assert not (tmp_path / "trigs_ckpt.pt").exists()


def test_interchange_is_refused():
    with pytest.raises(NotImplementedError):
        make_rep().to_interchange()


def test_interchange_seed_starts_with_the_velocity_as_a_translation_along_its_own_axis():
    rep = TrigsRepresentation(make_ctx(sh_degree=1, t1=2.0, fps=10.0))
    n = 5
    f = {"x": np.zeros(n), "y": np.zeros(n), "z": np.zeros(n), "vx": np.full(n, 0.5), "vy": np.zeros(n), "vz": np.zeros(n),
         "t_center": np.full(n, 1.0), "t_sigma": np.full(n, 0.3), "opacity": np.zeros(n), "f_dc_0": np.zeros(n),
         "f_dc_1": np.zeros(n), "f_dc_2": np.zeros(n)}
    for i in range(4):
        f[f"rot_{i}"] = np.array([1.0, 0, 0, 0][i] * np.ones(n))
    for i in range(3):
        f[f"scale_{i}"] = np.full(n, -3.0)
    rep.init_from_interchange({}, f)
    g = rep.slice(1.4, 1)     # 0.4 s after the centre
    assert torch.allclose(g.means[:, 0], torch.full((n,), 0.5 * 0.4), atol=1e-4)    # moved at the PLY's velocity
    assert torch.allclose(g.means[:, 1:], torch.zeros(n, 2), atol=1e-5)


def losses_inputs(rep, step=1):
    return LossInputs(pred_rgb=None, pred_alpha=None, gt_rgb=None, gt_alpha=None, params=rep.params, gaussians=None, info={}, step=step)


def test_the_motion_loss_is_zero_for_a_straight_curve_and_grows_with_the_bend():
    rep = make_rep()
    term = registry.create("loss", "trigs_motion")
    for key in ZETA_KEYS:
        with torch.no_grad():
            rep.params[key].zero_()
    assert float(term(losses_inputs(rep))) == 0.0
    with torch.no_grad():                      # c0 = 0, c1 = 1, c2 = 0 : second difference -2
        rep.params["zeta_c1_r"].fill_(1.0)
    assert float(term(losses_inputs(rep))) == pytest.approx(4.0 * 3, rel=1e-6)   # |(-2,-2,-2)|^2 = 12
    with torch.no_grad():                      # a straight ramp c0 = 0, c1 = 1, c2 = 2 has no second difference
        rep.params["zeta_c1_r"].fill_(1.0)
        rep.params["zeta_c2_r"].fill_(2.0)
    assert float(term(losses_inputs(rep))) == pytest.approx(0.0, abs=1e-10)


def test_the_rigid_loss_pulls_close_look_alikes_together_and_leaves_unlike_ones_free():
    rep = TrigsRepresentation(make_ctx(sh_degree=0))
    pts = {"xyz": np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [5, 5, 5], [5.01, 5, 5], [5.02, 5, 5]], np.float32),
           "rgb": np.array([[0.5] * 3] * 3 + [[0.5] * 3, [0.5] * 3, [0.5] * 3], np.float32), "time": np.zeros(6, np.float32)}
    rep.init_from_points(pts)
    with torch.no_grad():
        for key in ZETA_KEYS:
            rep.params[key].zero_()
        rep.params["zeta_base_t"][:, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    term = registry.create("loss", "trigs_rigid", k=2)
    assert float(term(losses_inputs(rep))) == pytest.approx(0.0, abs=1e-9)        # each cluster moves as one
    with torch.no_grad():
        rep.params["zeta_base_t"][:, 0] = torch.tensor([0.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    near = float(term(losses_inputs(rep, step=1000)))
    assert near > 0
    # the same differences between Gaussians of different colour cost less
    with torch.no_grad():
        rep.params["sh0"][1].add_(2.0)
        rep.params["sh0"][2].sub_(2.0)
    far = float(registry.create("loss", "trigs_rigid", k=2)(losses_inputs(rep)))
    assert far < near


def test_the_rigid_loss_is_differentiable_and_follows_a_population_that_changes_size():
    rep = make_rep()
    randomise(rep)
    term = registry.create("loss", "trigs_rigid")
    inp = losses_inputs(rep)
    value = term(inp)
    value.backward()
    assert rep.params["zeta_base_r"].grad is not None and torch.isfinite(rep.params["zeta_base_r"].grad).all()
    grown = make_rep()
    grown.init_from_points(make_points(n_per=40, times=(0.0, 1.0, 2.0)))
    term(losses_inputs(grown, step=2))      # a different size rebuilds the neighbours rather than indexing past the end


@pytest.mark.gpu
def test_train_export_and_resume_with_the_trigs_representation(tmp_path):
    from cumuli_trainer.config import build_config
    from cumuli_trainer.trainer import Trainer
    from test_train_gpu import make_dataset

    ds = str(tmp_path / "ds")
    make_dataset(ds, times=(0.0, 0.05, 0.1))
    out = str(tmp_path / "out")
    cfg = build_config({"source_path": ds, "model_path": out, "iterations": 60, "sh_degree": 1, "sh_degree_interval": 20,
                        "cameras_per_step": 2, "representation": {"name": "trigs"},
                        "strategy": {"name": "mcmc", "cap_max": 600, "grow_rate": 1.5, "refine_start": 10, "refine_every": 10},
                        "losses": {"trigs_motion": 1e-4, "trigs_rigid": 1.0}, "test_iterations": [60], "save_iterations": [30, 60],
                        "checkpoint_iterations": [30], "log_every": 5})
    summary = Trainer(cfg, log=lambda m: None).run()
    assert summary["tests"][-1]["psnr_db"] > 8
    assert os.path.exists(os.path.join(out, "trigs_ckpt.pt")) and os.path.exists(os.path.join(out, "trigs_30.pt"))
    assert not os.path.exists(os.path.join(out, "splat_4d.ply"))
    ck = torch.load(os.path.join(out, "trigs_ckpt.pt"), weights_only=False)
    assert ck["step"] == 60 and ck["params"]["means"].shape[0] == summary["exported_gaussians"]
    assert all(torch.isfinite(v).all() for v in ck["params"].values())
    tr = Trainer(dict(cfg, model_path=str(tmp_path / "resumed")), resume=os.path.join(out, "ckpt_30.pt"), log=lambda m: None)
    assert tr.start_step == 31
    tr.run()
