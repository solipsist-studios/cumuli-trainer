# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
import math

import numpy as np
import pytest
import torch

from cumuli_trainer.core import ply as core_ply
from cumuli_trainer.representations.base import RepresentationContext
from cumuli_trainer.representations.spacetime import SpacetimeRepresentation, knn_mean_distance


def make_ctx(sh_degree=2, t0=0.0, t1=0.5, fps=24.0, div=100.0):
    return RepresentationContext(time_min=t0, time_max=t1, fps=fps, scene_scale=3.0,
                                 sh_degree=sh_degree, t_init_div=div, device=torch.device("cpu"))


def make_points(n_per=50, times=(0.0, 0.25, 0.5), seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.concatenate([rng.normal(size=(n_per, 3)).astype(np.float32) for _ in times])
    rgb = rng.uniform(size=(len(xyz), 3)).astype(np.float32)
    time = np.repeat(np.asarray(times, dtype=np.float32), n_per)
    return {"xyz": xyz, "rgb": rgb, "time": time}


def test_init_shapes_and_values():
    rep = SpacetimeRepresentation(make_ctx(sh_degree=2))
    pts = make_points()
    rep.init_from_points(pts)
    p = rep.params
    n = len(pts["xyz"])
    assert p["means"].shape == (n, 3)
    assert p["quats"].shape == (n, 4)
    assert p["scales"].shape == (n, 3)
    assert p["opacities"].shape == (n,)
    assert p["sh0"].shape == (n, 1, 3)
    assert p["shN"].shape == (n, 8, 3)
    assert torch.all(p["velocities"] == 0)
    # t_center from point time, normalised over the window
    assert torch.allclose(p["t_centers"][:50], torch.zeros(50))
    assert torch.allclose(p["t_centers"][-50:], torch.ones(50))
    # t_sigma = sqrt(duration / t_init_div), in seconds, stored normalised
    sigma_s = math.exp(float(p["t_log_sigmas"][0].detach())) * 0.5
    assert sigma_s == pytest.approx(math.sqrt(0.5 / 100), rel=1e-5)
    # peak opacity starts at init_opacity
    assert torch.allclose(torch.sigmoid(p["opacities"]), torch.full((n,), 0.1), atol=1e-6)


def test_knn_scale_is_per_timestamp():
    # the same point cloud repeated at three timestamps must get the same
    # scale as one copy alone (duplicates would otherwise give distance 0)
    rng = np.random.default_rng(1)
    one = rng.normal(size=(40, 3)).astype(np.float32)
    pts = {"xyz": np.concatenate([one] * 3), "rgb": np.full((120, 3), 0.5, np.float32),
           "time": np.repeat(np.array([0.0, 0.25, 0.5], np.float32), 40)}
    rep = SpacetimeRepresentation(make_ctx())
    rep.init_from_points(pts)
    expect = np.log(knn_mean_distance(one, 3))
    got = rep.params["scales"][:40, 0].detach().numpy()
    np.testing.assert_allclose(got, expect, rtol=1e-5)


def test_slice_matches_spec():
    rep = SpacetimeRepresentation(make_ctx(t0=1.0, t1=3.0))
    rep.init_from_points(make_points(times=(1.0, 2.0, 3.0)))
    with torch.no_grad():
        rep.params["velocities"].normal_()
    f = rep.to_interchange()
    t = 1.7
    g = rep.slice(t, 0)
    dt = t - f["t_center"]
    vel = np.stack([f["vx"], f["vy"], f["vz"]], 1)
    xyz = np.stack([f["x"], f["y"], f["z"]], 1)
    alpha = 1 / (1 + np.exp(-f["opacity"])) * np.exp(-0.5 * (dt / f["t_sigma"]) ** 2)
    np.testing.assert_allclose(g.means.detach().numpy(), xyz + vel * dt[:, None], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(g.opacities.detach().numpy(), alpha, rtol=1e-4, atol=1e-7)


def test_interchange_round_trip(tmp_path):
    rep = SpacetimeRepresentation(make_ctx(sh_degree=3, t0=0.0, t1=0.458333))
    rep.init_from_points(make_points(times=(0.0, 0.2, 0.458333)))
    with torch.no_grad():
        for k in ("velocities", "shN", "t_centers", "t_log_sigmas", "opacities"):
            rep.params[k].add_(0.1 * torch.randn_like(rep.params[k]))
    fields = rep.to_interchange()
    assert fields["f_rest"].shape[1] == 45
    path = str(tmp_path / "splat_4d.ply")
    core_ply.write_sogst_ply(path, fields, 0.0, 0.458333, 24.0)
    header, back = core_ply.read_sogst_ply(path)
    assert header["fps"] == 24.0 and header["time_max"] == pytest.approx(0.458333)
    for k in ("x", "vx", "t_center", "t_sigma", "opacity", "rot_0", "scale_2", "f_dc_1"):
        np.testing.assert_allclose(back[k], fields[k], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(back["f_rest"], fields["f_rest"], rtol=1e-6)

    rep2 = SpacetimeRepresentation(make_ctx(sh_degree=3, t0=0.0, t1=0.458333))
    rep2.init_from_interchange(header, back)
    again = rep2.to_interchange()
    for k in ("x", "vy", "t_center", "t_sigma", "opacity", "scale_0", "f_dc_2"):
        np.testing.assert_allclose(again[k], fields[k], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(again["f_rest"], fields["f_rest"], rtol=1e-5, atol=1e-7)


def test_f_rest_is_channel_major():
    rep = SpacetimeRepresentation(make_ctx(sh_degree=1))
    rep.init_from_points(make_points(n_per=2, times=(0.0,)))
    with torch.no_grad():
        rep.params["shN"].zero_()
        rep.params["shN"][0, 2, 1] = 7.0  # coefficient k=2, channel j=1
    fr = rep.to_interchange()["f_rest"]
    assert fr[0, 1 * 3 + 2] == 7.0


def test_peak_opacity_inside_window():
    rep = SpacetimeRepresentation(make_ctx())
    rep.init_from_points(make_points(n_per=1, times=(0.0, 0.25, 0.5)))
    with torch.no_grad():
        rep.params["t_centers"].copy_(torch.tensor([0.5, 1.0 + 3 * 0.1, -3 * 0.1]))
        rep.params["t_log_sigmas"].fill_(math.log(0.1))
    peak = rep.peak_opacity()
    assert peak[0] == pytest.approx(0.1, rel=1e-5)  # centre inside: peak opacity
    # three sigmas outside the window: exp(-4.5) of the peak
    assert peak[1] == pytest.approx(0.1 * math.exp(-4.5), rel=1e-4)
    assert peak[2] == pytest.approx(0.1 * math.exp(-4.5), rel=1e-4)


def test_t_sigma_cap_and_export_pruning():
    rep = SpacetimeRepresentation(make_ctx(t0=0.0, t1=0.5), max_t_sigma=10.0)
    rep.init_from_points(make_points(n_per=2, times=(0.0, 0.5)))
    with torch.no_grad():
        rep.params["t_log_sigmas"][0] = 50.0  # runaway static Gaussian
        rep.params["opacities"][1] = -10.0  # peak opacity 4.5e-5
    f = rep.to_interchange()
    assert f["t_sigma"][0] == pytest.approx(10.0 * 0.5, rel=1e-6)  # 10 durations, in seconds
    assert len(f["x"]) == 4
    kept = rep.to_interchange(min_peak_opacity=1 / 255)
    assert len(kept["x"]) == 3
    assert kept["t_sigma"][0] == pytest.approx(5.0, rel=1e-6)


def test_export_honours_a_keep_mask():
    rep = SpacetimeRepresentation(make_ctx(t0=0.0, t1=0.5))
    rep.init_from_points(make_points(n_per=2, times=(0.0, 0.5)))
    n = rep.num_gaussians
    keep = torch.ones(n, dtype=torch.bool)
    keep[0] = False
    assert len(rep.to_interchange(keep=keep)["x"]) == n - 1
    with torch.no_grad():
        rep.params["opacities"][1] = -10.0
    assert len(rep.to_interchange(min_peak_opacity=1 / 255, keep=keep)["x"]) == n - 2


# -- degree-2 motion -------------------------------------------------------

def make_degree2(t0=1.0, t1=3.0, randomise=True, **kw):
    rep = SpacetimeRepresentation(make_ctx(t0=t0, t1=t1), motion_degree=2, **kw)
    rep.init_from_points(make_points(times=(t0, (t0 + t1) / 2, t1)))
    if randomise:
        with torch.no_grad():
            rep.params["velocities"].normal_()
            rep.params["accelerations"].normal_()
    return rep


@pytest.mark.parametrize("degree", [0, 3, "2"])
def test_invalid_motion_degree_raises(degree):
    with pytest.raises(ValueError, match="motion_degree"):
        SpacetimeRepresentation(make_ctx(), motion_degree=degree)


def test_degree_1_is_unchanged_by_default():
    rep = SpacetimeRepresentation(make_ctx())
    rep.init_from_points(make_points())
    assert rep.motion_degree == 1
    assert "accelerations" not in rep.params
    assert set(rep.decayed_params()) == {"means", "velocities"}
    assert "ax" not in rep.to_interchange()


def test_degree_2_allocates_zero_trained_accelerations():
    rep = make_degree2(randomise=False)
    a = rep.params["accelerations"]
    assert isinstance(a, torch.nn.Parameter) and a.requires_grad
    assert a.shape == rep.params["means"].shape and torch.all(a == 0)
    assert "accelerations" in rep.decayed_params()
    lr = {"means": 1, "quats": 1, "scales": 1, "opacities": 1, "sh0": 1, "shN": 1,
          "velocities": 1, "t_centers": 1, "t_log_sigmas": 1, "accelerations": 2.0}
    assert rep.learning_rates(lr)["accelerations"] == pytest.approx(2.0 * rep.ctx.scene_scale)


def test_degree_2_slice_matches_spec_with_no_half_factor():
    rep = make_degree2()
    f = rep.to_interchange()
    t = 1.7
    g = rep.slice(t, 0)
    dt = (t - f["t_center"])[:, None]
    xyz = np.stack([f["x"], f["y"], f["z"]], 1)
    vel = np.stack([f["vx"], f["vy"], f["vz"]], 1)
    acc = np.stack([f["ax"], f["ay"], f["az"]], 1)
    np.testing.assert_allclose(g.means.detach().numpy(), xyz + vel * dt + acc * dt * dt,
                               rtol=1e-4, atol=1e-4)


def test_acceleration_gradient_flows_through_the_slice():
    rep = make_degree2(randomise=False)
    rep.slice(1.7, 0).means.sum().backward()
    assert rep.params["accelerations"].grad.abs().sum() > 0


def test_acceleration_converts_by_duration_squared():
    d = 0.458333
    rep = SpacetimeRepresentation(make_ctx(t0=0.0, t1=d), motion_degree=2)
    rep.init_from_points(make_points(n_per=2, times=(0.0, d)))
    with torch.no_grad():
        rep.params["accelerations"].fill_(1.0)
        rep.params["velocities"].fill_(1.0)
    f = rep.to_interchange()
    assert f["ax"][0] == pytest.approx(1.0 / d ** 2, rel=1e-6)  # about 4.76
    assert f["vx"][0] == pytest.approx(1.0 / d, rel=1e-6)


def test_degree_2_ply_round_trip(tmp_path):
    rep = make_degree2(t0=0.0, t1=0.458333)
    fields = rep.to_interchange()
    path = str(tmp_path / "splat_4d.ply")
    core_ply.write_sogst_ply(path, fields, 0.0, 0.458333, 24.0)
    header, back = core_ply.read_sogst_ply(path)
    assert header["motion_degree"] == 2
    for k in ("ax", "ay", "az"):
        np.testing.assert_allclose(back[k], fields[k], rtol=1e-6, atol=1e-7)
    rep2 = SpacetimeRepresentation(make_ctx(t0=0.0, t1=0.458333), motion_degree=2)
    rep2.init_from_interchange(header, back)
    again = rep2.to_interchange()
    for k in ("ax", "ay", "az"):
        np.testing.assert_allclose(again[k], fields[k], rtol=1e-4, atol=1e-5)


def test_export_omits_all_zero_acceleration(tmp_path):
    rep = make_degree2(randomise=False)
    with pytest.warns(UserWarning, match="exactly zero"):
        fields = rep.to_interchange()
    assert not {"ax", "ay", "az"} & set(fields)
    path = str(tmp_path / "splat_4d.ply")
    core_ply.write_sogst_ply(path, fields, 1.0, 3.0, 24.0)
    assert core_ply.read_sogst_ply(path)[0]["motion_degree"] == 1


def test_export_keeps_real_zeros_next_to_real_coefficients():
    rep = make_degree2(randomise=False)
    with torch.no_grad():
        rep.params["accelerations"][0, 1] = 1.0
    f = rep.to_interchange()
    assert f["ay"][0] != 0 and f["ax"][0] == 0 and len(f["ax"]) == rep.num_gaussians


def test_seed_degree_1_ply_into_degree_2_starts_from_zero():
    src = SpacetimeRepresentation(make_ctx(t0=1.0, t1=3.0))
    src.init_from_points(make_points(times=(1.0, 2.0, 3.0)))
    fields = src.to_interchange()
    rep = SpacetimeRepresentation(make_ctx(t0=1.0, t1=3.0), motion_degree=2)
    rep.init_from_interchange({}, fields)
    assert torch.all(rep.params["accelerations"] == 0)


def test_seed_degree_2_ply_into_degree_1_raises():
    fields = make_degree2().to_interchange()
    rep = SpacetimeRepresentation(make_ctx(t0=1.0, t1=3.0))
    with pytest.raises(ValueError, match="motion_degree"):
        rep.init_from_interchange({}, fields)


def test_resume_degree_1_state_at_degree_2_adds_zero_acceleration():
    src = SpacetimeRepresentation(make_ctx(t0=1.0, t1=3.0))
    src.init_from_points(make_points(times=(1.0, 2.0, 3.0)))
    rep = SpacetimeRepresentation(make_ctx(t0=1.0, t1=3.0), motion_degree=2)
    rep.load_state_dict(src.state_dict())
    a = rep.params["accelerations"]
    assert isinstance(a, torch.nn.Parameter) and torch.all(a == 0)
    assert a.shape == rep.params["means"].shape


def test_resume_degree_2_state_at_degree_1_raises():
    src = make_degree2()
    rep = SpacetimeRepresentation(make_ctx(t0=1.0, t1=3.0))
    with pytest.raises(ValueError, match="motion_degree"):
        rep.load_state_dict(src.state_dict())


def test_resume_degree_2_state_keeps_its_accelerations():
    src = make_degree2()
    rep = SpacetimeRepresentation(make_ctx(t0=1.0, t1=3.0), motion_degree=2)
    rep.load_state_dict(src.state_dict())
    assert torch.equal(rep.params["accelerations"], src.params["accelerations"])
