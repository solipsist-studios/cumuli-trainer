# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
import math

import pytest
import torch

from cumuli_trainer.strategies import ops
from cumuli_trainer.strategies.base import StrategyContext
from cumuli_trainer.strategies.default import DefaultStrategy
from cumuli_trainer.strategies.mcmc import MCMCStrategy


def make_params(n=6, device="cpu"):
    g = torch.Generator().manual_seed(0)
    P = torch.nn.Parameter
    params = {
        "means": P(torch.randn(n, 3, generator=g).to(device)),
        "quats": P(torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1).to(device)),
        "scales": P(torch.full((n, 3), math.log(0.01)).to(device)),
        "opacities": P(torch.zeros(n).to(device)),
        "sh0": P(torch.zeros(n, 1, 3).to(device)),
        "shN": P(torch.zeros(n, 3, 3).to(device)),
        "velocities": P(torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3).to(device)),
        "t_centers": P(torch.linspace(0, 1, n).to(device)),
        "t_log_sigmas": P(torch.full((n,), -2.0).to(device)),
    }
    opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    # give every optimizer a state
    loss = sum((v ** 2).sum() for v in params.values())
    loss.backward()
    for o in opts.values():
        o.step()
    return params, opts


def ctx(opacity=None, device="cpu"):
    return StrategyContext(scene_scale=1.0, iterations=1000, device=torch.device(device),
                           opacity=opacity or (lambda: torch.full((6,), 0.5)))


def fake_info(n_cams=2, n=6):
    m2d = torch.zeros(n_cams, n, 2, requires_grad=True)
    m2d.grad = torch.ones(n_cams, n, 2) * 1e-3
    radii = torch.ones(n_cams, n, 2, dtype=torch.int32) * 5
    return {"means2d": m2d, "radii": radii, "width": 100, "height": 100, "n_cameras": n_cams}


def test_every_projected_gaussian_counts():
    st = DefaultStrategy()
    params, opts = make_params()
    st.initialize(params, opts, ctx())
    info = fake_info()
    info["radii"][0, 2] = 0                      # culled in camera 0
    st.accumulate(info)
    assert st.state["count"].tolist() == [2, 2, 1, 2, 2, 2]


def test_prune_uses_opacity():
    params, opts = make_params()
    opacity = torch.tensor([0.5, 0.5, 0.001, 0.5, 0.5, 0.5])
    st = DefaultStrategy(refine_start=0, refine_every=1, reset_every=10**9)
    st.initialize(params, opts, ctx(lambda: opacity[: params["means"].shape[0]]))
    n = st._prune(params, opts, step=1)
    assert n == 1 and params["means"].shape[0] == 5
    assert opts["means"].param_groups[0]["params"][0] is params["means"]


def test_ops_keep_temporal_rows_and_optimizer_state_aligned():
    params, opts = make_params()
    vel_before = params["velocities"].detach().clone()
    tc_before = params["t_centers"].detach().clone()
    mask = torch.tensor([False, True, False, False, True, False])
    ops.duplicate(params, opts, mask)
    assert params["velocities"].shape[0] == 8
    assert torch.equal(params["velocities"][6:], vel_before[[1, 4]])
    assert torch.equal(params["t_centers"][6:], tc_before[[1, 4]])
    for k, p in params.items():
        st = opts[k].state[p]
        assert st["exp_avg"].shape == p.shape
        assert torch.all(st["exp_avg"][6:] == 0)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[0] = True
    scale0 = params["scales"][0].detach().clone()
    ops.split(params, opts, mask)
    assert params["means"].shape[0] == 9
    # both children keep the parent's motion and window
    assert torch.equal(params["velocities"][-2:], vel_before[[0, 0]])
    assert torch.equal(params["t_centers"][-2:], tc_before[[0, 0]])
    assert torch.allclose(params["scales"][-1], scale0 - math.log(1.6))
    ops.remove(params, opts, torch.arange(9) < 3)
    assert params["means"].shape[0] == 6
    for k, p in params.items():
        assert opts[k].state[p]["exp_avg_sq"].shape == p.shape


def test_opacity_reset_clamps_and_clears():
    params, opts = make_params()
    with torch.no_grad():
        params["opacities"].fill_(3.0)
    ops.clamp_opacity(params, opts, 0.01)
    assert torch.allclose(torch.sigmoid(params["opacities"]), torch.full((6,), 0.01), atol=1e-6)
    assert torch.all(opts["opacities"].state[params["opacities"]]["exp_avg"] == 0)


@pytest.mark.gpu
def test_mcmc_relocates_dead_gaussians():
    dev = "cuda"
    params, opts = make_params(device=dev)
    opacity = torch.tensor([0.5, 0.5, 0.0001, 0.5, 0.5, 0.5], device=dev)
    st = MCMCStrategy(cap_max=6, refine_start=0, refine_every=1, noise_lr=0)
    st.initialize(params, opts, ctx(lambda: opacity[: params["means"].shape[0]], device=dev))
    before = params["velocities"].detach().clone()
    st.step_post_backward(params, opts, 1, {}, lr=1e-3)
    assert st.last["relocated"] == 1
    assert st.last["added"] == 0  # at the cap
    # the dead row now carries a live row's motion
    assert not torch.equal(params["velocities"][2], before[2])
    assert any(torch.equal(params["velocities"][2], before[i]) for i in (0, 1, 3, 4, 5))


@pytest.mark.gpu
def test_mcmc_grows_to_cap():
    dev = "cuda"
    params, opts = make_params(n=6, device=dev)
    st = MCMCStrategy(cap_max=7, refine_start=0, refine_every=1, noise_lr=0, grow_rate=1.5)
    st.initialize(params, opts, ctx(lambda: torch.full((params["means"].shape[0],), 0.5, device=dev), dev))
    st.step_post_backward(params, opts, 1, {}, lr=1e-3)
    assert params["means"].shape[0] == 7
    assert params["t_log_sigmas"].shape[0] == 7
    assert opts["t_log_sigmas"].state[params["t_log_sigmas"]]["exp_avg"].shape[0] == 7


def test_mcmc_recycles_gaussians_outside_the_scene_bounds():
    """Opaque enough to live, but drifted out of the scene: still dead, and
    never a copy source (Heidi, reg off: thousands ended 10-800 units out)."""
    dev = "cuda"
    params, opts = make_params(device=dev)
    peak = torch.full((6,), 0.5, device=dev)
    inside = torch.tensor([True, True, True, False, True, True], device=dev)
    st = MCMCStrategy(cap_max=6, refine_start=0, refine_every=1, noise_lr=0)
    c = ctx(lambda: peak[: params["means"].shape[0]], device=dev)
    c.in_bounds = lambda: inside[: params["means"].shape[0]]
    st.initialize(params, opts, c)
    before = params["means"].detach().clone()
    st.step_post_backward(params, opts, 1, {}, lr=1e-3)
    assert st.last["relocated"] == 1
    assert any(torch.equal(params["means"][3], before[i]) for i in (0, 1, 2, 4, 5))


def test_mcmc_never_samples_out_of_bounds_sources_when_growing():
    dev = "cuda"
    params, opts = make_params(device=dev)
    inside = torch.tensor([True, False, False, False, False, False], device=dev)
    st = MCMCStrategy(cap_max=12, refine_start=0, refine_every=1, noise_lr=0)
    c = ctx(lambda: torch.full((params["means"].shape[0],), 0.5, device=dev), device=dev)
    c.in_bounds = lambda: torch.cat([inside, torch.ones(params["means"].shape[0] - 6,
                                                       dtype=torch.bool, device=dev)])
    st.initialize(params, opts, c)
    before = params["means"].detach().clone()
    st.step_post_backward(params, opts, 1, {}, lr=1e-3)
    # the five out-of-bounds rows were recycled onto row 0, and growth copies row 0 too
    for row in params["means"][1:]:
        assert torch.equal(row, before[0])
