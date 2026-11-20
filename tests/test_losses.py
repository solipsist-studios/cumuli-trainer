# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
import pytest
import torch

from cumuli_trainer.losses.base import LossInputs
from cumuli_trainer.losses.masks import OpacityMaskLoss
from cumuli_trainer.losses.photometric import L1Loss, SSIMLoss
from cumuli_trainer.losses.ssim import ssim


def inputs(pred, gt, pred_a=None, gt_a=None, crop=None):
    c, h, w, _ = pred.shape
    pa = pred_a if pred_a is not None else torch.zeros(c, h, w, 1)
    ga = gt_a if gt_a is not None else torch.zeros(c, h, w, 1)
    return LossInputs(pred, pa, gt, ga, {}, None, {}, 0, extras={"crop": crop} if crop else {})


def test_ssim_identity():
    x = torch.rand(2, 3, 40, 50)
    assert torch.allclose(ssim(x, x), torch.tensor(1.0), atol=1e-6)
    assert ssim(x, torch.rand(2, 3, 40, 50)) < 0.5


def test_cropped_ssim_equals_full_when_outside_matches():
    torch.manual_seed(0)
    gt = torch.zeros(2, 64, 80, 3)
    pred = torch.zeros(2, 64, 80, 3)
    # subject inside each view's crop window, different positions per view
    gt[0, 10:30, 10:40] = torch.rand(20, 30, 3)
    gt[1, 30:50, 35:65] = torch.rand(20, 30, 3)
    pred[0, 10:30, 10:40] = torch.rand(20, 30, 3)
    pred[1, 30:50, 35:65] = torch.rand(20, 30, 3)
    full = SSIMLoss(crop=False)(inputs(pred, gt))
    crop = ((48, 48), [(0, 1), (16, 26)])  # 8+ px margin around each subject
    cropped = SSIMLoss(crop=True)(inputs(pred, gt, crop=crop))
    assert torch.allclose(full, cropped, rtol=1e-4, atol=1e-6)


def test_l1():
    a = torch.zeros(1, 4, 4, 3)
    b = torch.full((1, 4, 4, 3), 0.5)
    assert L1Loss()(inputs(a, b)).item() == 0.5


def test_opacity_mask_only_charges_background():
    pa = torch.full((1, 4, 4, 1), 0.5)
    fg = torch.ones(1, 4, 4, 1)
    rgb = torch.zeros(1, 4, 4, 3)
    assert OpacityMaskLoss()(inputs(rgb, rgb, pa, fg)).item() == 0.0
    bg = torch.zeros(1, 4, 4, 1)
    v = OpacityMaskLoss()(inputs(rgb, rgb, pa, bg)).item()
    assert abs(v - (-torch.log(torch.tensor(0.5 + 1e-6)).item())) < 1e-5


def test_aniso_reg_charges_only_the_ratio_above_the_limit():
    import math
    from cumuli_trainer.losses.regularizers import AnisoReg
    scales = torch.log(torch.tensor([[1.0, 1.0, 1.0],      # ratio 1: free
                                     [10.0, 1.0, 1.0],     # ratio 10: at the limit
                                     [100.0, 1.0, 2.0]]))  # ratio 100: costs log(10)
    term = AnisoReg(max_ratio=10.0)
    inp = type("I", (), {"params": {"scales": scales}})()
    assert float(term(inp)) == pytest.approx(math.log(10.0) / 3, rel=1e-5)
    with pytest.raises(ValueError):
        AnisoReg(max_ratio=1.0)


def test_sh_reg_charges_only_higher_bands():
    from cumuli_trainer.losses.regularizers import SHReg
    sh_n = torch.zeros(2, 8, 3)
    sh_n[0, 0, 0] = 2.0  # one coefficient of 2 on one Gaussian
    inp = type("I", (), {"params": {"shN": sh_n, "sh0": torch.ones(2, 1, 3) * 5,
                                    "means": torch.zeros(2, 3)}})()
    assert float(SHReg()(inp)) == pytest.approx(4.0 / 2)


# --- per-view loss weights (LossInputs.view_weights) ---------------------

def weighted(pred, gt, weights, **kw):
    inp = inputs(pred, gt, **kw)
    inp.view_weights = torch.tensor(weights, dtype=torch.float32)
    return inp


def two_views(seed=0):
    torch.manual_seed(seed)
    gt = torch.zeros(2, 64, 80, 3)
    pred = torch.zeros(2, 64, 80, 3)
    gt[0, 10:30, 10:40] = torch.rand(20, 30, 3)
    gt[1, 30:50, 35:65] = torch.rand(20, 30, 3)
    pred[0, 10:30, 10:40] = torch.rand(20, 30, 3)
    pred[1, 30:50, 35:65] = torch.rand(20, 30, 3)
    return pred, gt


CROP = ((48, 48), [(0, 1), (16, 26)])


@pytest.mark.parametrize("make_term, crop", [
    (lambda: L1Loss(), None),
    (lambda: SSIMLoss(crop=False), None),
    (lambda: SSIMLoss(crop=True), CROP),
])
def test_unit_weights_match_no_weights(make_term, crop):
    pred, gt = two_views()
    term = make_term()
    plain = term(inputs(pred, gt, crop=crop))
    unit = term(weighted(pred, gt, [1.0, 1.0], crop=crop))
    assert torch.allclose(plain, unit, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("make_term, crop", [
    (lambda: L1Loss(), None),
    (lambda: SSIMLoss(crop=False), None),
    (lambda: SSIMLoss(crop=True), CROP),
])
def test_weighted_loss_is_the_weighted_mean_of_each_views_own_loss(make_term, crop):
    pred, gt = two_views(1)
    term = make_term()
    w = [1.0, 0.25]
    got = term(weighted(pred, gt, w, crop=crop))
    singles = []
    for i in range(2):
        c = ((CROP[0]), [CROP[1][i]]) if crop else None
        singles.append(term(inputs(pred[i:i + 1], gt[i:i + 1], crop=c)))
    expected = (w[0] * singles[0] + w[1] * singles[1]) / 2
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("make_term, crop", [
    (lambda: L1Loss(), None),
    (lambda: SSIMLoss(crop=False), None),
    (lambda: SSIMLoss(crop=True), CROP),
])
def test_zero_weight_removes_the_view_from_the_colour_loss(make_term, crop):
    pred, gt = two_views(2)
    term = make_term()
    base = term(weighted(pred, gt, [1.0, 0.0], crop=crop))
    pred2, gt2 = pred.clone(), gt.clone()
    pred2[1] = torch.rand_like(pred2[1])  # wreck the zero-weight view
    gt2[1] = torch.rand_like(gt2[1])
    assert torch.allclose(base, term(weighted(pred2, gt2, [1.0, 0.0], crop=crop)))


def test_weight_scales_the_gradient_of_its_view():
    pred = torch.zeros(2, 4, 4, 3, requires_grad=True)
    gt = torch.ones(2, 4, 4, 3)
    L1Loss()(weighted(pred, gt, [1.0, 0.5])).backward()
    g0, g1 = pred.grad[0].abs().sum(), pred.grad[1].abs().sum()
    assert torch.allclose(g1, 0.5 * g0)


def test_silhouette_term_ignores_view_weights():
    pa = torch.full((2, 4, 4, 1), 0.5)
    bg = torch.zeros(2, 4, 4, 1)
    rgb = torch.zeros(2, 4, 4, 3)
    plain = OpacityMaskLoss()(inputs(rgb, rgb, pa, bg))
    assert torch.equal(plain, OpacityMaskLoss()(weighted(rgb, rgb, [0.1, 0.0], pred_a=pa, gt_a=bg)))
