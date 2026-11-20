# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Per-view loss weights reach the batch the trainer draws."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cumuli_trainer.core import cameras as core_cameras
from cumuli_trainer.data import TrainData

W, H = 16, 12


class FakeDataset:
    def load_image_uint8(self, cam):
        img = np.zeros((H, W, 4), np.uint8)
        img[3:9, 4:12] = (200, 100, 50, 255)
        return img


def cam(i, t, weight=None):
    kw = {} if weight is None else {"loss_weight": weight}
    return core_cameras.Camera(name=f"c{i}_t{t}", image_path=f"x/{i}", w2c=np.eye(4), K=np.diag([20.0, 20.0, 1.0]),
                  width=W, height=H, time=t, label=str(i), **kw)


def make(cams, cps=2):
    return TrainData(FakeDataset(), cams, torch.device("cpu"), cps, data_device="cpu",
                     log=lambda *a: None)


def test_batches_carry_each_views_weight():
    weights = [1.0, 0.5, 0.25, 0.0]
    cams = [cam(i, 0.0, w) for i, w in enumerate(weights)]
    data = make(cams)
    assert data.weighted
    for _ in range(6):  # across epoch boundaries
        batch = data.next()
        assert batch.weights is not None and batch.weights.shape == batch.index.shape
        assert batch.weights.tolist() == [weights[i] for i in batch.index.tolist()]


def test_a_dataset_without_weights_sends_none():
    data = make([cam(i, 0.0) for i in range(4)])
    assert not data.weighted
    assert data.next().weights is None


def test_all_ones_counts_as_unweighted():
    data = make([cam(i, 0.0, 1.0) for i in range(4)])
    assert not data.weighted and data.next().weights is None


def test_weights_are_reported_in_the_log():
    lines = []
    TrainData(FakeDataset(), [cam(0, 0.0, 1.0), cam(1, 0.0, 0.5)], torch.device("cpu"), 2,
              data_device="cpu", log=lines.append)
    assert any("per-view loss weights: 1 of 2" in line and "min 0.5" in line for line in lines)


def test_an_old_core_without_the_field_is_an_error_not_a_silent_full_weight():
    old = [SimpleNamespace(width=W, height=H, time=0.0, image_path="x") for _ in range(2)]
    with pytest.raises(RuntimeError, match="loss_weight"):
        make(old)
