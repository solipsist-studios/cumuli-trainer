# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
from types import SimpleNamespace

import torch

from cumuli_trainer.data import Batch, capture_time


def test_batch_times_default_to_the_instant():
    b = Batch(time=0.5, index=torch.tensor([3, 1]), images=None, crop_size=(16, 16), crop_origins=[(0, 0)] * 2)
    assert b.times == [0.5, 0.5]


def test_capture_time_falls_back_to_nominal_time():
    assert capture_time(SimpleNamespace(time=0.25)) == 0.25
    assert capture_time(SimpleNamespace(time=0.25, capture_time=0.254)) == 0.254
