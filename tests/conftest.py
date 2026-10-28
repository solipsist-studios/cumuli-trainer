# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
import pytest
import torch

from cumuli_trainer import load_builtin_plugins

load_builtin_plugins()


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
