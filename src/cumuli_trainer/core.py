# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""The shared cumuli_core library, under the names the trainer uses.

``from cumuli_trainer.core import ply, cameras, spacetime, dataset, metrics, render``
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from cumuli_core import cameras, dataset, metrics, ply, render, spacetime  # noqa: F401

try:
    CORE_VERSION = version("cumuli-core")
except PackageNotFoundError:  # imported from a source tree, not installed
    CORE_VERSION = "source"
