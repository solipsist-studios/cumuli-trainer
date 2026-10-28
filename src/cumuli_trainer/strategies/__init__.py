# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Densification strategies. Importing this package registers the built-ins."""

from . import default, mcmc  # noqa: F401
from .base import Strategy, StrategyContext  # noqa: F401
