# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""cumuli-trainer: a 4D Gaussian splatting trainer with pluggable
representations, strategies, losses and renderers."""

__version__ = "0.1.0"


def load_builtin_plugins() -> None:
    """Import the built-in plugin packages so they register themselves."""
    from . import losses, renderers, representations, strategies  # noqa: F401
