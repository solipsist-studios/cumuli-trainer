# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Loss terms. Importing this package registers the built-ins."""

from . import masks, photometric, regularizers  # noqa: F401
from .base import LossInputs, LossTerm  # noqa: F401
