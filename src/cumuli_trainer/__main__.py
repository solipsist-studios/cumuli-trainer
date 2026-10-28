# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""``python -m cumuli_trainer --config run.yaml``: same CLI as ``cumuli-train``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
