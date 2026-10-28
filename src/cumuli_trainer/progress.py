# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Console output the cumuli ComfyUI pack parses.

The pack (``cumuli_bridge/train.py``) reads the trainer's merged
stdout/stderr, splitting on both ``\\n`` and ``\\r``, and matches:

    Training progress:  33%|###   | 10000/30000 [12:34<25:08, 13.3it/s, Loss=0.0123, PSNR=31.20]
    [ITER 30000] Saving Gaussians

Those two shapes must not change. ``tests/test_progress.py`` checks them
against copies of the pack's regular expressions.
"""

from __future__ import annotations

import sys

from tqdm import tqdm

BAR_DESC = "Training progress"


def make_bar(total: int, initial: int = 0, file=None) -> tqdm:
    return tqdm(total=total, initial=initial, desc=BAR_DESC, ascii=True,
                file=file if file is not None else sys.stdout, dynamic_ncols=False,
                ncols=120, mininterval=0.5, smoothing=0.05)


def postfix(loss: float, psnr: float) -> dict:
    return {"Loss": f"{loss:.4f}", "PSNR": f"{psnr:.2f}"}


def saving_line(iteration: int) -> str:
    return f"[ITER {iteration}] Saving Gaussians"


def test_line(iteration: int, psnr: float, ssim: float, lpips: float, n_views: int) -> str:
    return (f"[ITER {iteration}] Evaluating test: PSNR {psnr:.3f} SSIM {ssim:.5f} "
            f"LPIPS {lpips:.5f} ({n_views} views)")
