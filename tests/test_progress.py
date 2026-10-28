# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Progress lines against the cumuli ComfyUI pack's parser.

The regular expressions are copied verbatim from
comfyui-cumuli/cumuli_bridge/train.py. If the pack changes them, update
these copies and keep the trainer's output matching.
"""

import io
import re

from cumuli_trainer import progress

_TRAIN_BAR = re.compile(r"Training progress:\s*\d{1,3}%\|.*?\|\s*(\d+)\s*/\s*(\d+)")
_PSNR = re.compile(r"PSNR=([\d.]+)")
_LOSS = re.compile(r"Loss=([\d.]+)")
_SAVING = re.compile(r"\[ITER (\d+)\] Saving Gaussians")


def _lines(text):
    # the pack splits on both \n and \r
    return [s for s in re.split(r"[\r\n]", text) if s.strip()]


def test_bar_lines_parse():
    buf = io.StringIO()
    bar = progress.make_bar(30000, file=buf)
    bar.update(10000)
    bar.set_postfix(progress.postfix(0.0123, 31.2), refresh=True)
    bar.update(5000)
    bar.close()
    hits = [m for m in (_TRAIN_BAR.search(l) for l in _lines(buf.getvalue())) if m]
    assert hits, buf.getvalue()
    last = [l for l in _lines(buf.getvalue()) if _TRAIN_BAR.search(l)][-1]
    m = _TRAIN_BAR.search(last)
    assert (int(m.group(1)), int(m.group(2))) == (15000, 30000)
    assert float(_PSNR.search(last).group(1)) == 31.2
    assert float(_LOSS.search(last).group(1)) == 0.0123
    assert last.startswith("Training progress:")


def test_bar_from_resume_offset():
    buf = io.StringIO()
    bar = progress.make_bar(30000, initial=20000, file=buf)
    bar.update(10)
    bar.close()
    last = [l for l in _lines(buf.getvalue()) if _TRAIN_BAR.search(l)][-1]
    assert _TRAIN_BAR.search(last).group(1) == "20010"


def test_saving_line():
    m = _SAVING.search(progress.saving_line(30000))
    assert m and m.group(1) == "30000"
    assert progress.saving_line(7000) == "[ITER 7000] Saving Gaussians"


def test_test_line_always_has_lpips_beside_psnr():
    line = progress.test_line(30000, 39.4, 0.99286, 0.00664, 48)
    assert "PSNR 39.400" in line and "LPIPS 0.00664" in line and "SSIM 0.99286" in line
