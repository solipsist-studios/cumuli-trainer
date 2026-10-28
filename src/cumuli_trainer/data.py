# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Training images, camera tensors and the time-batched sampler.

Decoded RGBA frames are kept as uint8 (a quarter of float32), on the GPU when
they fit, so a step gathers its batch with one indexing op and no host
transfer. Cameras of a synced rig share timestamps: each step takes one
instant and ``cameras_per_step`` cameras from it, so the Gaussians are sliced
once and rasterised into all of those cameras in one call.

The sampler walks epochs. In each epoch every instant's cameras are shuffled
and chunked into batches of ``cameras_per_step``, and the batches of all
instants are shuffled together, so every training view is used once per
epoch. Index tensors for the whole epoch are built up front and moved to the
device in one copy.

Each frame's foreground box (alpha > 0, plus a margin) is measured once at
load time. A batch carries one crop window per view, all the same size, so
losses that are expensive per pixel (SSIM) can run on the crop without a
host synchronisation during training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .core import cameras as core_cameras
from .core import render as core_render

# data_device "auto" keeps the decoded images on the GPU when this much
# memory stays free beside them. Heidi (610 views at 2656x2324, 14.03 GiB
# of images) peaked at 16.2 GiB with a 300k cap and 16.7 GiB with 1M, so
# training needs ~3 GiB on top. Host-resident images cost 4x the time
# there (20.5 vs 4.8 min) because every step waits on the copy.
GPU_HEADROOM_BYTES = 8 * 2**30


def _load_uint8(dataset, cam) -> np.ndarray:
    if hasattr(dataset, "load_image_uint8"):
        return dataset.load_image_uint8(cam)
    if hasattr(dataset, "image_file"):
        # decode straight to uint8: no float32 copy of every frame
        from PIL import Image
        with Image.open(dataset.image_file(cam)) as im:
            return np.array(im.convert("RGBA"), dtype=np.uint8)
    img = dataset.load_image(cam)
    if img.shape[-1] == 3:
        img = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def foreground_box(alpha: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
    """``(y0, y1, x0, x1)`` around ``alpha > 0`` grown by ``margin``; the whole
    frame when alpha is empty or full."""
    h, w = alpha.shape
    rows = np.nonzero(alpha.any(axis=1))[0]
    cols = np.nonzero(alpha.any(axis=0))[0]
    if len(rows) == 0:
        return 0, h, 0, w
    return (max(0, rows[0] - margin), min(h, rows[-1] + 1 + margin),
            max(0, cols[0] - margin), min(w, cols[-1] + 1 + margin))


@dataclass
class Batch:
    time: float  # seconds
    index: torch.Tensor  # [C] camera indices on the device
    images: torch.Tensor  # [C, H, W, 4] uint8 on the device
    crop_size: Tuple[int, int]  # (h, w) shared by every view
    crop_origins: List[Tuple[int, int]]  # per view (y0, x0)


class TrainData:
    def __init__(self, dataset, cams: list, device: torch.device, cameras_per_step: int,
                 data_device: str = "auto", seed: int = 0, crop_margin: int = 32, log=print):
        self.cams = cams
        self.device = device
        self.cameras_per_step = int(cameras_per_step)
        self.rng = random.Random(seed)
        sizes = {(c.width, c.height) for c in cams}
        if len(sizes) != 1:
            raise ValueError(f"training cameras have several image sizes {sorted(sizes)}; "
                             "time-batched rendering needs one size")
        self.width, self.height = sizes.pop()
        n = len(cams)
        nbytes = n * self.width * self.height * 4
        where = data_device
        if where == "auto":
            where = "cpu"
            if device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(device)
                if free - nbytes >= GPU_HEADROOM_BYTES:
                    where = "cuda"
        log(f"Loading Training Cameras: {n} views at {self.width}x{self.height}, "
            f"{nbytes / 2**30:.2f} GiB as uint8 on {where}")
        store = torch.empty((n, self.height, self.width, 4), dtype=torch.uint8,
                            device=device if where == "cuda" else "cpu",
                            pin_memory=(where == "cpu" and device.type == "cuda"))
        self.boxes = np.zeros((n, 4), dtype=np.int64)
        for i, cam in enumerate(cams):
            img = _load_uint8(dataset, cam)
            if img.shape[:2] != (self.height, self.width):
                raise ValueError(f"{cam.image_path}: image is {img.shape[1]}x{img.shape[0]}, "
                                 f"transforms say {self.width}x{self.height}")
            self.boxes[i] = foreground_box(img[..., 3] > 0, crop_margin)
            store[i].copy_(torch.from_numpy(img))
        self.images = store
        self.on_device = where == "cuda"
        self.viewmats, self.Ks = core_render.camera_tensors(cams, device)
        groups = core_cameras.group_by_time(cams)
        index = {id(c): i for i, c in enumerate(cams)}
        self.times: List[float] = []
        self.groups: List[List[int]] = []
        for t, members in groups.items():
            self.times.append(float(members[0].time))
            self.groups.append([index[id(c)] for c in members])
        self._epoch: list = []
        self._pos = 0
        area = ((self.boxes[:, 1] - self.boxes[:, 0]) * (self.boxes[:, 3] - self.boxes[:, 2])).mean()
        log(f"foreground boxes cover {100 * area / (self.width * self.height):.1f}% of a frame on average")

    def _crop(self, members: List[int]):
        b = self.boxes[members]
        hc = int(min(self.height, -(-int((b[:, 1] - b[:, 0]).max()) // 16) * 16))
        wc = int(min(self.width, -(-int((b[:, 3] - b[:, 2]).max()) // 16) * 16))
        origins = []
        for y0, y1, x0, x1 in b:
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            oy = int(min(max(cy - hc // 2, 0), self.height - hc))
            ox = int(min(max(cx - wc // 2, 0), self.width - wc))
            origins.append((oy, ox))
        return (hc, wc), origins

    def _new_epoch(self) -> None:
        cps = self.cameras_per_step
        batches = []
        for t, members in zip(self.times, self.groups):
            m = list(members)
            self.rng.shuffle(m)
            for k in range(0, len(m), cps):
                chunk = m[k:k + cps]
                if len(chunk) < cps and len(m) >= cps:
                    # top up a short tail from the same instant so every
                    # batch has the same camera count (same cost per step)
                    pool = [x for x in m if x not in chunk]
                    chunk = chunk + self.rng.sample(pool, cps - len(chunk))
                batches.append((t, chunk))
        self.rng.shuffle(batches)
        width = max(len(c) for _, c in batches)
        idx = torch.tensor([c + [c[-1]] * (width - len(c)) for _, c in batches], dtype=torch.long)
        idx_dev = idx.to(self.device, non_blocking=True)
        idx_img = idx_dev if self.on_device else idx
        self._epoch = []
        for b, (t, chunk) in enumerate(batches):
            size, origins = self._crop(chunk)
            self._epoch.append((t, idx_dev[b, :len(chunk)], idx_img[b, :len(chunk)], size, origins))
        self._pos = 0

    def next(self) -> Batch:
        if self._pos >= len(self._epoch):
            self._new_epoch()
        t, idx, idx_img, size, origins = self._epoch[self._pos]
        self._pos += 1
        imgs = self.images[idx_img]
        if not self.on_device:
            imgs = imgs.to(self.device, non_blocking=True)
        return Batch(time=t, index=idx, images=imgs, crop_size=size, crop_origins=origins)

    def state_dict(self) -> dict:
        return {"rng": self.rng.getstate()}

    def load_state_dict(self, state: Optional[dict]) -> None:
        if state and "rng" in state:
            self.rng.setstate(state["rng"])
