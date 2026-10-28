# cumuli-trainer

A 4D Gaussian splatting trainer for the Cumuli pipeline. It trains spacetime
Gaussians (position, velocity, temporal envelope) with gsplat and writes the
4D interchange PLY. Representations, densification strategies, loss terms and
the renderer are plugins chosen by name in the run config.

Clean-room: written from published papers, the gsplat API and Solipsist's own
format spec. See `PROVENANCE.md`.

## Install

Into an environment that already has PyTorch with CUDA and gsplat 1.5.3
(validated on torch 2.13.0+cu130, RTX 5090). `--no-deps` keeps pip from
replacing either:

```bash
pip install --no-deps "cumuli-core @ git+https://github.com/solipsist-studios/cumuli-core.git@v0.1.0"
pip install --no-deps "cumuli-trainer @ git+https://github.com/solipsist-studios/cumuli-trainer.git@v0.1.0"
pip install numpy scipy pyyaml tqdm pillow torchmetrics lpips
```

For development, install both editable from local checkouts instead
(`pip install --no-deps -e <path>`). The cumuli pipeline's
`scripts/setup_cumuli_env.sh` does this from its `deps/` submodules.

## Run

```bash
cumuli-train --config run.yaml [--resume ckpt_7000.pt] [--key value ...]
python -m cumuli_trainer --config run.yaml          # same CLI
cumuli-train --list-plugins
```

`--key value` overrides any config key; dotted keys reach nested sections
(`--strategy.cap_max 400000`, `--lr.velocities 3e-4`). Values are parsed as
YAML.

Minimal config:

```yaml
source_path: /path/to/dataset_4dgs      # transforms_train.json, transforms_test.json, points3d.ply
model_path: /path/to/output
time_duration: [0.0, 0.458333]          # seconds; default: span of the camera times
fps: 24                                 # default: worked out from the camera timestamps
iterations: 30000
sh_degree: 2
```

Full key list and defaults: the docstring of `cumuli_trainer/config.py`.
Unknown keys are an error.

Outputs in `model_path`:

- `splat_4d.ply`: the interchange PLY (cumuli `docs/sogst-format.md` section
  7) with `time_min`/`time_max` = `time_duration` and `fps`. Written at every
  `save_iterations` step (earlier steps also keep `splat_4d_<iter>.ply`).
- `ckpt_<iter>.pt` at `checkpoint_iterations` (default: the last step):
  parameters, Adam state, strategy state, sampler and RNG state, iteration,
  config.
- `test_<iter>.json`: per-view and mean PSNR, SSIM, LPIPS on the test cameras.
- `training_summary.json`: wall time (training and eval separately), splat
  count, peak GPU memory, test history.
- `config.json`: the resolved config.

`--resume ckpt_<iter>.pt` continues from that step with the saved strategy
state, so densification carries on if the step is inside the strategy's
refine window. The cap and every other strategy setting come from the
current config, never from the checkpoint. `init_ply: previous/splat_4d.ply`
seeds a new run from an interchange PLY and trains from step 1.

Console output the ComfyUI pack parses:

```
Training progress:  33%|###   | 10000/30000 [12:34<25:08, 13.3it/s, Loss=0.0123, PSNR=31.20]
[ITER 30000] Saving Gaussians
[ITER 30000] Evaluating test: PSNR 39.919 SSIM 0.99432 LPIPS 0.00631 (48 views)
```

Training PSNR in the bar is the current batch. The test line is the mean
over the dataset's test cameras.

## How a step works

Cameras of a synced rig share timestamps. Each step picks one instant and
`cameras_per_step` cameras from it, slices the Gaussians at that instant
once (`cumuli_core.spacetime.slice_at`, the same function the viewer and
evaluator use), rasterises all those cameras in one gsplat call, composites
render and ground truth onto the same background, and adds the loss terms.
Decoded frames stay on the GPU as uint8. The loop has no host syncs except
on progress-bar refreshes and on the strategy's refine steps.

SSIM runs on a per-view window around the ground-truth alpha (measured once
at load time) and counts every pixel outside as SSIM = 1. Where render and
ground truth are both background that is exact. A floater outside the
window is still charged by L1 and the opacity-mask term. On the 12-camera
parity set this cut the step cost from 24 ms to 5 ms per view.
`losses: {ssim: {weight: 0.2, crop: false}}` turns it off.

Time is normalised to [0, 1] over `time_duration` inside the trainer, so
learning rates do not depend on clip length. The PLY is written in seconds.

## Plugins

Four kinds, each chosen by name in the config:

| kind | config key | built-ins | interface |
|---|---|---|---|
| representation | `representation:` | `spacetime` | `representations/base.py` |
| strategy | `strategy:` | `mcmc` (default), `default` | `strategies/base.py` |
| loss | `losses:` | `l1`, `ssim`, `opacity_mask`, `opacity_reg`, `scale_reg` | `losses/base.py` |
| renderer | `renderer:` | `gsplat` | `renderers/base.py` |

A plugin section is a name or a mapping with `name` plus the plugin's own
constructor arguments:

```yaml
strategy: {name: mcmc, cap_max: 300000, noise_lr: 5.0e+5}
losses: {l1: 0.8, ssim: 0.2, opacity_mask: {weight: 0.005, eps: 1.0e-6}}
```

A separate package adds plugins through entry points, with no change to the
trainer:

```toml
[project.entry-points."cumuli_trainer.strategies"]
mrnf = "my_private_pkg.mrnf:MRNFStrategy"
```

then `strategy: {name: mrnf, ...}`. The groups are
`cumuli_trainer.representations`, `cumuli_trainer.strategies`,
`cumuli_trainer.losses` and `cumuli_trainer.renderers`. A strategy edits the
representation's parameter dict and its per-parameter Adam optimizers in
place (`strategies/ops.py` has the row operations). It sees the renderer's
info dict, an `opacity()` callable and, when bounds are on, an `in_bounds()`
callable. It can declare `default_losses` (MCMC adds its opacity and
scale regularisers this way). The config's `losses` override them, and a
weight of 0 removes a term.

## Spacetime representation

Parameters use gsplat's names (`means` at `t_center`, `quats`, log `scales`,
logit `opacities`, `sh0`, `shN`) plus `velocities`, `t_centers` and
`t_log_sigmas`. The strategy ops edit every tensor in the dict, so the
temporal ones follow: a clone or relocated copy inherits its source's motion
and window, and both halves of a split keep the parent's motion and window
(the split offset is applied at `t_center`, where `means` lives).

Initialisation: positions, colours and times from `points3d.ply`; spatial
scale from the mean distance to the 3 nearest neighbours, computed within
each timestamp so repeated copies of a static region do not shrink each
other; velocity 0; `t_sigma = sqrt(duration / t_init_div)` seconds with
`t_init_div` 100; peak opacity 0.1.

## Strategies

`mcmc` (default) follows Kheradmand et al. 2024. Every 100 steps from 500 to
25000, dead Gaussians (opacity at or below 0.005, or outside the scene
bounds) move onto live ones, and the population grows 5% up to `cap_max`.
Every step adds covariance-shaped exploration noise to nearly transparent
Gaussians.

`default` is adaptive density control (3DGS paper, section 5.2). Statistics
accumulate densely, without index lists, so there is no per-step sync.

## Learning rates

| parameter | LR | note |
|---|---|---|
| means | 1.6e-4 x scene scale | decays exponentially to 1% at the last step |
| scales | 5e-3 | |
| quats | 1e-3 | |
| opacities | 5e-2 | |
| sh0 | 2.5e-3 | |
| shN | 1.25e-4 | sh0 / 20 |
| velocities | 1.6e-4 x scene scale | units per normalised time; decays like means |
| t_centers | 1e-3 | normalised time |
| t_log_sigmas | 5e-3 | |

The spatial values are the 3DGS paper's (section 7.1) as used by gsplat's
simple_trainer. Scene scale is 1.1 x the largest camera distance from the
rig centre. The temporal values were chosen here. Velocity uses the position
LR: a velocity step moves a Gaussian by (step x dt), and dt within its window
is about one sigma (0.15 normalised at the default init), so a velocity step
is a fraction of a position step and motion cannot outrun position.
`t_centers` at 1e-3 moves a window by about a hundredth of a frame per step
on a 12-frame clip. `t_log_sigmas` uses the scale LR because both are
log-space widths.

## Tests

```bash
conda run -n cumuli python -m pytest -q
```

GPU tests are marked `gpu` and skipped without CUDA.

## Licence

Licensed under the PolyForm Noncommercial License 1.0.0 (see `LICENSE.md`).
Commercial licences: https://solipsist.studio
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
