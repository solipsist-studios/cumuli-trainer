# Provenance

cumuli-trainer is a clean-room implementation. This file says what each
module was written from.

## Sources consulted

- Papers:
  - Kerbl, Kopanas, Leimkühler, Drettakis, "3D Gaussian Splatting for
    Real-Time Radiance Field Rendering", SIGGRAPH 2023.
  - Yang, Yang, Pan, Zhang, "Real-time Photorealistic Dynamic Scene
    Representation and Rendering with 4D Gaussian Splatting", ICLR 2024.
  - Li, Chen, Li, Xu, "Spacetime Gaussian Feature Splatting for Real-Time
    Dynamic View Synthesis", CVPR 2024.
  - Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo",
    2024.
  - Xie et al., "PhysGaussian: Physics-Integrated 3D Gaussians for
    Generative Dynamics", CVPR 2024 (the anisotropy regulariser behind
    `losses/regularizers.py` `aniso_reg`).
  - Wang, Bovik, Sheikh, Simoncelli, "Image Quality Assessment: From Error
    Visibility to Structural Similarity", IEEE TIP 2004 (SSIM).
- gsplat 1.5.3 (Apache-2.0): the public API (`rasterization`,
  `compute_relocation`, `quat_scale_to_covar_preci`), and the source of
  `gsplat/strategy/{default,mcmc,ops}.py` and `examples/simple_trainer.py`
  read for conventions (parameter key names, how strategy ops edit Adam
  state, gradient normalisation, default thresholds and learning rates).
  No gsplat code was copied.
- PyTorch, torchmetrics and lpips documentation.
- Solipsist's own code and specs: cumuli `docs/sogst-format.md`,
  `docs/pipeline.md`, `scripts/eval_render.py`, `scripts/sogst_ply.py`,
  `scripts/sogst_io.py`, `scripts/sogst_pack.py`; the cumuli-core library;
  comfyui-cumuli `cumuli_bridge/train.py` and `model.py`.

## Sources not consulted

The forbidden sources were not consulted: nothing under
`~/Dev/github/cumuli/deps/`, no OMG4, fudan-zvg 4D Gaussian Splatting,
Spacetime Gaussians, LichtFeld Studio, graphdeco-inria gaussian-splatting,
diff-gaussian-rasterization or simple-knn code, no file carrying an Inria or
GRAPHDECO header, and cumuli `scripts/bake_sogst.py` was not opened. For the
OMG4 baseline only its outputs were used: the run's `gs4d_config.yaml`
(a Solipsist template), `experiment.json`, `eval_4d.json`, the `.sogst`
splat count in `meta.json`, and file modification times for its training
duration.

## Degree-2 motion

Pending legal review. The change was written from `docs/sogst-format.md`
sections 1, 4.8 and 7, the cumuli-core API and the existing `velocities` code
in this repository.

Also read while planning it, to check compatibility, in the Solipsist forks:
the accel handling in supersplat-viewer `src/core/sogst-motion.ts` and related
`sogst-*` files; the accel handling in splat-transform
`src/lib/writers/write-sogst.ts` and `src/lib/data-table/transform.ts`; and the
message and file list, not the diff, of supersplat-viewer commit `b2d8caa`
("omg4: degree-2 motion (accel) playback support"). No code was taken from them.

To test it, cumuli `scripts/bake_sogst.py` was run as a black box: its `--help`
text, run log and output archive were read, and its source was not opened. The
supersplat-viewer build and splat-transform CLI were run on the trained clips.

## Modules

| module | written from |
|---|---|
| `registry.py` | Solipsist spec (four plugin kinds, entry-point groups). Python `importlib.metadata` docs. |
| `config.py` | Solipsist spec (key names shared with the ComfyUI pack and the cumuli orchestrator). Default LRs from the 3DGS paper section 7.1 and gsplat `simple_trainer.py` defaults. Temporal LRs chosen here; see README. |
| `core.py`, `_shims/*` | The agreed cumuli-core API. `_shims/ply.py` from cumuli `docs/sogst-format.md` section 7; `_shims/cameras.py`, `_shims/metrics.py` from cumuli `scripts/eval_render.py` conventions; `_shims/spacetime.py` from `docs/sogst-format.md` section 1; `_shims/render.py` from the gsplat `rasterization` API. Only used when cumuli-core is missing. |
| `representations/base.py` | Solipsist spec. |
| `representations/spacetime.py` | Temporal model from `docs/sogst-format.md` section 1 (polynomial motion of degree 1 or 2 and temporal radial-basis opacity, after Li et al. CVPR 2024 section 3). Degree-2 motion from `docs/sogst-format.md` sections 1, 4.8 and 7 and the cumuli-core `slice_at(accel=)` and PLY API; see "Degree-2 motion" below. Initialisation per the Solipsist brief (k-NN scale as in the 3DGS paper section 5.1, using scipy `cKDTree`; `t_sigma = sqrt(duration / t_init_div)`). SH DC from RGB with the standard band-0 constant. |
| `strategies/base.py` | Solipsist spec. |
| `strategies/ops.py` | Clone, split (scale / 1.6) and prune from the 3DGS paper section 5.2; relocation and growth from Kheradmand et al. section 4 (equation 9 via gsplat's `compute_relocation`) and the noise of equation 8. Interface conventions (one Adam per parameter, moments edited alongside rows) follow gsplat's strategy ops, read but not copied. |
| `strategies/mcmc.py` | Kheradmand et al. 2024; parameter names and defaults follow gsplat `MCMCStrategy`. Scene-bounds recycling is new. |
| `strategies/default.py` | 3DGS paper section 5.2; thresholds and screen-space gradient normalisation follow gsplat `DefaultStrategy`. Dense (sync-free) accumulation and the cap are new. |
| `losses/photometric.py` | 3DGS paper section 7 (0.8 L1 + 0.2 D-SSIM). Foreground-crop SSIM is new. |
| `losses/ssim.py` | Wang et al. 2004 (11x11 Gaussian window, sigma 1.5, C1 = 0.01^2, C2 = 0.03^2). |
| `losses/masks.py` | Solipsist spec (`lambda_opa_mask` term). |
| `losses/regularizers.py` | Kheradmand et al. 2024 section 5 (opacity and scale regularisers). |
| `renderers/*` | gsplat `rasterization` API; settings match cumuli `scripts/eval_render.py`. |
| `data.py` | Solipsist spec (time-batched rendering, GPU-resident images). |
| `evaluate.py` | cumuli `scripts/eval_render.py` conventions, via cumuli-core metrics. |
| `progress.py` | comfyui-cumuli `cumuli_bridge/train.py` regular expressions. |
| `trainer.py`, `cli.py`, `__main__.py` | Solipsist spec; cumuli orchestrator contract (`python -m cumuli_trainer --config`). Exponential position-LR decay as in the 3DGS paper section 7.1. |
| `tests/*` | Written here. `tests/test_progress.py` copies the pack's four regular expressions. |

## Dependencies

torch (BSD-3), gsplat (Apache-2.0), numpy (BSD-3), scipy (BSD-3), pyyaml
(MIT), tqdm (MIT and MPL-2.0 dual licence; used unmodified as a library),
pillow (MIT-CMU), torchmetrics (Apache-2.0), lpips (BSD-2), cumuli-core
(Solipsist). No plyfile: PLY reading and writing is done with numpy.
