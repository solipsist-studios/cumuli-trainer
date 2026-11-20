# Attributions

cumuli-trainer was written from the publications, software and Solipsist
projects below. No third-party source code is included in this repository.

## Publications

| Publication | Used for |
|---|---|
| Kerbl, Kopanas, Leimkühler, Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023 | Adaptive density control (clone, split, prune, opacity reset), k-NN scale initialisation, the L1 and D-SSIM loss weights, the learning-rate schedule |
| Yang, Yang, Pan, Zhang, "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting", ICLR 2024 | The 4D Gaussian representation |
| Li, Chen, Li, Xu, "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis", CVPR 2024 | Polynomial motion and temporal radial-basis opacity |
| Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo", 2024 | MCMC densification, relocation, and the opacity and scale regularisers |
| Xie et al., "PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics", CVPR 2024 | The anisotropy regulariser |
| Wang, Bovik, Sheikh, Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004 | SSIM |

## Software

- **gsplat** 1.5.3 (Apache-2.0, https://github.com/nerfstudio-project/gsplat):
  the rasterizer and `compute_relocation`, called as an unmodified runtime
  dependency. Parameter names, default thresholds and learning rates for the
  densification strategies follow gsplat's `DefaultStrategy`, `MCMCStrategy`
  and `simple_trainer.py`.
- **OMG4** (the OMG4 trainer's dataset convention): the per-view `loss_weight`
  key of a transforms entry, and its scope, which is the colour loss terms.
- **PyTorch**, **torchmetrics** and **lpips**: the tensor library and the
  evaluation metrics, used as libraries.

## Solipsist Projects

- cumuli `docs/sogst-format.md` (the `.sogst` and 4D interchange PLY
  specification), `docs/pipeline.md` and `scripts/eval_render.py`: the file
  formats, the render settings and the evaluation conventions the trainer shares
  with the rest of the pipeline.
- cumuli-core: the shared library for cameras, datasets, the PLY format, the
  spacetime slice and metrics.
- comfyui-cumuli `cumuli_bridge/train.py`: the progress-line patterns in
  `progress.py` and `tests/test_progress.py`.

## Dependencies

| Package | License | Use |
|---|---|---|
| torch | BSD-3-Clause | tensors, autograd, optimisers |
| gsplat | Apache-2.0 | rasterization |
| numpy | BSD-3-Clause | arrays, PLY reading and writing |
| scipy | BSD-3-Clause | k-NN initialisation |
| pyyaml | MIT | config files |
| tqdm | MIT and MPL-2.0 | progress bars |
| pillow | MIT-CMU | image loading |
| torchmetrics | Apache-2.0 | PSNR and SSIM for evaluation |
| lpips | BSD-2-Clause | LPIPS for evaluation; fetches its pretrained AlexNet weights on first use |
| cumuli-core | PolyForm Noncommercial 1.0.0 | shared library |
