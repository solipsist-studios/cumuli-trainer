# Changelog

## 0.2.0

Requires cumuli-core 0.2.0.

- TRiGS representation: `representation: trigs` trains rigid-body motion about
  an anchor (arXiv 2604.00538), with the `trigs_motion` and `trigs_rigid`
  losses. It writes `trigs_ckpt.pt`, in the layout the Open4DGS-Core packager
  reads, instead of `splat_4d.ply`. See the README section TRiGS representation.
- Degree-2 motion: `representation: {name: spacetime, motion_degree: 2}` adds
  a trained `accelerations` tensor and writes `ax, ay, az` to the PLY. The
  default stays degree 1. A degree-1 checkpoint can continue at degree 2.
- Per-view loss weights: a dataset's `loss_weight` scales that view's `l1` and
  `ssim` terms. Datasets without it train as before.
- Sub-frame timing: views render at their camera's capture time
  (`time_offset`); rolling-shutter views render each Gaussian at the time its
  raw sensor row was read (`readout_s`); an instant's `visit_weight` sets how
  often it is sampled; and the `time_offsets` config learns per-camera
  corrections. See the README section Sub-Frame Timing.
- `data_device: auto` keeps 4 GiB of GPU headroom and refuses host-resident
  images that would exhaust RAM.
- Creates the cuSOLVER handle at start and frees cached GPU memory before each
  evaluation, which fixed a failure at the held-out evaluation of a long window.

## 0.1.0

First release.
