# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ARGAvatar reconstructs an animatable 3D Gaussian human avatar from one or a few
images in a single feed-forward pass. Gaussians are predicted in the UV space
of the MHR body model, so the reconstruction is riggable and can be re-posed
and rendered from novel views without per-subject optimisation.

This is an **evaluation- and inference-only public release**. Training and data
preprocessing code are intentionally absent; do not add a training entrypoint.
Exactly one model ships: `configs/ARGAvatar-Final.yaml` +
`checkpoints/ARGAvatar-Final.pth` (other `.pth` files that may be present
locally in `checkpoints/` are not part of the release).

**Pipeline:** SAM-3D-Body estimates MHR body pose and camera per input view →
a DINOv3 ConvNeXt backbone extracts per-view features → a multi-view avatar
transformer with FAC-RoPE token-to-image attention lifts them onto UV-space
avatar tokens → a ResNet-unpatchify decoder and a token-cascade refinement
stage emit per-Gaussian parameters → 3D Gaussian splatting through
`diff-gaussian-rasterization-da`.

## Running commands

Run everything from the repository root **with `PYTHONPATH` empty** so `xlib`
resolves to this checkout:

```bash
PYTHONPATH= python <script> ...
```

The installer creates a Conda environment (`bash scripts/install_argavatar_env.sh`,
then `conda activate argavatar`). Every dependency lives in the single
`requirements.txt`: PyTorch, PyPI packages (MHR included), Kaolin from NVIDIA's
wheel index, the vendored CUDA extensions (`simple-knn`,
`diff-gaussian-rasterization-da`, sam2) and pinned git sources (`nvdiffrast`,
Detectron2, MoGe). Keep it one file. The extensions import torch while pip
prepares their metadata, so the installer first installs the torch/build-tool
pins it greps from that file, then the whole file with `--no-build-isolation`;
the README documents the same two pip commands for an existing environment.
Weights come from `bash scripts/download_weights.sh` (gated HF repos; needs `HF_TOKEN`).

### Verification gates — pick by what you touched

| Changed | Run |
|---|---|
| Anything | `PYTHONPATH= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_release.py --imports --configs --checkpoints --mirrors --models --entrypoints` |
| `app/`, pipeline, streaming | `PYTHONPATH= python -m unittest discover -s tests -v` |
| Inference / rendering / animation paths | `PYTHONPATH= python scripts/test/test_app_case.py` and `scripts/test/test_app_video.py` (add `--video assets/example_videos/dnarendering_0012_09_cam52.mp4` to cover motion fitting) |
| Model graph, numerics, renderer, PLY writer | `scripts/test/check_eval_parity.py` and `scripts/test/check_artifact_parity.py` — run with `--freeze` on the unmodified code first, then without it after the change. The smoke tests only assert that files exist; only the parity gates prove numbers/pixels are unchanged |
| Browser UI | `scripts/test/test_app_mobile.py` / `scripts/test/test_app_streaming.py --url http://127.0.0.1:7860` — need Playwright and a running app (`python -m app --no_load` suffices for the mobile layout test) |

`verify_release.py` exits non-zero and names the offender if an entrypoint
fails to import, a `class_name:` in `configs/**/*.yaml` fails to resolve, a
checkpoint fails to load strictly, an `expr/<model>/PT_*.pth` mirror is
missing, or an entrypoint is undocumented (see below).

### Demo / inference / evaluation

```bash
python -m app                                                          # Gradio UI on :7860
python scripts/inference/infer_argavatar.py --images assets/example_images/dnarendering/542_00004   # one subject
python scripts/inference/infer_argavatar.py --input_dir path/to/subjects --save_avatar              # batch
python scripts/inference/infer_argavatar.py --images <imgs> --save_ply --motion <M.pth> --ply_frame 14   # export
python scripts/inference/estimate_motion_mhr.py --video my_clip --video_path <mp4>          # fit a driving motion

# needs the NeuMan benchmark: python scripts/download_evaluation/prepare_neuman.py --all
python evaluation/argavatar/eval_argavatar.py \
  --config configs/ARGAvatar-Final.yaml \
  --checkpoint_path checkpoints/ARGAvatar-Final.pth \
  --data_config configs/_shared/data_val_list.yaml --eval_views 1 2 3 4 6
```

Each `--eval_views` count writes its own `expr/results/<model>/<ckpt>_n<N>/`.
Every cell is recomputed by default — never pass `--skip_existing` when numbers
must be comparable across a run.

## Architecture

### Config-driven model selection

Inference and evaluation are driven by a YAML config's `model.class_name`,
resolved dynamically — there is no hardcoded model switch. The shipped config
resolves to `xlib.model.arg_net_uv_token_cascade.ARGNetUVTokenCascade`, a
subclass of `xlib.model.arg_net_uv.ARGNetUV` (depth-2 transformer, `fac-rope`
attention, pose-dependent deformation, token-cascade refinement).

Checkpoints resolve two ways: a config's `model.load_checkpoint` may point
directly at `checkpoints/<name>.pth` (what `ARGAvatar-Final.yaml` does), or a
bare model name resolves through `evaluation/argavatar/manager.py::resolve_jobs`,
which globs `expr/<name>/PT_*.pth` — a hardlink `download_weights.sh` creates
alongside the canonical checkpoint. Prefer passing `--checkpoint_path`
explicitly over relying on the fallback.

### One pipeline object shared by every entrypoint

`scripts/inference/infer_argavatar.py` holds the low-level
reconstruction/animation/rendering helpers; single- and multi-view both go
through the same `[B=1, V, ...]` view-stacked batch (single-view is `V=1`).
`scripts/inference/pipeline.py` wraps it in `ARGAvatarPipeline`, one loaded
model reused across calls, which `app/`, the `infer_argavatar.py` CLI (`main()`)
and the smoke tests call. There is exactly **one** inference CLI,
`infer_argavatar.py`: it covers single subject (`--images`), batch (`--input_dir`),
Gaussian export (`--save_ply`, `--ply_frame`, `--ply_format`) and animation
(`--motion`). Don't add another inference script — add a flag. The PLY writers
(`write_gaussian_ply`, `export_gaussians`) live in `pipeline.py`; they export
`index_gs(gs, 0, 0)`, the same slice the renderer draws (multi-view motion frames
carry one scale/rotation copy per input view). **Never duplicate reconstruction/rendering logic in a new
entrypoint — route it through `pipeline.py` (or the `infer_argavatar` helpers
it wraps) so the demo and batch paths cannot drift apart.**

### Directory map (what requires cross-file reading)

- `app/` — Gradio demo. `__main__.py` (CLI flags, launch), `ui.py` (layout
  only), `callbacks.py` (orchestration over `ARGAvatarPipeline`), `state.py`,
  `streaming.py` (chunked H.264 motion streaming), `files.py` (per-browser
  ownership of uploads/outputs via an HttpOnly cookie — bundled `assets/` stay
  public), `viewer.py` (Spark three.js splat viewer inside an
  `<iframe srcdoc>`).
- `xlib/model/` — network code: `arg_net_uv.py` / `arg_net_uv_token_cascade.py`,
  `gs_layer.py` (Gaussian head), `hand_refinement.py`, `rope_3d.py` (FAC-RoPE),
  `sam_3d_body/` (vendored, own license — see below).
- `xlib/render/` — Gaussian splatting rasterization, cameras, visualisation.
- `xlib/dataset/` — validation datasets (`sam3d_image_dataset.py` is what
  `configs/_shared/data_val_list.yaml` instantiates via
  `xlib.dataset.simple.ConcatDataset`).
- `xlib/geometry/`, `xlib/train/` — UV/positional encodings; metrics and
  model-setup helpers (`train` means "constructing a trained model", not a
  training loop).
- `xlib/fitting/` — temporal MHR pose fitting for `estimate_motion_mhr.py`
  (strategies `s3`/`s4` import `kaolin`).
- `xlib/runtime_assets.py` + `scripts/download_runtime_assets.py` — pinned
  local DINOv3 config/source in `pretrained_models/`; model construction must
  not hit Torch Hub or the network.
- `evaluation/argavatar/` — `eval_argavatar.py` (entry), `manager.py` (job
  resolution, GPU worker pool, results writing), `worker.py` (per-GPU model
  host + metric accumulation). `evaluation/common/` — shared results I/O,
  caching, timing, val-list parsing.
- `configs/_shared/` — SAM-3D-Body defaults, the validation dataset config,
  `release_eval_jobs.yaml` (the one-model sweep).
- `scripts/download_evaluation/` — NeuMan benchmark assembly from Apple's
  frames + the bundled detection pack in `assets/benchmark/` (SHA256-checked).
- `scripts/release/` — maintainer-only builders for `assets/`, the
  detection pack and the README gallery (`build_gallery.py` → `assets/gallery/`,
  from curated subject folders under `expr/rendering_videos/` plus the
  `woman_dance_2_loopNN` motions, so every GIF loops seamlessly).
- `thirdparty/` — vendored CUDA extensions and models, compiled by the install
  script.
- `checkpoints/`, `pretrained_models/`, `data/`, `expr/` — gitignored, except
  `expr/val_lists/`, which is force-tracked (the pinned NeuMan val list).
  `tests/` and the local release provenance in `expr/release_eval/` are untracked.

## Release invariants (easy to break by accident)

- **Every script must be named in the docs.**
  `verify_release.py --entrypoints` fails if a path in its
  `DOCUMENTED_ENTRYPOINTS` is missing from `README.md` / `evaluation/README.md`
  (`app/__main__.py` may appear as `python -m app`), if a path in
  `MAINTAINER_ENTRYPOINTS` (the release check and `scripts/test/`) is missing
  from this file, *or* if a new `scripts/**/*.py|sh` is in neither list
  (`scripts/release/` is exempt). Adding a script means adding it to a list and
  to the matching doc. The user README carries no developer/test section.
- **README anchors are linked from other files:** `evaluation/README.md` →
  `README.md#21-benchmark-data`. The README uses numbered sections (modelled on
  the video-diffusion-avatars README), so renumbering a section changes its
  anchor — update these links and the README's own cross-references.
- **README figures** in `assets/` (`teaser.jpg`, `pipeline.jpg`, `bubble.png`,
  `latency.png`) are crops of Figures 1, 4, 2 and 9 of the paper; the runtime
  table in README §3 is the paper's Table 4. `assets/gallery/*.gif` come from
  `scripts/release/build_gallery.py`.
- **Licensing:** original code and the ARGAvatar weights are CC BY-NC-SA 4.0
  (`LICENSE`: a header listing third-party exceptions, then the verbatim CC
  legal text — don't edit the CC part). Vendored code, downloaded weights and
  datasets keep upstream terms, listed in both the `LICENSE` header and
  `THIRD_PARTY_NOTICES.md` — update both when vendoring anything. Don't add
  dependencies whose licenses forbid non-commercial ShareAlike redistribution.
  The six `assets/example_images/<dataset>/` clips are real dataset frames
  shipped under their datasets' terms as a one-off exception, and the renders of
  dataset subjects in `assets/gallery/` carry the same terms. Do not add other raw benchmark/dataset imagery to the repo (the
  gallery's input views stay outside it).
  NeuMan imagery is never redistributed — only our masks/poses pack.
- **Published numbers** in README §2.3 come from
  `expr/release_eval/RUN_NOTE.md`; don't edit them without a new recorded run.

## Conventions

- Motion files for driving animation are `sam3db`-format dicts
  (`{"images/<frame>.png": {pose + camera}}`), produced by
  `estimate_motion_mhr.py` and consumed by `infer_argavatar.py --motion` (alias
  `--driving_mhr_pose`) and the app's Animate tab.
- `eval_results.json` reports `*_cross` (novel-view, the headline metric)
  alongside `*_input` (reconstruction of input views) — don't conflate the two
  when reading or aggregating results.
- The README is written for end users (quick start first, detail in `<details>`
  blocks); developer and test workflows live in this file, not the README. Keep
  internal history and run names out of it.
