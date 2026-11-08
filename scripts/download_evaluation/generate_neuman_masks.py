#!/usr/bin/env python3
"""Build ``masks/`` for a NeuMan subject by inverting ``segmentations/``.

Apple's ``dataset.zip`` ships each scene's foreground mask as ``segmentations/`` (human
~255, background ~0). ``xlib/dataset/sam3d_image_dataset.py`` wants the opposite
convention, so this writes ``255 - value`` per channel into ``<scene>/masks/`` with
matching filenames -- exactly what ``prepare_neuman.py --detections`` would otherwise
fetch as a precomputed pack.

This reproduces the *naive* inversion only. The masks actually shipped in
``argavatar-neuman-detections.tar.gz`` were further refined by a GroundingDINO+SAM2 pass
that is out of scope for this inference-only release (new heavy dependencies, no
checkpoints bundled here) -- the output of this script is a close but not bit-identical
stand-in (~94% IoU against the shipped masks on the one scene checked).

``segmentations/`` lives under the raw extracted archive
(``data/neuman_raw/dataset/<scene>/segmentations``, from ``prepare_neuman.py --unzip``'s
``--raw_dir``), not the prepared root -- only ``images/`` gets copied into the root by
that step. This script reads from the raw dir and writes into the prepared root.

    python scripts/download_evaluation/generate_neuman_masks.py

or for one scene while iterating:

    python scripts/download_evaluation/generate_neuman_masks.py --scenes bike
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, ".")

from scripts.download_evaluation.neuman_manifest import SUBJECTS

DEFAULT_RAW = Path("data/neuman_raw/dataset")
DEFAULT_ROOT = Path("data/AnimatableHuman/Rendering/NeuMan")


def _as_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _invert(arr: np.ndarray) -> np.ndarray:
    """Invert a uint8 (H,W) or (H,W,C) array. For RGBA, only RGB is inverted."""
    u = _as_uint8(arr)
    if u.ndim == 2:
        return 255 - u
    if u.shape[2] == 4:
        return np.concatenate([255 - u[:, :, :3], u[:, :, 3:4]], axis=-1)
    return 255 - u


def _load_segmentation(path: Path) -> Image.Image:
    im = Image.open(path)
    if im.mode == "P":
        im = im.convert("L")
    return im


def invert_scene(seg_dir: Path, out_dir: Path, force: bool) -> int:
    sources = sorted(seg_dir.glob("*.png"))
    if not sources:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for src in sources:
        dst = out_dir / src.name
        if dst.is_file() and not force:
            continue
        im = _load_segmentation(src)
        arr = _invert(np.asarray(im))
        mode = "RGBA" if arr.ndim == 3 and arr.shape[2] == 4 else ("RGB" if arr.ndim == 3 else "L")
        Image.fromarray(arr, mode=mode).save(dst)
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW,
                        help="Extracted Apple archive, one folder per scene")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Prepared NeuMan root to write masks/ into")
    parser.add_argument("--scenes", nargs="*", default=None, metavar="NAME",
                        help="Only these scenes (default: all of SUBJECTS)")
    parser.add_argument("--force", action="store_true", help="Rewrite masks that already exist")
    args = parser.parse_args()

    scenes = args.scenes if args.scenes else list(SUBJECTS)
    total = 0
    for scene in scenes:
        seg_dir = args.raw_dir / scene / "segmentations"
        if not seg_dir.is_dir():
            print(f"[skip] {seg_dir} not found", file=sys.stderr)
            continue
        out_dir = args.root / scene / "masks"
        n = invert_scene(seg_dir, out_dir, args.force)
        print(f"{scene}: wrote {n} -> {out_dir}")
        total += n
    print(f"total: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
