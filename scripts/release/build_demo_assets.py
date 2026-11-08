#!/usr/bin/env python3
"""Maintainer tool: build `assets/` from the source demo data.

Users do not run this -- they get `assets/` from the repository. This is how those files are produced, kept in the repo
so the demo data is reproducible rather than a pile of files someone once copied in.

Example images come one clip per training/eval dataset, so the shipped examples span every
dataset the model is evaluated on: mostly the curated `input_images` of that dataset's own
val list (`--val_lists_source`), 4D-Dress and DNARendering from a visualisation run
(`--visualize_source`). This is a deliberate exception to the usual "don't redistribute
benchmark imagery" policy (see README's Benchmark data section) -- each dataset keeps its
own upstream license, noted in README's License section, and this is a maintainer call,
not something to repeat without re-checking that section.

The driving clip comes from a dance clip under `--source` -- AI-generated (Gemini), not
third-party dataset material. Bundled motions are every valid and looping segment of the
driving motions in `--visualize_source`, sliced from the fitted MHR poses in
`--motion_source` and capped at MOTION_FRAMES frames.

    python scripts/release/build_demo_assets.py
    python scripts/release/build_demo_assets.py --source /path/to/CustomAnimation --force
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, ".")

DEFAULT_SOURCE = Path("/data1/datasets/AnimatableHuman/CustomAnimation")
DEFAULT_MOTION_SOURCE = Path("/home/jianjinx/data2/SAM3DGS/expr/motion_pose")
DEFAULT_VAL_LISTS_SOURCE = Path("/home/jianjinx/data2/SAM3DGS/expr/val_lists")
DEFAULT_VISUALIZE_SOURCE = Path("/home/jianjinx/data2/SAM3DGS/expr/infer_sam3dgs_visualize_all")
ASSETS = Path("assets")

# One subject per training/eval dataset: (val list filename under --val_lists_source,
# the dataset's image root, the item key to pull `input_images` from). Item keys are
# pinned rather than "first in the val list" so a re-run always ships the same subject.
# Data roots are absolute mount paths rather than each val list's own `data_root` field,
# since some of those point at mounts that are not always attached; these paths are
# where the images actually live.
VAL_LIST_SUBJECTS = {
    "2k2k": (
        "2K2K_val_list.yaml", Path("/mnt/turing_ssd/dataset/Rendering/2K2K"), "00093/0",
    ),
    "human4dit": (
        "Human4DiT_val_list.yaml", Path("/mnt/turing_ssd/dataset/Rendering/Human4DiT"), "000_00003/0",
    ),
    "neuman": (
        "Neuman_val_list.yaml", Path("/data1/datasets/AnimatableHuman/Rendering/NeuMan"), "bike/0",
    ),
    "thuman21": (
        "THuman21_val_list.yaml", Path("/mnt/turing_ssd/dataset/Rendering/THuman21"), "0002/0",
    ),
}

# Subjects taken from a visualisation run (--visualize_source) instead of a val list:
# slug -> (run subject name, clip folder). Its
# `human_input/<sample>__input_images__<NNNN>.png` crops ship as `<NNNN>.png`.
VISUALIZE_SUBJECTS = {
    "4ddress": ("4D-Dress_00175_1", "00175_1"),
    "dnarendering": ("DNARendering-Part1234_542_00004", "542_00004"),
}

# Longest side of a shipped example image. The model works from a 512-px crop, so full
# resolution buys nothing and costs repository size.
MAX_EDGE = 1024

DEMO_VIDEO = "woman_dance_1"
VIDEO_SECONDS = 5
MOTION_FRAMES = 300


def _save_resized(src: Path, out_path: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        im.save(out_path, optimize=True)
    print(f"[make] {out_path} {im.size} {out_path.stat().st_size / 1e6:.2f} MB")


def build_images(val_lists_source: Path, visualize_source: Path, force: bool) -> None:
    """One folder per dataset clip under `assets/example_images/<slug>/<clip>/`, holding
    six views of that clip (for val-list subjects, the curated `input_images` under their
    original filenames; the clip folder is the item key with `/` -> `_`) -- the multi- and
    single-view demos both read from here, single-view by taking just the first file in a
    clip's folder."""
    base_dir = ASSETS / "example_images"
    base_dir.mkdir(parents=True, exist_ok=True)

    for slug, (val_list_name, data_root, item_key) in VAL_LIST_SUBJECTS.items():
        val_list_path = val_lists_source / val_list_name
        if not val_list_path.is_file():
            print(f"[skip] {val_list_path} not found")
            continue
        items = yaml.safe_load(val_list_path.read_text())["items"]
        if item_key not in items:
            print(f"[skip] {item_key} not in {val_list_path}")
            continue
        rel_images = items[item_key]["input_images"]

        subject_dir = base_dir / slug / item_key.replace("/", "_")
        subject_dir.mkdir(parents=True, exist_ok=True)
        for rel in rel_images:
            out_path = subject_dir / Path(rel).name
            if out_path.exists() and not force:
                print(f"[skip] {out_path} exists")
                continue
            src = data_root / rel
            if not src.is_file():
                print(f"[skip] {src} not found")
                continue
            _save_resized(src, out_path)

    for slug, (run_name, clip) in VISUALIZE_SUBJECTS.items():
        src_dir = visualize_source / run_name / run_name / "human_input"
        sources = sorted(src_dir.glob("*__input_images__*.png"))
        if not sources:
            print(f"[skip] no input images under {src_dir}")
            continue
        subject_dir = base_dir / slug / clip
        subject_dir.mkdir(parents=True, exist_ok=True)
        for src in sources:
            out_path = subject_dir / src.name.split("__input_images__")[-1]
            if out_path.exists() and not force:
                print(f"[skip] {out_path} exists")
                continue
            _save_resized(src, out_path)


def build_video(source: Path, force: bool) -> None:
    video_dir = ASSETS / "example_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    src = source / "motion" / f"{DEMO_VIDEO}.mp4"
    out_path = video_dir / f"{DEMO_VIDEO}.mp4"
    if not src.is_file():
        print(f"[skip] {src} not found")
        return
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists")
        return
    # Re-encoded rather than stream-copied so the clip starts on a keyframe.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-t", str(VIDEO_SECONDS), "-c:v", "libx264", "-crf", "26",
         "-pix_fmt", "yuv420p", "-an", str(out_path)],
        check=True,
    )
    print(f"[make] {out_path} {out_path.stat().st_size / 1e6:.2f} MB")


# Best-to-worst: s4/s3 are the fully temporally-fitted variants, s2/s1/s0 progressively
# less filtered, gated/tracked the pre-strategy stages -- used when a motion has nothing
# further along the pipeline.
STRATEGY_PREFERENCE = ("s4", "s3", "s2", "s1", "s0", "gated", "tracked")

# A visualisation run's `valid_segments/` and `looping_segments/` clips are named
# `<motion>_original_seg<NN>_<start>-<end>.mp4` / `..._loop<NN>_<start>-<end>_p<period>s.mp4`,
# with <start>/<end> the 1-based, inclusive frame keys of the motion's sam3db poses.
_SEGMENT_RE = re.compile(
    r"^(?P<motion>.+)_original_(?P<kind>seg|loop)(?P<num>\d+)_(?P<start>\d+)-(?P<end>\d+)"
)


def _motion_segments(visualize_source: Path) -> dict:
    """`{example name: (motion, kind, start, end, run clip name)}` for every distinct
    segment across all subjects of the visualisation run (every subject driven by a motion
    gets the same segments, so each one is kept once)."""
    segments = {}
    for clip in sorted(visualize_source.glob("*/*_segments/*.mp4")):
        match = _SEGMENT_RE.match(clip.stem)
        if match is None:
            continue
        name = f"{match['motion']}_{match['kind']}{match['num']}"
        segments.setdefault(name, (
            match["motion"], match["kind"], int(match["start"]), int(match["end"]), clip.stem,
        ))
    return segments


def _segment_preview(motion_source: Path, motion: str, kind: str, clip_stem: str) -> Path:
    """The fitted-mesh render of a segment, as the motion pipeline wrote it. Previews are
    only ever cut from these: a motion's source clip can be third-party footage (e.g.
    woman_dance_2), so it is never shipped as a preview."""
    sub = "valid_segments" if kind == "seg" else "looping_segments"
    suffix = clip_stem.split("_original_", 1)[1]
    return motion_source / motion / sub / f"mesh_{suffix}.mp4"


def _detach_storage(value):
    """Copy every tensor in a (nested) pose entry, so `torch.save` writes just this frame
    rather than the whole motion's storage the tensor is a view into."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _detach_storage(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_storage(item) for item in value)
    return value


def build_motion(motion_source: Path, visualize_source: Path, force: bool) -> None:
    """Ship every valid and looping segment of the visualisation run's driving motions as
    `<motion>_seg<NN>` / `<motion>_loop<NN>`: the segment's sam3db poses (`.pth`) plus,
    where a mesh render exists, a preview (`.mp4`), both capped at MOTION_FRAMES so the set
    can ship in-repo."""
    motion_dir = ASSETS / "example_motion"
    motion_dir.mkdir(parents=True, exist_ok=True)

    segments = _motion_segments(visualize_source)
    if not segments:
        print(f"[skip] no *_segments/*.mp4 under {visualize_source}")
        return

    sam3db_cache = {}
    for name, (motion, kind, start, end, clip_stem) in segments.items():
        out_path = motion_dir / f"{name}.pth"
        preview_path = motion_dir / f"{name}.mp4"
        if out_path.exists() and not force:
            print(f"[skip] {out_path} exists")
            continue

        if motion not in sam3db_cache:
            for strategy in STRATEGY_PREFERENCE:
                src = motion_source / motion / f"sam3db_{strategy}.pth"
                if src.is_file():
                    break
            else:
                print(f"[skip] no sam3db_*.pth under {motion_source / motion}")
                continue
            sam3db_cache.clear()  # segments are grouped by motion; hold one at a time
            sam3db_cache[motion] = (src, torch.load(src, map_location="cpu", weights_only=False))
        src, sam3db = sam3db_cache[motion]

        last = min(end, start + MOTION_FRAMES - 1)
        keys = [f"images/{idx:06d}.png" for idx in range(start, last + 1)]
        missing = [key for key in keys if key not in sam3db]
        if missing:
            print(f"[skip] {name}: {len(missing)} frames missing from {src}, e.g. {missing[0]}")
            continue
        torch.save({key: _detach_storage(sam3db[key]) for key in keys}, out_path)
        print(f"[make] {out_path} ({len(keys)}/{end - start + 1} frames from {src.name}) "
              f"{out_path.stat().st_size / 1e6:.2f} MB")

        video = _segment_preview(motion_source, motion, kind, clip_stem)
        if not video.is_file():
            print(f"[skip] {name}: no mesh render {video}, shipping without a preview")
            continue
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-vf", "scale='min(640,iw)':-2",
             "-frames:v", str(len(keys)), "-c:v", "libx264", "-crf", "28",
             "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(preview_path)],
            check=True,
        )
        print(f"[make] {preview_path} {preview_path.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="CustomAnimation root holding motion/ (the driving clip)")
    parser.add_argument("--motion_source", type=Path, default=DEFAULT_MOTION_SOURCE,
                        help="Directory of fitted motions (expr/motion_pose)")
    parser.add_argument("--val_lists_source", type=Path, default=DEFAULT_VAL_LISTS_SOURCE,
                        help="Directory of per-dataset val list yamls (expr/val_lists)")
    parser.add_argument("--visualize_source", type=Path, default=DEFAULT_VISUALIZE_SOURCE,
                        help="Visualisation run holding VISUALIZE_SUBJECTS and the "
                             "valid/looping motion segments (expr/infer_sam3dgs_visualize_all)")
    parser.add_argument("--force", action="store_true", help="Rebuild files that already exist")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to build the example video")

    build_images(args.val_lists_source, args.visualize_source, args.force)
    build_video(args.source, args.force)
    build_motion(args.motion_source, args.visualize_source, args.force)

    total = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file())
    print(f"\nassets/ is now {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
