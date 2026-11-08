#!/usr/bin/env python
"""Smoke test: reconstruct an avatar and drive it with a motion, headless.

Exercises the ``app`` package's Animate tab. Two modes:

    # play a bundled motion (fast, no kaolin needed)
    python scripts/test/test_app_video.py

    # fit a new motion from a video first, then play it (slow; needs kaolin for s3/s4)
    python scripts/test/test_app_video.py --video assets/example_videos/dnarendering_0012_09_cam52.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from scripts.inference.pipeline import (
    DEFAULT_CONFIG_PATH,
    ARGAvatarPipeline,
    PipelineOptions,
    expand_image_args,
)

ASSETS = Path("assets")


def default_images() -> list:
    directory = ASSETS / "example_images"
    if not directory.is_dir():
        return []
    for subject_dir in sorted(directory.glob("*/*")):
        if not subject_dir.is_dir():
            continue
        images = sorted(subject_dir.glob("*.png")) + sorted(subject_dir.glob("*.jpg"))
        if images:
            return [str(images[0])]
    return []


def default_motion() -> str:
    motions = sorted((ASSETS / "example_motion").glob("*.pth")) \
        if (ASSETS / "example_motion").is_dir() else []
    return str(motions[0]) if motions else ""


def fit_motion(video_path: str, max_frames: int, strategy: str) -> str:
    """Extract frames from ``video_path`` and fit an MHR motion over them."""
    motion_name = f"test-{Path(video_path).stem}-{time.strftime('%H%M%S')}"
    frames_dir = Path("expr/motion_pose") / motion_name / "images"
    frames_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
         *(["-frames:v", str(max_frames)] if max_frames > 0 else []),
         str(frames_dir / "%06d.png")],
        check=True,
    )
    n_frames = len(list(frames_dir.glob("*.png")))
    print(f"Extracted {n_frames} frame(s) -> {frames_dir}")
    if n_frames == 0:
        raise RuntimeError(f"no frames decoded from {video_path}")

    subprocess.run(
        [sys.executable, "scripts/inference/estimate_motion_mhr.py",
         "--video", motion_name, "--frames_dir", str(frames_dir),
         "--stage", "all", "--strategy", strategy],
        check=True, env={**os.environ, "PYTHONPATH": ""},
    )
    motion_path = Path("expr/motion_pose") / motion_name / f"sam3db_{strategy}.pth"
    if not motion_path.is_file():
        raise RuntimeError(f"motion fitting produced no {motion_path}")
    return str(motion_path)


def main(args: argparse.Namespace) -> int:
    images = expand_image_args(args.images) if args.images else default_images()
    if not images:
        print("No example images under assets/example_images/; pass --images.", file=sys.stderr)
        return 1

    if args.video:
        motion_path = fit_motion(args.video, args.max_frames, args.strategy)
    else:
        motion_path = args.motion or default_motion()
        if not motion_path:
            print("No motion under assets/example_motion/; pass --motion or --video.", file=sys.stderr)
            return 1
    print(f"Motion: {motion_path}")

    pipeline = ARGAvatarPipeline.build(
        PipelineOptions(config=args.config, checkpoint_path=args.checkpoint_path,
                        max_render_height=args.max_render_height)
    )
    avatar = pipeline.reconstruct(images, name="motion_test")
    outputs = pipeline.render_motion(
        avatar, motion_path, args.output_folder, max_frames=args.max_frames
    )
    for label, path in outputs.items():
        size = Path(path).stat().st_size
        if size == 0:
            print(f"FAIL: {label} video is empty ({path})", file=sys.stderr)
            return 1
        print(f"  {label:9s} {path} ({size / 1e6:.1f} MB)")

    print("\nOK — animation smoke test passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless animation smoke test")
    parser.add_argument("--images", nargs="+", default=[], help="Subject views to reconstruct")
    parser.add_argument("--motion", default="", help="A sam3db motion .pth to play")
    parser.add_argument("--video", default="",
                        help="Fit a new motion from this video instead of playing one")
    parser.add_argument("--strategy", default="s0", choices=["s0", "s1", "s2", "s3", "s4"],
                        help="Motion-fitting strategy, with --video")
    parser.add_argument("--max_frames", type=int, default=48,
                        help="Cap on frames, keeping the smoke test fast (0 = all)")
    parser.add_argument("--output_folder", default="expr/test_app/motion")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--max_render_height", type=int, default=1024)
    raise SystemExit(main(parser.parse_args()))
