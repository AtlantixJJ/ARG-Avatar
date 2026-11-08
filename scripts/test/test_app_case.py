#!/usr/bin/env python
"""Smoke test: reconstruct the bundled example subjects, headless.

Exercises the same pipeline the ``app`` package's Reconstruct tab does -- reconstruction, input-view
renders, canonical stills, turntable video, PLY export -- and fails loudly if any of them
produce nothing. This is the check to run after installing.

    python scripts/test/test_app_case.py
    python scripts/test/test_app_case.py --images "assets/example_images/2k2k/00093_0/*.png"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

from scripts.inference.pipeline import (
    DEFAULT_CONFIG_PATH,
    ARGAvatarPipeline,
    PipelineOptions,
    expand_image_args,
)
from scripts.inference.pipeline import write_gaussian_ply

ASSETS = Path("assets")


def default_cases() -> list:
    """One multi-view case and one single-view case from assets/example_images/, if present."""
    cases = []
    base = ASSETS / "example_images"
    if not base.is_dir():
        return cases
    subjects = sorted(p for p in base.glob("*/*") if p.is_dir())
    if not subjects:
        return cases

    def _images(subject_dir: Path) -> list:
        return sorted(str(p) for p in subject_dir.iterdir() if p.suffix.lower() in {".png", ".jpg"})

    multi_images = _images(subjects[0])
    if multi_images:
        cases.append((subjects[0].parent.name, multi_images))

    single_subject = subjects[1] if len(subjects) > 1 else subjects[0]
    single_images = _images(single_subject)
    if single_images:
        cases.append((f"{single_subject.parent.name}_single", single_images[:1]))
    return cases


def main(args: argparse.Namespace) -> int:
    if args.images:
        cases = [("cli", expand_image_args(args.images))]
    else:
        cases = default_cases()
    if not cases:
        print("No example images found under assets/example_images/; pass --images.", file=sys.stderr)
        return 1

    pipeline = ARGAvatarPipeline.build(
        PipelineOptions(config=args.config, checkpoint_path=args.checkpoint_path,
                        max_render_height=args.max_render_height)
    )
    print(f"Loaded {args.config} @ {pipeline.options.checkpoint_path}")

    failures = []
    for name, images in cases:
        out_dir = Path(args.output_folder) / name
        print(f"\n=== {name}: {len(images)} view(s) ===")
        try:
            avatar = pipeline.reconstruct(images, name=name)

            views = pipeline.render_views(avatar, str(args.output_folder))
            assert views, "no input-view renders written"
            print(f"  views     {len(views)} file(s)")

            canonical = pipeline.render_canonical(avatar, str(out_dir))
            assert len(canonical) == 8, f"expected 8 canonical stills, got {len(canonical)}"
            print(f"  canonical {len(canonical)} file(s)")

            orbit = pipeline.render_orbit_video(
                avatar, str(out_dir / "orbit.mp4"), n_frames=args.orbit_frames
            )
            assert Path(orbit).stat().st_size > 0, "orbit video is empty"
            print(f"  orbit     {orbit}")

            ply = write_gaussian_ply(pipeline.posed_gaussians(avatar), str(out_dir / "canonical.ply"))
            assert Path(ply).stat().st_size > 0, "ply is empty"
            print(f"  ply       {ply} ({Path(ply).stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001 - report every case, then fail
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  FAIL {type(exc).__name__}: {exc}", file=sys.stderr)

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"\nOK — {len(cases)} case(s) passed.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless reconstruction smoke test")
    parser.add_argument("--images", nargs="+", default=[],
                        help="Override the bundled examples with one subject's views")
    parser.add_argument("--output_folder", default="expr/test_app")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--max_render_height", type=int, default=1024)
    parser.add_argument("--orbit_frames", type=int, default=24,
                        help="Kept low so the smoke test stays fast")
    raise SystemExit(main(parser.parse_args()))
