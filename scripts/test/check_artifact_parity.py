#!/usr/bin/env python
"""Artifact-parity gate: compare rendered pixels against a frozen baseline.

Covers what the eval-parity gate cannot. `check_eval_parity.py` exercises
`two_phase_model_forward` + `render_in_raw_space` on NeuMan -- it never touches the
canonical-view renderer, the turntable, the PLY writer or the motion/animation path.
Those are exactly the paths the visualiser, RGBA-helper and pipeline-layering cleanups
modify, and the existing smoke tests only assert that files *exist*, never that the
pixels in them are right.

    # freeze (once, on unmodified code) -- runs both smoke tests, then snapshots outputs
    python scripts/test/check_artifact_parity.py --freeze

    # after a change
    python scripts/test/check_artifact_parity.py

Comparison is per-pixel with a tolerance, not a checksum: the pipeline is not bitwise
deterministic (GPU rasterization varies slightly run to run). PNGs are compared as uint8 arrays;
videos and PLYs are compared on size only, since their encoders add their own noise --
the frames inside the turntable are covered by the canonical stills, which are PNGs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BASELINE = Path("expr/release_eval/artifact_baseline")
MANIFEST = BASELINE / "manifest.json"


def run_smoke_tests(out_dir: Path, skip_video: bool) -> None:
    env = {**os.environ, "PYTHONPATH": ""}
    tests = [["scripts/test/test_app_case.py", "--output_folder", str(out_dir)]]
    if not skip_video:
        tests.append(["scripts/test/test_app_video.py", "--output_folder", str(out_dir)])
    for test in tests:
        print(f"$ python {' '.join(test)}", flush=True)
        proc = subprocess.run([sys.executable, *test], env=env)
        if proc.returncode != 0:
            raise SystemExit(f"{test[0]} exited {proc.returncode}")


def collect(root: Path) -> list:
    """Every artifact under root, as paths relative to root."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".png", ".mp4", ".ply"})


def compare_png(a: Path, b: Path, tol_mean: float, tol_frac: float,
                level: int) -> tuple:
    """Return (ok, message) for two PNGs compared as uint8 arrays.

    Deliberately statistical, not max-based. The Gaussian rasterizer is not
    pixel-deterministic at silhouette edges -- back-to-back runs of identical code
    differ by up to ~100 uint8 levels on ~0.2% of pixels, while mean|d| stays around
    0.03. A max|d| threshold would therefore fail every time. Mean absolute difference
    and the fraction of substantially-changed pixels are both stable across runs, and
    a real change to the render path moves them by orders of magnitude.
    """
    ia, ib = np.asarray(Image.open(a)), np.asarray(Image.open(b))
    if ia.shape != ib.shape:
        return False, f"shape {ia.shape} -> {ib.shape}"
    diff = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
    mean_d = float(diff.mean())
    frac = float((diff > level).sum()) / diff.size
    msg = (f"mean|d|={mean_d:.4f}, frac(|d|>{level})={100 * frac:.3f}%, "
           f"max|d|={int(diff.max())}")
    if mean_d > tol_mean or frac > tol_frac:
        return False, (msg + f"  [limits: mean<={tol_mean}, "
                             f"frac<={100 * tol_frac:.1f}%]")
    return True, msg


def compare_size(baseline_bytes: int, cur: Path, rel_tol: float) -> tuple:
    """Videos and PLYs: size only, within a relative tolerance.

    Their bytes are never stored -- a canonical.ply is ~38 MB, and the encoders add
    their own run-to-run noise, so the pixels that matter are covered by the PNGs.
    """
    sb = cur.stat().st_size
    if baseline_bytes == 0 or sb == 0:
        return False, f"empty file ({baseline_bytes} -> {sb} bytes)"
    rel = abs(sb - baseline_bytes) / baseline_bytes
    if rel > rel_tol:
        return False, (f"size {baseline_bytes} -> {sb} bytes "
                       f"(rel {rel:.2%} > {rel_tol:.0%})")
    return True, f"size {baseline_bytes} -> {sb} bytes (rel {rel:.2%})"


def main(args: argparse.Namespace) -> int:
    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    run_smoke_tests(work, args.skip_video)
    produced = collect(work)
    if not produced:
        print(f"No artifacts produced under {work}", file=sys.stderr)
        return 1

    if args.freeze:
        if BASELINE.exists():
            shutil.rmtree(BASELINE)
        BASELINE.mkdir(parents=True)
        sizes = {}
        for rel in produced:
            if rel.endswith(".png"):
                dst = BASELINE / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(work / rel, dst)
            else:
                sizes[rel] = (work / rel).stat().st_size
        MANIFEST.write_text(json.dumps(
            {"artifacts": produced, "sizes": sizes}, indent=2) + "\n")
        n_png = sum(1 for r in produced if r.endswith(".png"))
        print(f"\nFroze {len(produced)} artifact(s) -> {BASELINE} "
              f"({n_png} PNG(s) stored, {len(sizes)} compared on size only)")
        return 0

    if not MANIFEST.is_file():
        print(f"No baseline at {BASELINE}. Run with --freeze first (on unmodified "
              f"code).", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST.read_text())
    expected, sizes = manifest["artifacts"], manifest.get("sizes", {})
    failures = []

    missing = [r for r in expected if r not in produced]
    extra = [r for r in produced if r not in expected]
    for rel in missing:
        failures.append(f"{rel}: MISSING from this run")
    for rel in extra:
        print(f"  note {rel}: new artifact, not in baseline")

    for rel in expected:
        if rel in missing:
            continue
        base, cur = BASELINE / rel, work / rel
        if rel.endswith(".png"):
            ok, msg = compare_png(base, cur, args.tol_mean, args.tol_frac,
                                  args.diff_level)
        else:
            ok, msg = compare_size(sizes.get(rel, 0), cur, args.tol_size)
        print(f"  {'ok  ' if ok else 'FAIL'} {rel}  {msg}")
        if not ok:
            failures.append(f"{rel}: {msg}")

    if failures:
        print(f"\nARTIFACT PARITY FAILED ({len(failures)} deviation(s)):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"\nOK -- {len(expected)} artifact(s) match the baseline.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rendered-artifact parity gate")
    p.add_argument("--freeze", action="store_true",
                   help="Snapshot this run's artifacts as the baseline")
    p.add_argument("--work_dir", default="expr/test_parity",
                   help="Scratch output dir for the smoke tests (wiped each run)")
    p.add_argument("--skip_video", action="store_true",
                   help="Skip test_app_video.py (the animation path)")
    p.add_argument("--tol_mean", type=float, default=0.5,
                   help="Max allowed mean per-pixel uint8 difference "
                        "(~15x the measured run-to-run floor of ~0.03)")
    p.add_argument("--tol_frac", type=float, default=0.02,
                   help="Max allowed fraction of pixels differing by more than "
                        "--diff_level (measured floor ~0.3%%)")
    p.add_argument("--diff_level", type=int, default=8,
                   help="uint8 delta above which a pixel counts as changed")
    p.add_argument("--tol_size", type=float, default=0.05,
                   help="Max allowed relative size difference for .mp4/.ply "
                        "(measured floor ~1%% for videos; PLYs are byte-stable)")
    raise SystemExit(main(p.parse_args()))
