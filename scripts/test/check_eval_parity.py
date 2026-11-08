#!/usr/bin/env python
"""Eval-parity gate: re-run the NeuMan benchmark and compare against a frozen baseline.

This is the gate that proves a refactor did not change what the model *computes*.
`verify_release.py --models` proves the checkpoint still loads; the smoke tests prove
files still appear. Only this proves the numbers are unchanged.

Two modes:

    # freeze the current numbers as the baseline (do this once, before any cleanup)
    python scripts/test/check_eval_parity.py --freeze

    # after a change: re-run and compare
    python scripts/test/check_eval_parity.py

The results cache: with `--skip_existing`, `eval_argavatar.py` re-reads cells already in
`eval_results.json`, so a parity check that shared the default results directory could pass
on old numbers. This script always evaluates into a fresh scratch directory instead.

Tolerances default to ~30x the measured run-to-run noise floor of this benchmark
(PSNR 3.5e-4 dB, SSIM 1.3e-6, LPIPS 2.9e-6 -- derived by diffing the two independent
runs recorded in expr/release_eval/RUN_NOTE.md and expr/results/). Anything that moves
the model graph or its numerics moves the metrics far more than that.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE = Path("expr/release_eval/eval_baseline.json")
METRICS = ("psnr_cross", "ssim_cross", "lpips_cross",
           "psnr_input", "ssim_input", "lpips_input")
COUNTS = ("n_cross", "n_input")


def run_eval(views: list, config: str, checkpoint: str, data_config: str,
             out_root: Path, n_samples: int) -> dict:
    """Evaluate into a fresh directory and return {n_views: {ds_key: metrics}}."""
    cmd = [
        sys.executable, "evaluation/argavatar/eval_argavatar.py",
        "--config", config,
        "--checkpoint_path", checkpoint,
        "--data_config", data_config,
        "--eval_views", *[str(v) for v in views],
        "--output_folder", str(out_root / "n"),
        "--n_vis", "0",
    ]
    if n_samples > 0:
        cmd += ["--n_samples", str(n_samples)]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env={"PYTHONPATH": "", **_clean_env()})
    if proc.returncode != 0:
        raise SystemExit(f"eval_argavatar.py exited {proc.returncode}")

    results = {}
    for v in views:
        # _results_dir_for_nviews appends _n<N> when more than one count is requested
        path = out_root / (f"n_n{v}" if len(views) > 1 else "n") / "eval_results.json"
        if not path.is_file():
            raise SystemExit(f"eval produced no results for n={v} (missing {path})")
        results[str(v)] = json.loads(path.read_text())["datasets"]
    return results


def _clean_env() -> dict:
    import os
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def compare(baseline: dict, current: dict, tol: dict) -> list:
    """Return a list of human-readable failures; empty means parity holds."""
    failures = []
    for n in sorted(baseline, key=int):
        if n not in current:
            failures.append(f"n={n}: missing from this run")
            continue
        for ds in baseline[n]:
            if ds not in current[n]:
                failures.append(f"n={n} {ds}: dataset missing from this run")
                continue
            base, cur = baseline[n][ds], current[n][ds]

            # Sample counts must match exactly -- a changed count means the run
            # evaluated different data, which makes the metric comparison meaningless.
            for key in COUNTS:
                if base.get(key) != cur.get(key):
                    failures.append(
                        f"n={n} {ds} {key}: {base.get(key)} -> {cur.get(key)} "
                        f"(different data evaluated; metrics not comparable)")

            for key in METRICS:
                if key not in base:
                    continue
                delta = abs(cur[key] - base[key])
                limit = tol["psnr"] if key.startswith("psnr") else \
                        tol["ssim"] if key.startswith("ssim") else tol["lpips"]
                mark = "ok  " if delta <= limit else "FAIL"
                print(f"  {mark} n={n:>2} {ds:<10} {key:<12} "
                      f"{base[key]:.6f} -> {cur[key]:.6f}  (d={delta:.2e}, tol={limit:.0e})")
                if delta > limit:
                    failures.append(
                        f"n={n} {ds} {key}: {base[key]:.6f} -> {cur[key]:.6f} "
                        f"(delta {delta:.2e} > tol {limit:.0e})")
    return failures


def main(args: argparse.Namespace) -> int:
    views = args.eval_views
    if args.freeze:
        out_root = Path(tempfile.mkdtemp(prefix="eval_parity_freeze_"))
    else:
        if not BASELINE.is_file():
            print(f"No baseline at {BASELINE}. Run with --freeze first "
                  f"(on unmodified code).", file=sys.stderr)
            return 1
        out_root = Path(tempfile.mkdtemp(prefix="eval_parity_"))

    try:
        current = run_eval(views, args.config, args.checkpoint_path,
                           args.data_config, out_root, args.n_samples)
    finally:
        if not args.keep:
            shutil.rmtree(out_root, ignore_errors=True)

    if args.freeze:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"\nFroze baseline for view counts {views} -> {BASELINE}")
        return 0

    baseline = json.loads(BASELINE.read_text())
    tol = {"psnr": args.tol_psnr, "ssim": args.tol_ssim, "lpips": args.tol_lpips}
    print(f"\nComparing against {BASELINE}:")
    failures = compare(baseline, current, tol)

    if failures:
        print(f"\nEVAL PARITY FAILED ({len(failures)} deviation(s)):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("\nOK -- eval parity holds.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="NeuMan eval-parity gate")
    p.add_argument("--freeze", action="store_true",
                   help="Write the current numbers to the baseline instead of comparing")
    p.add_argument("--config", default="configs/ARGAvatar-Final.yaml")
    p.add_argument("--checkpoint_path", default="checkpoints/ARGAvatar-Final.pth")
    p.add_argument("--data_config", default="configs/_shared/data_val_list.yaml")
    p.add_argument("--eval_views", type=int, nargs="+", default=[1, 6],
                   help="View counts to check. Default [1 6] is the fast gate "
                        "(~2 min); use 1 2 3 4 6 for the full release sweep.")
    p.add_argument("--n_samples", type=int, default=-1)
    p.add_argument("--tol_psnr", type=float, default=1e-2, help="dB")
    p.add_argument("--tol_ssim", type=float, default=1e-4)
    p.add_argument("--tol_lpips", type=float, default=1e-4)
    p.add_argument("--keep", action="store_true", help="Keep the scratch results dir")
    raise SystemExit(main(p.parse_args()))
