"""Shared caching helpers: per-subject .npy resume and per-dataset JSON cache."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Per-subject .npy resume
# ---------------------------------------------------------------------------

def save_sample_result(npy_path: str, sample_key: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(npy_path)), exist_ok=True)
    data: Dict[str, Any] = {}
    if os.path.isfile(npy_path):
        try:
            data = np.load(npy_path, allow_pickle=True).item()
        except Exception:
            pass
    data[sample_key] = result
    np.save(npy_path, data)


# ---------------------------------------------------------------------------
# Per-dataset JSON cache (ARGAvatar eval path)
# ---------------------------------------------------------------------------

_DATASET_RESULT_KEYS = (
    "psnr_input", "psnr_cross", "lpips_input", "lpips_cross",
    "ssim_input", "ssim_cross", "n_input", "n_cross",
)


def save_results_json(
    json_path: str,
    all_results: Dict[str, Any],
    val_ds_keys: List[str],
) -> Dict[str, Any]:
    """Write all_results for val_ds_keys to json_path, adding an overall block."""
    from xlib.train.metrics import compute_overall_eval_metrics
    payload = {
        "datasets": {ds_key: all_results[ds_key] for ds_key in val_ds_keys if ds_key in all_results},
        "overall": compute_overall_eval_metrics({k: all_results[k] for k in val_ds_keys if k in all_results}),
    }
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def load_cached_dataset_results(json_path: str) -> Dict[str, Any]:
    """Return ds_key -> metrics dict for complete entries in json_path.

    Entries are considered complete when they have all _DATASET_RESULT_KEYS.
    An entry that is otherwise complete but lacks the new ``timing`` key is
    still accepted — timing is additive metadata only.
    """
    if not json_path or not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    datasets = None
    if isinstance(payload, dict) and "datasets" in payload and isinstance(payload["datasets"], dict):
        datasets = payload["datasets"]
    if datasets is None:
        return {}
    out = {}
    for ds_key, row in datasets.items():
        if not isinstance(row, dict):
            continue
        if any(k not in row for k in _DATASET_RESULT_KEYS):
            continue
        try:
            out[ds_key] = {
                "psnr_input": float(row["psnr_input"]),
                "psnr_cross": float(row["psnr_cross"]),
                "lpips_input": float(row["lpips_input"]),
                "lpips_cross": float(row["lpips_cross"]),
                "ssim_input": float(row["ssim_input"]),
                "ssim_cross": float(row["ssim_cross"]),
                "n_input": int(row["n_input"]),
                "n_cross": int(row["n_cross"]),
            }
            if "timing" in row:
                out[ds_key]["timing"] = row["timing"]
        except (TypeError, ValueError):
            continue
    return out
