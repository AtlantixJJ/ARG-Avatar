"""Val-list YAML expansion utilities (env-agnostic)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str) -> Any:
    with open(path, "r") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def is_val_list_yaml(path: str) -> bool:
    """Return True if path is a val-list YAML ({data_root, items} format)."""
    if not path.endswith((".yaml", ".yml")):
        return False
    try:
        data = load_yaml(path)
        return (
            isinstance(data, dict)
            and "data_root" in data
            and "items" in data
            and isinstance(data["items"], dict)
        )
    except Exception:
        return False


def expand_val_list_samples(
    yaml_path: str,
    output_root: str,
    num_input_views: int = 0,
    verification_num: int = 0,
    num_sup_views: int = -1,
) -> tuple:
    """Expand a val-list YAML ({data_root, items}) into per-sample run specs.

    Returns (ds_key, ds_out, samples). Each sample dict has:
        sample_key      original key in the YAML items mapping
        image_paths     absolute paths: reference views first, then deduped supervision views
        input_view_mask bool tensor aligned with image_paths (True = reference view)
        sample_out      per-sample output directory under ds_out
        data_root       absolute dataset root

    num_input_views > 0 truncates reference views and appends an _n{N} suffix to ds_key;
    verification_num > 0 limits to the first N samples (quick checks).
    """
    val_data = load_yaml(yaml_path)
    data_root = val_data["data_root"]
    if not os.path.isabs(data_root):
        data_root = os.path.join(str(_REPO_ROOT), data_root)
    items: Dict[str, Any] = val_data["items"]

    ds_key = Path(yaml_path).stem + (f"_n{num_input_views}" if num_input_views > 0 else "")
    ds_out = os.path.join(output_root, ds_key)

    sample_keys = list(items.keys())
    if verification_num and verification_num > 0:
        sample_keys = sample_keys[:verification_num]

    samples: List[Dict[str, Any]] = []
    for sample_key in sample_keys:
        entry = items[sample_key]
        input_rel_paths: List[str] = entry.get("input_images", [])
        if num_input_views > 0:
            input_rel_paths = input_rel_paths[:num_input_views]
        if not input_rel_paths:
            print(f"[WARN] No input_images for sample {sample_key!r}. Skipping.")
            continue
        sup_rel_paths: List[str] = entry.get("images", [])
        input_set = set(input_rel_paths)
        unique_sup = [p for p in sup_rel_paths if p not in input_set]
        if num_sup_views > 0:
            unique_sup = unique_sup[:num_sup_views]
        image_paths = [os.path.join(data_root, p) for p in input_rel_paths + unique_sup]
        input_view_mask = torch.tensor(
            [True] * len(input_rel_paths) + [False] * len(unique_sup), dtype=torch.bool
        )
        sample_stem = sample_key.replace("/", "_").replace(os.sep, "_")
        samples.append({
            "sample_key": sample_key,
            "image_paths": image_paths,
            "input_view_mask": input_view_mask,
            "sample_out": os.path.join(ds_out, sample_stem),
            "data_root": data_root,
        })
    return ds_key, ds_out, samples


def collect_val_list_yamls(
    data_val_list_yaml: str,
    skip_datasets: Optional[List[str]] = None,
) -> List[str]:
    """Return per-dataset val-list YAML paths from a combined data config.

    Reads data.val.ds_list[].image_list entries, resolves them relative to the repo
    root, and filters out datasets named in skip_datasets (case-insensitive,
    matched against the last component of root_dir).
    """
    skip_set = set(s.lower() for s in (skip_datasets or []))
    if is_val_list_yaml(data_val_list_yaml):
        return []
    cfg = load_yaml(data_val_list_yaml)
    ds_list = cfg["data"]["val"]["ds_list"]
    result: List[str] = []
    for ds_cfg in ds_list:
        image_list_rel = ds_cfg.get("image_list", "")
        if not image_list_rel:
            continue
        yaml_path = str(_REPO_ROOT / image_list_rel)
        if not os.path.isfile(yaml_path):
            print(f"[WARN] val list yaml not found, skipping: {yaml_path}")
            continue
        ds_data_root = ds_cfg.get("root_dir", "")
        ds_name = Path(ds_data_root).name if ds_data_root else Path(yaml_path).stem
        if ds_name.lower() in skip_set:
            print(f"[INFO] Skipping dataset: {ds_name} because it is in skip_datasets")
            continue
        result.append(yaml_path)
    return result
