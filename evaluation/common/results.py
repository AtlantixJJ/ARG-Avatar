"""Aggregation, table printing, image helpers, and visualization utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Simple image I/O helpers (replaces eval_utils.save_rgb / resize_like)
# ---------------------------------------------------------------------------

def save_rgb(path: str, image: np.ndarray) -> None:
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def save_rgba(path: str, rgb: np.ndarray, alpha: np.ndarray) -> None:
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(alpha, 0, 255).astype(np.uint8)
    if alpha_u8.ndim == 3:
        if alpha_u8.shape[-1] == 1:
            alpha_u8 = alpha_u8[..., 0]
        elif alpha_u8.shape[0] == 1:
            alpha_u8 = alpha_u8[0]
        else:
            alpha_u8 = alpha_u8[..., 0]
    rgba = np.concatenate([rgb_u8[..., :3], alpha_u8[..., None]], axis=-1)
    Image.fromarray(rgba, mode="RGBA").save(path)


def resize_like(image: np.ndarray, target_hw: tuple) -> np.ndarray:
    target_h, target_w = target_hw
    with Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)) as pil:
        return np.asarray(pil.resize((target_w, target_h), Image.BILINEAR))


def chw_tensor_to_uint8(t) -> np.ndarray:
    """Convert a CHW float32 tensor [0, 1] to HWC uint8."""
    import torch
    return (t.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def mask_tensor_to_uint8(t) -> np.ndarray:
    """Convert a mask tensor [1,H,W] or [H,W] in [0,1] to HW uint8."""
    mask = t.detach().cpu().float().clamp(0, 1)
    if mask.ndim == 3:
        mask = mask[0]
    return (mask.numpy() * 255).astype(np.uint8)


def make_comparison_image(gt: np.ndarray, render: np.ndarray) -> np.ndarray:
    """Half-size GT | render side-by-side comparison."""
    comp_h, comp_w = gt.shape[0] // 2, gt.shape[1] // 2
    return np.concatenate([resize_like(gt, (comp_h, comp_w)), resize_like(render, (comp_h, comp_w))], axis=1)


# ---------------------------------------------------------------------------
# Eval table printing
# ---------------------------------------------------------------------------

def print_eval_table(ds_results: Dict[str, Dict]) -> None:
    from xlib.train.metrics import compute_overall_eval_metrics
    col_w = 12
    header_fmt = f"{{:<30}} {{:>{col_w}}} {{:>{col_w}}} {{:>{col_w}}} {{:>{col_w}}} {{:>{col_w}}} {{:>{col_w}}} {{:>8}} {{:>8}}"
    row_fmt = f"{{:<30}} {{:>{col_w}.3f}} {{:>{col_w}.3f}} {{:>{col_w}.4f}} {{:>{col_w}.4f}} {{:>{col_w}.4f}} {{:>{col_w}.4f}} {{:>8}} {{:>8}}"
    sep = "=" * (30 + 6 * col_w + 2 * 9 + 6)
    print("\n" + sep)
    print(header_fmt.format("Dataset", "PSNR Input", "PSNR Cross", "LPIPS Input", "LPIPS Cross", "SSIM Input", "SSIM Cross", "N In", "N Cross"))
    print(sep)
    for ds_key, r in ds_results.items():
        print(row_fmt.format(
            ds_key,
            r.get("psnr_input", 0.0), r.get("psnr_cross", 0.0),
            r.get("lpips_input", 0.0), r.get("lpips_cross", 0.0),
            r.get("ssim_input", 0.0), r.get("ssim_cross", 0.0),
            r.get("n_input", 0), r.get("n_cross", 0),
        ))
    if len(ds_results) > 1:
        overall = compute_overall_eval_metrics(ds_results)
        print(sep)
        print(row_fmt.format(
            "OVERALL",
            overall["psnr_input"], overall["psnr_cross"],
            overall["lpips_input"], overall["lpips_cross"],
            overall["ssim_input"], overall["ssim_cross"],
            overall["n_input"], overall["n_cross"],
        ))
    print(sep)


# ---------------------------------------------------------------------------
# Visualization helper (ARGAvatar eval path)
# ---------------------------------------------------------------------------

def encoded_artifact_name(image_path, suffix: str = ".jpg") -> str:
    """Frame-unique render filename mirroring the dataset relpath encoding
    used by scripts/figure/collect_qualitative.py (parts joined with '__')."""
    p = Path(image_path)
    return f"{p.parent.parent.name}__{p.parent.name}__{p.stem}{suffix}"


def save_sample_visualizations(
    vis_dir: str,
    sample_name: str,
    render_entries: List[Dict[str, Any]],
) -> None:
    """Save per-view renders and GTs for one sample."""
    sample_dir = os.path.join(vis_dir, sample_name)
    render_subdir = os.path.join(sample_dir, "render")
    render_human_input_subdir = os.path.join(sample_dir, "render_human_input")
    os.makedirs(render_subdir, exist_ok=True)
    os.makedirs(render_human_input_subdir, exist_ok=True)

    for entry in render_entries:
        render_np = chw_tensor_to_uint8(entry["render_result"]["body"]["image"][0])
        gt_np = chw_tensor_to_uint8(entry["gt_image"])

        image_path = entry.get("image_path")
        assert image_path is not None, "image_path is required"
        fname = encoded_artifact_name(image_path, suffix=".png")

        render_mask = entry["render_result"]["body"].get("mask")
        if render_mask is not None:
            alpha_np = mask_tensor_to_uint8(render_mask[0])
            save_rgba(os.path.join(render_subdir, fname), render_np, alpha_np)
        else:
            save_rgb(os.path.join(render_subdir, fname), render_np)
        save_rgb(os.path.join(render_human_input_subdir, fname), gt_np)
