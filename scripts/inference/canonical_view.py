"""Lightweight canonical-view renderer used by multi-view inference."""
import math
import os

import numpy as np
import torch
from PIL import Image

from xlib.model.mhr_animatable_gaussians import animate_gs
from xlib.model_input_utils import slice_batch
from xlib.train.train_helper import build_gs_camera, render_posed_gs


def _save_rgba(path: str, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="RGBA").save(path)


def _scale_to_max_long_side(height: int, width: int, max_size: int) -> tuple[int, int, float]:
    scale = min(max_size / max(height, width), 1.0)
    return int(round(height * scale)), int(round(width * scale)), scale


def _scale_intrinsics(intrinsics: torch.Tensor, scale: float) -> torch.Tensor:
    scaled = intrinsics.clone()
    if scale < 1.0:
        scaled[0] *= scale
        scaled[1] *= scale
    return scaled


def _render_result_to_rgba(render_result) -> np.ndarray:
    image = render_result["body"]["image"][0]
    mask = render_result["body"]["mask"][0, 0]
    rgb = (image.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    alpha = (mask.detach().cpu().float().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def _canonical_bbox_frame(gs_posed, intrinsics: torch.Tensor, height: int, width: int):
    xyz = gs_posed.xyz[0, 0].float()
    lower, upper = xyz.amin(dim=0), xyz.amax(dim=0)
    center, extent = 0.5 * (lower + upper), upper - lower
    horizontal = max(float(extent[0]), float(extent[2]))
    vertical = float(extent[1])
    margin = 0.85
    depth = max(float(intrinsics[0, 0]) * horizontal / max(margin * width, 1e-6),
                float(intrinsics[1, 1]) * vertical / max(margin * height, 1e-6), 1e-3)
    return center, depth


def _orbit_camera(center: torch.Tensor, depth: float, theta: float, intrinsics, height, width):
    sine, cosine = math.sin(theta), math.cos(theta)
    device = center.device
    rotation = torch.stack([
        torch.tensor([cosine, 0.0, -sine], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
        torch.tensor([sine, 0.0, cosine], device=device),
    ])
    eye = center + depth * torch.tensor([-sine, 0.0, -cosine], device=device)
    return build_gs_camera(-rotation @ eye, intrinsics, height, width, cam_rot=rotation)


def render_canonical_view(model, batch, rest_pose_dic, gs_tpose_b, subject_dir: str,
                          max_render_size: int, device: torch.device,
                          canonical_canvas_size: int | None = None) -> None:
    """Render eight rest-pose orbit images without the optional Kaolin stack."""
    if canonical_canvas_size is not None:
        height = width = canonical_canvas_size
        intrinsics = torch.eye(3, device=device)
        intrinsics[0, 0] = intrinsics[1, 1] = canonical_canvas_size * 1.2
        intrinsics[0, 2] = intrinsics[1, 2] = canonical_canvas_size / 2.0
    elif "K" in batch:
        height, width = batch["images"][0, 0].shape[-2:]
        height, width, scale = _scale_to_max_long_side(height, width, max_render_size)
        intrinsics = _scale_intrinsics(batch["K"][0, 0].clone(), scale)
    else:
        height, width = (int(v.item()) for v in batch["raw_sizes"][0, 0])
        height, width, scale = _scale_to_max_long_side(height, width, max_render_size)
        intrinsics = _scale_intrinsics(batch["K_raw"][0, 0].clone(), scale)
    lbs_cache = slice_batch(model.mhr_animatable_gs.precompute_lbs_cache([rest_pose_dic]), 0)
    gs_posed, _ = animate_gs(model.mhr_animatable_gs, gs_tpose_b, None, lbs_cache=lbs_cache,
                              pose_indices=[0], to_colmap=True, offset_pose=None)
    center, depth = _canonical_bbox_frame(gs_posed, intrinsics, height, width)
    background = torch.ones(3, device=device, dtype=torch.float32)
    for angle in range(0, 360, 45):
        camera, _ = _orbit_camera(center, depth, math.radians(angle), intrinsics, height, width)
        result = render_posed_gs(gs_posed, b_idx=0, cam=camera, bg_color=background)
        _save_rgba(os.path.join(subject_dir, f"canonical_render_a{angle:03d}.png"), _render_result_to_rgba(result))
