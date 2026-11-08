#!/usr/bin/env python
"""ARG-Avatar inference: the one command-line entrypoint.

Reconstruct an avatar from one or more images of a person, render it, export its
Gaussians, and optionally animate it with MHR motion files. Every subject -- one photo
or six -- becomes the same ``[B=1, V, ...]`` view-stacked batch; a single-view
reconstruction is just ``V=1``.

One subject: pass its views as files, a folder, or a glob::

    python scripts/inference/infer_argavatar.py --images "assets/example_images/2k2k/00093_0/*.png"

Many subjects: ``--input_dir`` is a parent folder with one subfolder of views per
subject, and the model is loaded once for all of them::

    subjects/
      alice/  front.png  back.png  left.png  right.png   # V=4
      bob/    photo.jpg                                  # V=1, single-view

    python scripts/inference/infer_argavatar.py --input_dir subjects

Per subject, ``<output_folder>/<subject>/`` gets the re-rendered input views
(``render/``), eight canonical rest-pose stills, a 360-degree turntable video, and with

* ``--save_ply``      the canonical Gaussians as ``canonical.ply`` (``--ply_format``),
* ``--save_avatar``   the canonical Gaussians + pose as ``avatar_<model>_<ckpt>.pth``,
* ``--motion M.pth``  one animation per motion under ``motion/``; add ``--ply_frame N``
                      to also export frame N of each motion as a posed ``.ply``.

Motion files are ``sam3db``-format dicts -- ``{"<folder>/<frame>.png": {pose + camera}}``
-- as written by ``scripts/inference/estimate_motion_mhr.py``.

The command line runs through ``scripts/inference/pipeline.ARGAvatarPipeline``, like the
Gradio app. The rest of this module is the low-level helpers that pipeline (and
``estimate_motion_mhr.py``) build on.
"""
import argparse
import glob
import os
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, ".")

from xlib.utils import load_config
from xlib.train.train_helper import (
    setup_model_sam3d,
    get_unwrapped_model,
    resolve_sam3d_cfg,
)
from xlib.model.sam_3d_body import SAM3DBodyEstimator
from xlib.model.sam_3d_body.data.utils.prepare_batch import prepare_batch
from xlib.model.sam_3d_body.utils import recursive_to
from xlib.model.sam_3d_body.tools.build_detector import HumanDetector
from xlib.model.sam_3d_body.tools.build_sam import HumanSegmentor
from xlib.model.sam_3d_body.tools.build_fov_estimator import FOVEstimator
from xlib.render.camera import intrinsics_to_fov
from xlib.dataset.common import apply_affine_to_intrinsics
from xlib.dataset.sam3d_image_dataset import SAM3DImageDataset, BG_COLOR
from xlib.render.gaussian_splatting import set_gaussian_type
from xlib.model.mhr_animatable_gaussians import animate_gs
from xlib.train.train_helper import build_gs_camera, render_posed_gs


DEFAULT_CONFIG_PATH = "configs/ARGAvatar-Final.yaml"
DEFAULT_OUTPUT_FOLDER = "expr/inference"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_rgb_image(image_path: str):
    """Load an image as (PIL image, RGB uint8 numpy array)."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image, np.array(image)


def find_latest_checkpoint(project_name: str, output_dir: str = "expr") -> Optional[str]:
    """Find the latest checkpoint for a project, preferring PT_final then newest PT_<step> file."""
    if not project_name:
        return None

    project_dir = os.path.join(output_dir, project_name)
    if not os.path.isdir(project_dir):
        return None

    final_path = os.path.join(project_dir, "PT_final.pth")
    if os.path.isfile(final_path):
        return final_path

    candidates = []
    for path in glob.glob(os.path.join(project_dir, "PT_*.pth")):
        base = os.path.basename(path)
        if base == "PT_final.pth":
            continue
        step_str = os.path.splitext(base)[0].replace("PT_", "")
        if step_str.isdigit():
            candidates.append((os.path.getmtime(path), int(step_str), path))

    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][2]


def resolve_checkpoint_path(cli_checkpoint_path: str, train_args) -> str:
    """Resolve checkpoint from CLI, config load_checkpoint, or latest checkpoint in expr/<project>."""
    if cli_checkpoint_path:
        if not os.path.exists(cli_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {cli_checkpoint_path}")
        return cli_checkpoint_path

    model_cfg = getattr(train_args, "model", None)
    cfg_checkpoint = ""
    if model_cfg is not None:
        cfg_checkpoint = model_cfg.get("load_checkpoint", "")
    if cfg_checkpoint:
        if not os.path.exists(cfg_checkpoint):
            raise FileNotFoundError(f"Configured checkpoint not found at: {cfg_checkpoint}")
        return cfg_checkpoint

    project_name = getattr(train_args, "project_name", "")
    output_dir = getattr(getattr(train_args, "training", None), "output_dir", "expr")
    latest_checkpoint = find_latest_checkpoint(project_name, output_dir=output_dir)
    if latest_checkpoint is not None:
        return latest_checkpoint

    raise FileNotFoundError(
        "Could not resolve a checkpoint. Pass --checkpoint_path, set model.load_checkpoint in the config, "
        f"or place checkpoints under {output_dir}/{project_name}/."
    )


def get_hr_image_size(train_args) -> Optional[tuple[int, int]]:
    dataset_cfg = getattr(train_args, "dataset", None)
    hr_image_size = None
    if dataset_cfg is not None:
        hr_image_size = dataset_cfg.get("hr_image_size", None)
    if hr_image_size is None:
        return None
    if len(hr_image_size) != 2:
        raise ValueError(f"dataset.hr_image_size must be [H, W], got: {hr_image_size}")
    return int(hr_image_size[0]), int(hr_image_size[1])


def get_estimator(model, model_cfg, args, device) -> SAM3DBodyEstimator:
    """Initialize SAM3DBodyEstimator with optional components."""
    human_detector = None
    if args.detector_name:
        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=args.detector_path
        )

    human_segmentor = None
    if args.segmentor_name and args.segmentor_path:
        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=args.segmentor_path
        )

    fov_estimator = None
    if args.fov_name:
        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=args.fov_path)

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )


def list_view_images(folder: str) -> list:
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ]
    return files


def discover_subjects(input_dir: str) -> list:
    """Return list of (subject_name, [view_paths]).

    Every subdirectory of input_dir is one subject, and the images inside it are that
    subject's views — one person per image. A subject with one image is just V=1. Loose
    images directly under input_dir are not subjects and are ignored.
    """
    subjects = []
    for name in sorted(os.listdir(input_dir)):
        sub = os.path.join(input_dir, name)
        if os.path.isdir(sub):
            views = list_view_images(sub)
            if views:
                subjects.append((name, views))
    return subjects


def detect_all_bboxes(estimator, image_path, bbox_thr=0.5):
    """Detect all humans in image. Returns (img_pil, img_rgb, bboxes [N,4]) sorted by area desc."""
    img_pil, img_rgb = load_rgb_image(image_path)
    height, width = img_rgb.shape[:2]
    img_bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])

    if estimator.detector is not None:
        bboxes = estimator.detector.run_human_detection(img_bgr, bbox_thr=bbox_thr)
        if len(bboxes) == 0:
            return img_pil, img_rgb, bboxes
    else:
        bboxes = np.array([[0, 0, width, height]])

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    bboxes = bboxes[np.argsort(areas)[::-1]]
    return img_pil, img_rgb, bboxes


def prepare_view(estimator, img_pil, img_rgb, bbox, device, hr_image_size):
    """Process one detected person (single bbox [1,4]) into [B=1,V=1,...] batch tensors.

    Returns (view_batch, img_pil, raw_mask_pil).
    """
    height, width = img_rgb.shape[:2]
    bboxes = bbox.reshape(1, 4)

    if estimator.sam is None:
        raise RuntimeError("Segmentor (SAM) is required for inference")
    masks, masks_score = estimator.sam.run_sam(img_rgb, bboxes)
    mask_hw = masks[0].astype(np.float32)  # [H, W] in {0,1}
    img_rgb_masked = (img_rgb * mask_hw[..., None]).astype(np.uint8)

    batch = prepare_batch(img_rgb, estimator.transform, bboxes, masks, masks_score)
    batch = recursive_to(batch, device)
    estimator.model._initialize_batch(batch)

    if estimator.fov_estimator is None:
        raise RuntimeError("FOV estimator is required for inference")
    cam_int = estimator.fov_estimator.get_cam_intrinsics(img_rgb).to(batch["img"])
    batch["cam_int"] = cam_int.clone()
    fov_x, fov_y = intrinsics_to_fov(cam_int)
    batch["raw_fov_x"] = fov_x
    batch["raw_fov_y"] = fov_y

    bg = torch.tensor(BG_COLOR, device=device).view(1, 1, 3, 1, 1)
    batch["img"] = batch["img"] * batch["mask"] + (1 - batch["mask"]) * bg

    raw_cam_int = batch["cam_int"].clone()
    if raw_cam_int.ndim == 4:
        raw_cam_int = raw_cam_int[:, 0]
    affine = batch["affine_trans"]
    if affine.ndim == 4:
        affine = affine[:, 0]
    crop_cam_int = torch.stack(
        [
            apply_affine_to_intrinsics(raw_cam_int[i].detach().cpu().numpy(), affine[i])
            for i in range(raw_cam_int.shape[0])
        ],
        dim=0,
    ).to(device)

    view = {
        "images": batch["img"],                 # [1,1,3,H,W]
        "masks": batch["mask"].float(),         # [1,1,1,H,W]
        "K_raw": raw_cam_int.unsqueeze(1).clone(),    # [1,1,3,3]
        "K_proc": crop_cam_int.unsqueeze(1).clone(),  # [1,1,3,3]
        "crop_bboxes": torch.as_tensor(bboxes[0], device=device, dtype=torch.float32).view(1, 1, 4),
        "raw_sizes": torch.tensor([[height, width]], device=device, dtype=torch.float32).unsqueeze(1),
        "bbox_center": batch["bbox_center"],    # [1,1,2]
        "bbox_scale": batch["bbox_scale"],      # [1,1,2]
        "img_size": batch["img_size"],          # [1,1,2]
        "ori_img_size": batch["ori_img_size"],  # [1,1,2]
        "affine_trans": batch["affine_trans"],  # [1,1,2,3]
        "mask_score": batch["mask_score"],      # [1,1]
        "person_valid": batch["person_valid"],  # [1,1]
    }

    if hr_image_size is not None:
        hr_h, hr_w = hr_image_size
        proc_h, proc_w = batch["img"].shape[-2:]
        aff = batch["affine_trans"][0, 0].detach().cpu().numpy().astype(np.float64)
        scale = np.array([[hr_w / proc_w, 0, 0], [0, hr_h / proc_h, 0], [0, 0, 1]])
        hr_affine = (scale @ np.vstack([aff, [0, 0, 1]]))[:2]
        hr_warped = cv2.warpAffine(
            img_rgb_masked, hr_affine, (hr_w, hr_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        hr_img_t = torch.from_numpy(hr_warped).permute(2, 0, 1).float() / 255.0
        view["hr_images"] = hr_img_t.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,3,hr_h,hr_w]

    raw_mask_pil = Image.fromarray((mask_hw * 255.0).round().astype(np.uint8), mode="L")
    return view, img_pil, raw_mask_pil


def stack_views(view_batches: list, device: torch.device) -> dict:
    """Stack a list of single-view ([1,1,...]) dicts into one [1,V,...] batch."""
    keys = view_batches[0].keys()
    batch = {}
    for k in keys:
        batch[k] = torch.cat([vb[k] for vb in view_batches], dim=1)
    return batch


def build_input_pose_dic(num_views: int, device: torch.device) -> dict:
    """Pre-allocate a zero input_pose_dic (has_pose=False) so the model estimates
    per-view pose+camera with SAM-3D-Body inside get_input_pose_dic."""
    per_view = SAM3DImageDataset._zero_input_pose()
    pose_dic = {}
    for k, v in per_view.items():
        stacked = torch.stack([v.clone() for _ in range(num_views)], dim=0)  # [V, ...]
        pose_dic[k] = stacked.unsqueeze(0).to(device)                        # [1, V, ...]
    pose_dic["has_pose"] = torch.zeros(1, num_views, dtype=torch.bool, device=device)
    return pose_dic


def collect_subject_views(estimator, view_paths, device, hr_image_size, args):
    """Turn one subject's images into per-view batches — one person per image, its top
    detection becoming that view. Returns (view_batches, raw_imgs, raw_masks, view_names).
    """
    view_paths = list(view_paths)[: args.max_views]
    view_batches, raw_imgs, raw_masks, view_names = [], [], [], []
    for image_path in view_paths:
        img_pil, img_rgb, bboxes = detect_all_bboxes(estimator, image_path, args.bbox_thr)
        if len(bboxes) == 0:
            print(f"  WARNING: no person detected in {image_path}, skipping this view.")
            continue
        vb, raw_img, raw_mask = prepare_view(estimator, img_pil, img_rgb, bboxes[0], device, hr_image_size)
        view_batches.append(vb)
        raw_imgs.append(raw_img)
        raw_masks.append(raw_mask)
        view_names.append(image_path)
    print(f"  Detected a person in {len(view_batches)}/{len(view_paths)} image(s)")
    return view_batches, raw_imgs, raw_masks, view_names


def build_subject_batch(view_batches, raw_imgs, raw_masks, view_names, device) -> dict:
    """Stack per-view batches into the one [B=1, V, ...] format the model consumes."""
    V = len(view_batches)
    batch = stack_views(view_batches, device)
    batch["bg_color"] = torch.ones(1, V, 3, device=device)
    batch["input_view_mask"] = torch.ones(V, dtype=torch.bool, device=device)
    batch["input_pose_dic"] = build_input_pose_dic(V, device)
    batch["raw_images"] = [raw_imgs]
    batch["raw_masks"] = [raw_masks]
    batch["image_paths"] = [[p] for p in view_names]  # [V][B]

    batch["images"] = batch["images"] * batch["masks"] + (1 - batch["masks"]) * batch["bg_color"].view(1, V, 3, 1, 1)
    if "hr_images" in batch:
        Hh, Wh = batch["hr_images"].shape[-2:]
        sr_mask = torch.nn.functional.interpolate(
            batch["masks"].view(V, 1, *batch["masks"].shape[-2:]), size=(Hh, Wh),
            mode="bilinear", align_corners=False,
        ).view(1, V, 1, Hh, Wh)
        batch["hr_images"] = batch["hr_images"] * sr_mask + (1 - sr_mask) * batch["bg_color"].view(1, V, 3, 1, 1)
    return batch


def reconstruct_subject(model, estimator, subject_name, source_images, args, hr_image_size, device):
    """Reconstruct one subject from its source image(s) without writing any renders.

    Returns ``(canonical_gs_dic, avatar_tokens)``, or None when too few views are usable.
    ``avatar_tokens`` is what drives the per-frame pose-dependent offset in animation mode.
    """
    if isinstance(source_images, (str, os.PathLike)):
        source_images = [str(source_images)]
    view_batches, raw_imgs, raw_masks, view_names = collect_subject_views(
        estimator, list(source_images), device, hr_image_size, args
    )
    if len(view_batches) < args.min_views:
        print(f"  WARNING: need >= {args.min_views} view(s) for {subject_name}, "
              f"got {len(view_batches)}, skipping.")
        return None

    batch = build_subject_batch(view_batches, raw_imgs, raw_masks, view_names, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        result = get_unwrapped_model(model).two_phase_model_forward(batch)
    last_pass = result["passes"][-1]
    return last_pass["output_canonical_gs_dic"], last_pass["avatar_tokens"]


# --- SAM-3D-Body pose records ------------------------------------------------
# The `sam3db.pth` format: {"<folder>/<file>.png": {pose + camera tensors}}. Motion files
# passed to --driving_mhr_pose are dicts in exactly this shape, as written by
# scripts/inference/estimate_motion_mhr.py.

POSE_KEEP_KEYS = [
    "pred_pose_raw",
    "shape",
    "scale",
    "hand",
    "face",
    "pred_cam",
    "pred_keypoints_2d",
    "pred_keypoints_3d",
    "pred_joint_coords",
    "mhr_model_params",
    "pred_cam_t",
    "cam_rot",
    "cam_trans",
]


def image_key(image_folder: str, filename: str) -> str:
    return f"{image_folder}/{filename}"


def build_sam3db_entry(image_folder: str, filename: str, view: dict, pose_dic: dict) -> tuple[str, dict]:
    key = image_key(image_folder, filename)
    entry = {}
    for pose_key in POSE_KEEP_KEYS:
        if pose_key in pose_dic and isinstance(pose_dic[pose_key], torch.Tensor):
            value = pose_dic[pose_key]
            if value.ndim >= 2 and value.shape[:2] == (1, 1):
                value = value[0, 0]
            elif value.ndim >= 1 and value.shape[0] == 1:
                value = value[0]
            entry[pose_key] = value.detach().cpu().float()

    cache_map = {
        "crop_bbox": "crop_bboxes",
        "K_raw": "K_raw",
        "K_proc": "K_proc",
        "bbox_center": "bbox_center",
        "bbox_scale": "bbox_scale",
        "img_size": "img_size",
        "ori_img_size": "ori_img_size",
        "affine_trans": "affine_trans",
        "mask_score": "mask_score",
        "person_valid": "person_valid",
        "raw_size": "raw_sizes",
    }
    for out_key, view_key in cache_map.items():
        value = view[view_key]
        if isinstance(value, torch.Tensor):
            while value.ndim > 0 and value.shape[0] == 1:
                value = value[0]
            entry[out_key] = value.detach().cpu().float()
    entry["bg_color"] = torch.ones(3, dtype=torch.float32)
    return key, entry


def parse_gpus(gpus: str) -> list[int]:
    if gpus == "all":
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    return [int(x.strip()) for x in gpus.split(",") if x.strip()]


def init_model_and_estimator(args, gpu_id: int | None):
    """Build model + estimator on one GPU. Used by the multi-GPU workers in
    scripts/inference/estimate_motion_mhr.py, which run one of these per rank."""
    device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else torch.device("cpu")
    if gpu_id is not None:
        torch.cuda.set_device(device)
    train_args = load_config(args.config)
    cfg = resolve_sam3d_cfg(train_args, args.config)
    set_gaussian_type(train_args.model.get("gaussian_type", "3d"))
    if args.checkpoint_path:
        train_args.model["load_checkpoint"] = args.checkpoint_path
    model = setup_model_sam3d(train_args, cfg, device)
    model.eval()
    estimator = get_estimator(model, cfg, args, device)
    return model, estimator, get_hr_image_size(train_args), device


def run_parallel(worker_fn, gpu_ids: list[int], *worker_args) -> None:
    n_workers = max(1, len(gpu_ids))
    if n_workers == 1:
        worker_fn(0, gpu_ids, *worker_args)
        return
    ctx = get_context("spawn")
    procs = []
    for rank in range(n_workers):
        proc = ctx.Process(target=worker_fn, args=(rank, gpu_ids, *worker_args))
        proc.start()
        procs.append(proc)
    failures = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failures.append(proc.exitcode)
    if failures:
        raise RuntimeError(f"Worker failures: {failures}")


def extract_avatar_data(last_pass: dict) -> dict:
    """Extract the canonical Gaussians and first-frame pose from a forward pass."""
    avatar_data = {}

    def to_cpu(d):
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    for key in ("output_gs_dic", "output_canonical_gs_dic", "canonical_gs"):
        if key in last_pass and last_pass[key] is not None:
            avatar_data["gs_tpose"] = to_cpu(last_pass[key])
            break

    for key in ("predicted_pose_dic", "mhr"):
        if key in last_pass and last_pass[key] is not None:
            avatar_data["first_pose_dic"] = to_cpu(last_pass[key])
            break
    return avatar_data


# --- Animation mode (--driving_mhr_pose) -------------------------------------


def fps_arg(fps: float) -> str:
    return f"{fps:.8f}".rstrip("0").rstrip(".")


def compute_frame_offset_pose(unwrapped, avatar_tokens: torch.Tensor, model_params: torch.Tensor) -> torch.Tensor:
    """Per-frame pose-dependent Gaussian offset, mirroring the supervision-view branch of
    ``two_phase_model_forward`` (see ``evaluation/argavatar/eval_argavatar.py``): the target pose
    is embedded, fused with the reconstructed avatar tokens, and decoded into ``offset_pose``.

    Args:
        avatar_tokens: [1, L, C] avatar tokens from the reconstruction pass.
        model_params: [1, D] factorized (root-rotation-removed) pose for this frame.
    """
    pose_embed = unwrapped.pose_proj(model_params).unsqueeze(1)
    posed_token = unwrapped.pose_dep_mlp(avatar_tokens + pose_embed)
    deform = unwrapped.compute_target_deform_dic(posed_token)
    return deform["offset_pose"]


def _animate_motion_frame_gs(unwrapped, gs_tpose: dict, avatar_tokens: torch.Tensor, entry: dict, device: torch.device):
    """Animate the canonical Gaussians into one motion-frame pose, independent of render camera.

    Instead of baking the motion pose straight into the canonical Gaussians, this computes the
    frame's pose-dependent ``offset_pose`` and feeds it to ``animate_gs``, matching the eval path.
    Models built without a pose_dep branch simply animate with no offset.
    """
    animatable_gs = unwrapped.mhr_animatable_gs

    def tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(entry[key], dtype=torch.float32, device=device)

    model_params = tensor("mhr_model_params").view(1, 1, -1)
    pose_dic = {
        "shape_params": tensor("shape").view(1, -1),
        "model_params": model_params,
        "expr_params": tensor("face").view(1, 1, -1),
    }
    lbs_cache = animatable_gs.precompute_lbs_cache([pose_dic])

    offset_pose = None
    if getattr(unwrapped, "pose_dep_enabled", False) and avatar_tokens is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            offset_pose = compute_frame_offset_pose(unwrapped, avatar_tokens, model_params.view(1, -1))

    gs_posed, _ = animate_gs(
        animatable_gs, gs_tpose, None,
        lbs_cache=lbs_cache, pose_indices=[0],
        to_colmap=True, offset_pose=offset_pose,
    )
    return gs_posed


def _render_posed_gs_rgba(gs_posed, entry: dict, device: torch.device, K: torch.Tensor, H_v: int, W_v: int, include_alpha: bool = True) -> np.ndarray:
    """Render already-posed Gaussians with the given intrinsics/size. Returns an HxWx4 uint8
    RGBA frame with the rendered Gaussian mask as the alpha channel."""

    def tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(entry[key], dtype=torch.float32, device=device)

    cam, _ = build_gs_camera(
        tensor("pred_cam_t").view(-1), K, H_v, W_v,
        K_raw=tensor("K_raw").view(3, 3), raw_size=tensor("raw_size"),
        cam_rot=tensor("cam_rot").view(3, 3),
        model_trans=tensor("cam_trans").view(3),
    )
    render_result = render_posed_gs(
        gs_posed, b_idx=0, cam=cam,
        bg_color=torch.ones(3, dtype=torch.float32, device=device),
    )
    body = render_result["body"]
    frame = body["image"][0].detach().float().clamp(0, 1)
    rgb = frame.mul(255).round().to(torch.uint8).permute(1, 2, 0)
    if not include_alpha:
        return rgb.contiguous().cpu().numpy()
    mask_t = body.get("mask")
    if mask_t is not None:
        mask = mask_t[0].detach().float().clamp(0, 1)
        while mask.ndim > 2:
            mask = mask[0]
        alpha = mask.mul(255).round().to(torch.uint8)
    else:
        alpha = torch.full(rgb.shape[:2], 255, dtype=torch.uint8, device=rgb.device)
    return torch.cat([rgb, alpha[..., None]], dim=-1).contiguous().cpu().numpy()


def render_motion_frame_crop(gs_posed, entry: dict, device: torch.device, max_render_height: int) -> np.ndarray:
    """Render one already-posed motion frame in human-crop space: ``K_proc``/``img_size`` (the
    human-centered crop the model was supervised on), scaled to ``max_render_height`` — matching
    ``build_precomputed_pose_camera_cache``, not the raw/original video frame."""

    def tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(entry[key], dtype=torch.float32, device=device)

    # img_size is cached as [W, H] (see xlib/model/sam_3d_body/data/transforms/common.py),
    # unlike raw_size which is [H, W] — do not swap these.
    W_proc = int(tensor("img_size")[0].item())
    H_proc = int(tensor("img_size")[1].item())
    scale = float(max_render_height) / float(H_proc) if max_render_height > 0 else 1.0
    H_v = max(1, int(round(H_proc * scale)))
    W_v = max(1, int(round(W_proc * scale)))
    K = tensor("K_proc").view(3, 3).clone()
    if scale != 1.0:
        K[0] *= scale
        K[1] *= scale
    return _render_posed_gs_rgba(gs_posed, entry, device, K, H_v, W_v)


def render_motion_frame_original(gs_posed, entry: dict, device: torch.device,
                                 max_render_height: int = 0, include_alpha: bool = True) -> np.ndarray:
    """Render source camera framing, optionally capped in height for streaming."""
    raw_size = torch.as_tensor(entry["raw_size"]).reshape(-1).tolist()
    H_raw, W_raw = int(round(raw_size[0])), int(round(raw_size[1]))
    scale = min(1.0, max_render_height / H_raw) if max_render_height > 0 else 1.0
    H_v, W_v = max(1, round(H_raw * scale)), max(1, round(W_raw * scale))
    K = torch.as_tensor(entry["K_raw"], dtype=torch.float32, device=device).view(3, 3).clone()
    K[0] *= W_v / W_raw
    K[1] *= H_v / H_raw
    return _render_posed_gs_rgba(gs_posed, entry, device, K, H_v, W_v, include_alpha)


def render_motion_frame_window(
    gs_posed, entry: dict, device: torch.device,
    crop_xyxy: tuple[float, float, float, float], out_hw: tuple[int, int],
) -> np.ndarray:
    """Render one already-posed motion frame through a fixed pixel window in the source
    frame's own raw coordinates (``crop_xyxy``), resized to ``out_hw``.

    Unlike ``render_motion_frame_original`` (the full raw frame) and ``render_motion_frame_crop``
    (the model's human-centered crop), the window here is caller-chosen -- e.g. a fixed
    per-clip subject box for a guidance-corpus build. ``K_raw``/``raw_size`` are left at the
    entry's own full-frame values so ``build_gs_camera``'s ``full_K`` normalisation stays
    correct; only the intrinsics passed to the renderer are re-centered and rescaled:
    ``fx,fy *= s; cx -= x0; cy -= y0; cx,cy *= s``.
    """

    def tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(entry[key], dtype=torch.float32, device=device)

    x0, y0, x1, y1 = crop_xyxy
    H_v, W_v = out_hw
    sx = float(W_v) / float(x1 - x0)
    sy = float(H_v) / float(y1 - y0)
    K = tensor("K_raw").view(3, 3).clone()
    K[0, 0] *= sx
    K[0, 2] = (K[0, 2] - x0) * sx
    K[1, 1] *= sy
    K[1, 2] = (K[1, 2] - y0) * sy
    return _render_posed_gs_rgba(gs_posed, entry, device, K, H_v, W_v)


def load_driving_motion(motion_path: str, max_frames: int = 0) -> tuple[str, dict, list]:
    """Load one MHR motion file. Returns (motion name, sam3db dict, ordered frame keys)."""
    sam3db = torch.load(motion_path, map_location="cpu", weights_only=False)
    if not isinstance(sam3db, dict) or not sam3db:
        raise ValueError(f"{motion_path} is not a non-empty sam3db-style dict")
    frame_keys = sorted(sam3db.keys())
    if max_frames > 0:
        frame_keys = frame_keys[:max_frames]
    return Path(motion_path).stem, sam3db, frame_keys


def iter_subjects(args) -> list:
    """``[(subject_name, [view paths])]`` from ``--images`` (one subject) or ``--input_dir``."""
    if args.images:
        from scripts.inference.pipeline import expand_image_args

        images = expand_image_args(args.images)
        return [(args.name or Path(images[0]).stem, images)]
    return discover_subjects(args.input_dir)


def main(args) -> None:
    # Imported here: pipeline.py imports this module's helpers at load time.
    from scripts.inference.pipeline import ARGAvatarPipeline, PipelineOptions, export_gaussians

    subjects = iter_subjects(args)
    if not subjects:
        raise FileNotFoundError(f"No subjects/views found under {args.input_dir}")

    pipeline = ARGAvatarPipeline.build(PipelineOptions(
        config=args.config,
        checkpoint_path=args.checkpoint_path,
        min_views=args.min_views,
        max_views=args.max_views,
        max_render_height=args.max_render_height,
        bbox_thr=args.bbox_thr,
        detector_name=args.detector_name,
        detector_path=args.detector_path,
        segmentor_name=args.segmentor_name,
        segmentor_path=args.segmentor_path,
        fov_name=args.fov_name,
        fov_path=args.fov_path,
        driving_format=args.driving_format,
        driving_fps=args.driving_fps,
        driving_max_frames=args.driving_max_frames,
    ))
    print(f"Model: {args.config}\nCheckpoint: {pipeline.options.checkpoint_path}")
    centerize = not args.no_centerize
    output_root = Path(args.output_folder)

    for subject, images in subjects:
        print(f"\nSubject {subject!r}: {len(images)} view(s)")
        for path in images:
            print(f"  view: {path}")
        try:
            avatar = pipeline.reconstruct(images, name=subject)
        except RuntimeError as exc:
            if args.input_dir:  # batch mode: skip a subject without enough usable views
                print(f"  WARNING: {exc}; skipping.")
                continue
            raise
        out_dir = output_root / subject
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline.render_views(avatar, str(output_root))
        print(f"  input-view renders  -> {out_dir / 'render'}")
        pipeline.render_canonical(avatar, str(out_dir))
        print(f"  canonical stills    -> {out_dir}/canonical_render_a*.png")
        if not args.no_orbit:
            orbit = pipeline.render_orbit_video(avatar, str(out_dir / "orbit.mp4"),
                                                n_frames=args.orbit_frames)
            print(f"  turntable video     -> {orbit}")
        if args.save_avatar:
            path = pipeline.save_avatar(avatar, str(out_dir / f"avatar_{pipeline.subfix}.pth"))
            print(f"  avatar              -> {path}")
        if args.save_ply:
            path = export_gaussians(pipeline.posed_gaussians(avatar), str(out_dir / "canonical.ply"),
                                    args.ply_format, centerize)
            print(f"  gaussians           -> {path}")

        for motion_path in args.motion:
            print(f"  animating with {Path(motion_path).stem} ...")
            for label, path in pipeline.render_motion(avatar, motion_path, str(out_dir / "motion")).items():
                print(f"  motion ({label})      -> {path}")
            if args.ply_frame is not None:
                gs = pipeline.posed_gaussians(avatar, motion_path=motion_path, frame=args.ply_frame)
                ply = out_dir / "motion" / f"{Path(motion_path).stem}_frame{args.ply_frame:04d}.ply"
                print(f"  posed gaussians     -> {export_gaussians(gs, str(ply), args.ply_format, centerize)}")

    print("\nDone!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ARG-Avatar inference: reconstruct, render, export and animate avatars",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", nargs="+", metavar="PATH",
                        help="One subject's views: image files, a folder, or a glob")
    inputs.add_argument("--input_dir",
                        help="Batch mode: a parent folder with one subfolder of views per subject")
    parser.add_argument("--name", default="",
                        help="Subject name with --images; empty means the first image's stem")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint_path", default="",
                        help="Defaults to the config's model.load_checkpoint")
    parser.add_argument("--min_views", type=int, default=1,
                        help="Need at least this many views with a detected person")
    parser.add_argument("--max_views", type=int, default=6, help="Cap on views per subject")
    parser.add_argument("--max_render_height", type=int, default=1024)

    parser.add_argument("--orbit_frames", type=int, default=90,
                        help="Frames in the 360-degree turntable video")
    parser.add_argument("--no_orbit", action="store_true", help="Skip the turntable video")
    parser.add_argument("--save_avatar", action="store_true",
                        help="Also write avatar_<model>_<ckpt>.pth (canonical Gaussians + pose)")

    export = parser.add_argument_group("Gaussian export")
    export.add_argument("--save_ply", action="store_true",
                        help="Write the canonical Gaussians as canonical.ply")
    export.add_argument("--ply_format", choices=["3dgs", "spark"], default="3dgs",
                        help="3dgs: INRIA layout (SuperSplat, most viewers); spark: compact 14-float layout")
    export.add_argument("--ply_frame", type=int, default=None, metavar="N",
                        help="With --motion: also export frame N of each motion as a posed .ply")
    export.add_argument("--no_centerize", action="store_true",
                        help="Keep the world origin instead of centering the exported Gaussians")

    motion = parser.add_argument_group("animation")
    motion.add_argument("--motion", "--driving_mhr_pose", dest="motion", nargs="+", default=[],
                        metavar="MOTION.pth",
                        help="MHR motion file(s) in sam3db format, as written by "
                             "scripts/inference/estimate_motion_mhr.py")
    motion.add_argument("--driving_format", choices=["video", "images"], default="video",
                        help="Animation output: mp4 videos, or one PNG per frame")
    motion.add_argument("--driving_fps", type=float, default=30.0)
    motion.add_argument("--driving_max_frames", type=int, default=0,
                        help="Use only the first N frames of each motion (0 = all)")

    front = parser.add_argument_group("detection / segmentation / FOV")
    front.add_argument("--detector_name", default="vitdet")
    front.add_argument("--detector_path", default="")
    front.add_argument("--segmentor_name", default="sam2")
    front.add_argument("--segmentor_path", default="thirdparty/sam2")
    front.add_argument("--fov_name", default="moge2")
    front.add_argument("--fov_path", default="")
    front.add_argument("--bbox_thr", type=float, default=0.5)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
