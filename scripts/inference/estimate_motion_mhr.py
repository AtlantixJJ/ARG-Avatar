#!/usr/bin/env python
"""Fit MHR motion from a video of one person, for use as a driving motion.

The output is a ``sam3db``-format dict -- ``{"images/<frame>.png": {pose + camera}}`` --
at ``expr/motion_pose/<video>/sam3db_<strategy>.pth``, which is exactly what
``infer_argavatar.py --motion`` and the Gradio app consume.

    python scripts/inference/estimate_motion_mhr.py --video my_clip --video_path clip.mp4

Six stages run in order under ``--stage all`` (the default); each can be run alone to
resume a partial fit:

    0a  detect every person box on every frame, then associate them into one track
    0b  score each tracked frame's confidence (mask, keypoints-in-frame, bbox scale)
    0c  gate out low-confidence frames and interpolate short gaps
    0d  subsample the surviving frames (``--eval_stride``)
    0e  reconstruct a reference avatar from the single most confident frame
    strategy  turn the per-frame poses into a temporally coherent motion

``--strategy`` selects how much temporal work that last stage does: ``s0`` none (raw
per-frame poses), ``s1`` median shape/scale across frames, ``s2`` One-Euro filtering,
``s3`` windowed temporal fitting, ``s4`` all of the above. Only ``s3``/``s4`` run the
optimiser, whose mesh-depth term imports ``kaolin``
(installed by ``requirements.txt``); ``s0``-``s2`` never reach it.
"""
import argparse
import gc
import json
import math
import os
import shutil
import subprocess
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, ".")

from scripts.inference.infer_argavatar import (
    build_input_pose_dic,
    build_sam3db_entry,
    detect_all_bboxes,
    get_estimator,
    get_hr_image_size,
    init_model_and_estimator,
    prepare_view,
    reconstruct_subject,
    run_parallel,
    stack_views,
)
from xlib.fitting.motion_filters import (
    interpolate_gated_frames,
    one_euro_filter_track,
    quat_to_rotmat,
    rotmat_to_quat,
    slerp_smooth_rotations,
)
from xlib.fitting.pose_fit_common import prepare_pose_fit_inputs
from xlib.fitting.temporal_pose_fitting import fit_motion_windows
from xlib.train.train_helper import get_unwrapped_model

DEFAULT_CHECKPOINT = "checkpoints/ARGAvatar-Final.pth"

def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit MHR motion (a sam3db-format .pth) from a video of one person.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/ARGAvatar-Final.yaml")
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--root", type=str, default="data/AnimatableHuman/CustomAnimation")
    parser.add_argument("--video", type=str, required=True,
                        help="Name for this motion. Outputs go to expr/motion_pose/<video>/. "
                             "Unless --video_path or --frames_dir is given, frames are read "
                             "from <root>/preprocessed/motions/<video>/images/.")
    parser.add_argument("--video_path", type=str, default="",
                        help="A video file to fit. Its frames are extracted with ffmpeg into "
                             "expr/motion_pose/<video>/images/ and used as the source frames.")
    parser.add_argument("--frames_dir", type=str, default="",
                        help="A directory of already-extracted PNG frames to fit, instead of "
                             "the <root>-relative default.")
    parser.add_argument("--eval_stride", type=int, default=0,
                        help="Keep every Nth confident frame in stage 0d. 0 selects the "
                             "per-video default (4 for woman_dance_1, 8 otherwise).")
    parser.add_argument("--stage", type=str, default="all", choices=["0a", "0b", "0c", "0d", "0e", "strategy", "all"])
    parser.add_argument("--strategy", type=str, choices=["s0", "s1", "s2", "s3", "s4", "s5"], default="s0")
    parser.add_argument("--gpus", type=str, default="all")
    parser.add_argument("--force-restage", action="store_true")
    parser.add_argument("--bbox_thr", type=float, default=0.5)
    parser.add_argument("--detector_name", default="vitdet")
    parser.add_argument("--detector_path", default="")
    parser.add_argument("--segmentor_name", default="sam2")
    parser.add_argument("--segmentor_path", default="thirdparty/sam2")
    parser.add_argument("--fov_name", default="moge2")
    parser.add_argument("--fov_path", default="")
    parser.add_argument("--max_render_height", type=int, default=2048)
    # Stage 0e reconstructs its reference avatar from exactly one selected frame.
    parser.set_defaults(min_views=1, max_views=1)
    parser.add_argument("--fit_iters", type=int, default=300)
    parser.add_argument("--f_min", type=float, default=1.0, help="One-Euro min cutoff (S2/S4).")
    parser.add_argument("--beta", type=float, default=-1.0,
                        help="One-Euro speed coefficient; <0 uses the per-strategy default.")
    parser.add_argument("--out_suffix", type=str, default="",
                        help="Suffix for sam3db_<strategy><suffix>.pth, used by the P2 (f_min, beta) sweep.")
    return parser.parse_args(argv)

def parse_gpus(gpus: str) -> list[int]:
    if gpus == "all":
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    return [int(x.strip()) for x in gpus.split(",") if x.strip()]


def resolve_frames_dir(args) -> Path:
    """The directory of source PNG frames for this motion.

    Three ways to name it, in precedence order: an explicit ``--frames_dir``; a
    ``--video_path`` video file, whose frames are extracted once into
    ``expr/motion_pose/<video>/images/``; or the ``<root>/preprocessed/motions/<video>/images``
    layout, which is where the two bundled dance clips live.
    """
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"--frames_dir not found: {frames_dir}")
        return frames_dir

    if args.video_path:
        if not Path(args.video_path).is_file():
            raise FileNotFoundError(f"--video_path not found: {args.video_path}")
        frames_dir = Path("expr/motion_pose") / args.video / "images"
        if any(frames_dir.glob("*.png")) and not args.force_restage:
            print(f"Frames already extracted in {frames_dir}")
            return frames_dir
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting frames from {args.video_path} -> {frames_dir}")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", args.video_path,
             str(frames_dir / "%06d.png")],
            check=True,
        )
        return frames_dir

    frames_dir = Path(args.root) / "preprocessed" / "motions" / args.video / "images"
    if not frames_dir.is_dir():
        raise FileNotFoundError(
            f"No frames for {args.video!r}: {frames_dir} does not exist. Pass --video_path "
            "to fit a video file, or --frames_dir to point at extracted frames."
        )
    return frames_dir

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def _detect_worker(rank: int, gpu_ids: list[int], frame_paths: list[str], shard_dir: str, args) -> None:
    """Detect every person box in this rank's stripe of frames. Writes one JSON shard."""
    torch.set_grad_enabled(False)
    gpu_id = gpu_ids[rank] if gpu_ids else None
    _, estimator, _, _ = init_model_and_estimator(args, gpu_id)
    n_workers = max(1, len(gpu_ids))
    out = {}
    my_frames = frame_paths[rank::n_workers]
    for idx, fpath in enumerate(my_frames):
        if idx % 200 == 0:
            print(f"[detect rank {rank}] {idx}/{len(my_frames)}", flush=True)
        _, _, bboxes = detect_all_bboxes(estimator, fpath, bbox_thr=args.bbox_thr)
        out[Path(fpath).name] = np.asarray(bboxes).reshape(-1, 4).tolist()
    with open(Path(shard_dir) / f"detect_{rank}.json", "w") as f:
        json.dump(out, f)


def _pose_worker(rank: int, gpu_ids: list[int], frame_paths: list[str], track_path: str, shard_dir: str, args) -> None:
    """Run SAM-3D-Body on this rank's stripe of tracked boxes. Writes one sam3db shard."""
    torch.set_grad_enabled(False)
    gpu_id = gpu_ids[rank] if gpu_ids else None
    model, estimator, hr_image_size, device = init_model_and_estimator(args, gpu_id)
    unwrapped = get_unwrapped_model(model)
    with open(track_path) as f:
        track = json.load(f)
    n_workers = max(1, len(gpu_ids))
    my_frames = frame_paths[rank::n_workers]
    shard = {}
    for idx, fpath in enumerate(my_frames):
        if idx % 200 == 0:
            print(f"[pose rank {rank}] {idx}/{len(my_frames)}", flush=True)
        name = Path(fpath).name
        tdata = track[name]
        if tdata["box"] is None or tdata["score"] < 0.5:
            continue
        img_pil = Image.open(fpath).convert("RGB")
        img_rgb = np.array(img_pil)
        view, _, _ = prepare_view(estimator, img_pil, img_rgb, np.array(tdata["box"]), device, hr_image_size)
        batch = stack_views([view], device)
        batch["bg_color"] = torch.ones(1, 1, 3, device=device)
        batch["input_view_mask"] = torch.ones(1, dtype=torch.bool, device=device)
        batch["input_pose_dic"] = build_input_pose_dic(1, device)
        pose_dic = unwrapped.get_input_pose_dic(batch)
        pose_dic = unwrapped._ensure_factorized_input_pose_dic(dict(pose_dic))
        key, entry = build_sam3db_entry("images", name, view, pose_dic)
        shard[key] = entry
    torch.save(shard, Path(shard_dir) / f"pose_{rank}.pth")


def build_track(frames: list[Path], all_bboxes: list[np.ndarray]) -> dict:
    """Greedy forward/backward association from a seed frame, scored by IoU + size + center continuity."""
    def assoc_score(prev_box, cand):
        s_iou = iou(prev_box, cand)
        prev_w, prev_h = prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]
        cand_w, cand_h = cand[2] - cand[0], cand[3] - cand[1]
        size_ratio = min(prev_w * prev_h, cand_w * cand_h) / max(prev_w * prev_h, cand_w * cand_h, 1e-6)
        prev_c = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2])
        cand_c = np.array([(cand[0] + cand[2]) / 2, (cand[1] + cand[3]) / 2])
        center_dist = float(np.linalg.norm(cand_c - prev_c)) / max(prev_h, 1e-6)
        s_center = math.exp(-center_dist)
        return 0.6 * s_iou + 0.2 * size_ratio + 0.2 * s_center

    # Seed at the frame with the largest detection (the subject is nearest/most complete there).
    seed_idx, seed_area = 0, -1.0
    for i, boxes in enumerate(all_bboxes):
        if len(boxes) == 0:
            continue
        area = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])
        if area > seed_area:
            seed_area, seed_idx = area, i

    track = {}
    for direction in (1, -1):
        curr_box = all_bboxes[seed_idx][0] if len(all_bboxes[seed_idx]) else None
        rng = range(seed_idx, len(frames)) if direction == 1 else range(seed_idx - 1, -1, -1)
        for i in rng:
            boxes = all_bboxes[i]
            if len(boxes) == 0 or curr_box is None:
                track[frames[i].name] = {"box": None, "score": 0.0}
                continue
            scores = [assoc_score(curr_box, b) for b in boxes]
            best = int(np.argmax(scores))
            track[frames[i].name] = {"box": np.asarray(boxes[best]).tolist(), "score": float(scores[best])}
            if scores[best] >= 0.5:
                curr_box = np.asarray(boxes[best])
    return track


def stage_0a_track(args, gpus):
    # Detect all boxes for all frames and build a track
    video_stem = args.video
    frames_dir = resolve_frames_dir(args)

    expr_dir = Path("expr/motion_pose") / video_stem
    expr_dir.mkdir(parents=True, exist_ok=True)

    if (expr_dir / "sam3db_tracked.pth").exists() and not args.force_restage:
        print("Stage 0a already complete.")
        return

    frames = sorted(frames_dir.glob("*.png"))
    frame_paths = [str(p) for p in frames]
    shard_dir = expr_dir / "_shards"
    shard_dir.mkdir(exist_ok=True)

    # Pass 1: detect every person box on every frame, sharded across GPUs.
    print(f"Stage 0a: detecting bboxes on {len(frames)} frames with {max(1, len(gpus))} worker(s)...")
    run_parallel(_detect_worker, gpus, frame_paths, str(shard_dir), args)

    detections = {}
    for rank in range(max(1, len(gpus))):
        with open(shard_dir / f"detect_{rank}.json") as f:
            detections.update(json.load(f))
    all_bboxes = [np.asarray(detections[p.name], dtype=np.float32).reshape(-1, 4) for p in frames]

    track = build_track(frames, all_bboxes)
    with open(expr_dir / "track.json", "w") as f:
        json.dump(track, f)

    # Pass 2: predict pose on the tracked box only, sharded across GPUs.
    print("Stage 0a: predicting pose on tracked boxes...")
    run_parallel(_pose_worker, gpus, frame_paths, str(expr_dir / "track.json"), str(shard_dir), args)

    sam3db_tracked = {}
    for rank in range(max(1, len(gpus))):
        sam3db_tracked.update(torch.load(shard_dir / f"pose_{rank}.pth", map_location="cpu", weights_only=False))
    torch.save(sam3db_tracked, expr_dir / "sam3db_tracked.pth")
    shutil.rmtree(shard_dir)

def stage_0b_confidence(args):
    video_stem = args.video
    expr_dir = Path("expr/motion_pose") / video_stem
    
    with open(expr_dir / "track.json") as f:
        track = json.load(f)
        
    sam3db = torch.load(expr_dir / "sam3db_tracked.pth", map_location="cpu", weights_only=False)
    
    scales = []
    for key, entry in sam3db.items():
        scales.append(entry["bbox_scale"].mean().item())
    median_scale = np.median(scales) if scales else 1.0
    
    confidence = {}
    for key in sorted(track.keys()):
        tdata = track[key]
        db_key = f"images/{key}"
        
        if db_key not in sam3db:
            confidence[key] = {
                "person_valid": 0,
                "assoc_score": tdata["score"],
                "mask_score": 0.0,
                "kp_inframe": 0.0,
                "bbox_scale": 0.0,
                "mask_area_frac": 0.0,
                "accept": False,
                "w_t": 0.0
            }
            continue
            
        entry = sam3db[db_key]
        person_valid = entry["person_valid"].item()
        assoc_score = tdata["score"]
        mask_score = entry["mask_score"].item()
        bbox_scale = entry["bbox_scale"].mean().item()
        
        # kp inside image
        kp2d = entry["pred_keypoints_2d"]
        h_raw, w_raw = entry["raw_size"].tolist()
        inside = ((kp2d[:, 0] >= 0) & (kp2d[:, 0] < w_raw) & (kp2d[:, 1] >= 0) & (kp2d[:, 1] < h_raw))
        kp_inframe_fraction = inside.float().mean().item()
        
        # mask area. We don't have the raw mask in sam3db, but we have crop_bboxes and scale.
        # Estimate from scale or just assume it's fine for now. We can use mask_score as proxy for silhouette validity,
        # but let's approximate mask area frac:
        bbox = tdata["box"]
        if bbox is not None:
            area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            mask_area_frac = area / (w_raw * h_raw)
        else:
            mask_area_frac = 0.0
            
        accept = (
            person_valid == 1 and
            assoc_score >= 0.5 and
            mask_score >= 0.95 and
            kp_inframe_fraction >= 0.95 and
            bbox_scale >= 0.6 * median_scale and
            mask_area_frac >= 0.01
        )
        
        w_t = mask_score * kp_inframe_fraction * max(0.0, min(1.0, assoc_score)) if accept else 0.0
        
        confidence[key] = {
            "person_valid": person_valid,
            "assoc_score": assoc_score,
            "mask_score": mask_score,
            "kp_inframe": kp_inframe_fraction,
            "bbox_scale": bbox_scale,
            "mask_area_frac": mask_area_frac,
            "accept": accept,
            "w_t": w_t
        }
        
    with open(expr_dir / "confidence.json", "w") as f:
        json.dump(confidence, f, indent=2)


def stage_0c_gated_track(args):
    video_stem = args.video
    expr_dir = Path("expr/motion_pose") / video_stem
    
    with open(expr_dir / "confidence.json") as f:
        confidence = json.load(f)
        
    sam3db = torch.load(expr_dir / "sam3db_tracked.pth", map_location="cpu", weights_only=False)
    
    keys = sorted(confidence.keys())
    T = len(keys)
    
    # We need to fill in missing entries using interpolation
    # Extract tensors
    # To interpolate, we first need to ensure every frame has SOME tensor to start with.
    # We'll use the first accepted frame as a dummy for missing ones, then interpolate overwrites it.
    accepted_idx = [i for i, k in enumerate(keys) if confidence[k]["accept"]]
    if not accepted_idx:
        raise ValueError("No accepted frames found!")
        
    dummy_key = f"images/{keys[accepted_idx[0]]}"
    dummy_entry = sam3db[dummy_key]
    
    pose_cont = []
    cam_trans = []
    pred_cam_t = []
    face = []
    cam_rot = []
    
    accept_mask = torch.zeros(T, dtype=torch.bool)
    
    for i, k in enumerate(keys):
        db_key = f"images/{k}"
        if confidence[k]["accept"] and db_key in sam3db:
            entry = sam3db[db_key]
            accept_mask[i] = True
        else:
            entry = dummy_entry
            
        pose_cont.append(entry["mhr_model_params"])
        cam_trans.append(entry["cam_trans"])
        pred_cam_t.append(entry["pred_cam_t"])
        face.append(entry["face"])
        cam_rot.append(entry["cam_rot"])
        
    pose_cont = torch.stack(pose_cont)
    cam_trans = torch.stack(cam_trans)
    pred_cam_t = torch.stack(pred_cam_t)
    face = torch.stack(face)
    cam_rot = torch.stack(cam_rot)
    
    pose_cont = interpolate_gated_frames(pose_cont, accept_mask, is_rotation=False)
    cam_trans = interpolate_gated_frames(cam_trans, accept_mask, is_rotation=False)
    pred_cam_t = interpolate_gated_frames(pred_cam_t, accept_mask, is_rotation=False)
    face = interpolate_gated_frames(face, accept_mask, is_rotation=False)
    
    cam_rot_quats = rotmat_to_quat(cam_rot)
    cam_rot_quats = interpolate_gated_frames(cam_rot_quats, accept_mask, is_rotation=True)
    cam_rot = quat_to_rotmat(cam_rot_quats)
    
    sam3db_gated = {}
    gap_mask = {}
    
    for i, k in enumerate(keys):
        db_key = f"images/{k}"
        if confidence[k]["accept"] and db_key in sam3db:
            sam3db_gated[db_key] = sam3db[db_key]
            gap_mask[k] = False
        else:
            # create new entry
            entry = dummy_entry.copy()
            entry["mhr_model_params"] = pose_cont[i]
            entry["cam_trans"] = cam_trans[i]
            entry["pred_cam_t"] = pred_cam_t[i]
            entry["face"] = face[i]
            entry["cam_rot"] = cam_rot[i]
            sam3db_gated[db_key] = entry
            gap_mask[k] = True
            
    # Mark long gaps
    gap_runs = []
    curr_run = 0
    for i in range(T):
        if gap_mask[keys[i]]:
            curr_run += 1
        else:
            if curr_run > 15:
                for j in range(i - curr_run, i):
                    gap_mask[keys[j]] = "long_gap"
            curr_run = 0
    if curr_run > 15:
        for j in range(T - curr_run, T):
            gap_mask[keys[j]] = "long_gap"
            
    torch.save(sam3db_gated, expr_dir / "sam3db_gated.pth")
    with open(expr_dir / "gap_mask.json", "w") as f:
        json.dump(gap_mask, f, indent=2)

def stage_0d_eval_frames(args):
    video_stem = args.video
    expr_dir = Path("expr/motion_pose") / video_stem
    
    with open(expr_dir / "confidence.json") as f:
        confidence = json.load(f)
    with open(expr_dir / "gap_mask.json") as f:
        gap_mask = json.load(f)
        
    keys = sorted(confidence.keys())
    valid_keys = [k for k in keys if confidence[k]["accept"] and gap_mask.get(k) != "long_gap"]
    
    # Subsample. The two bundled clips keep their original strides so previously produced
    # motions reproduce exactly; --eval_stride overrides for anything else.
    stride = args.eval_stride or (4 if video_stem == "woman_dance_1" else 8)
    eval_keys = valid_keys[::stride]

    if not eval_keys:
        raise ValueError(
            f"No confident frames survived for {video_stem!r} at stride {stride}. "
            "The subject may be too small, occluded, or absent in most frames."
        )

    with open(expr_dir / "confident_frames.json", "w") as f:
        json.dump(eval_keys, f, indent=2)

def stage_0e_avatar(args, gpus):
    video_stem = args.video
    expr_dir = Path("expr/motion_pose") / video_stem
    
    with open(expr_dir / "confident_frames.json") as f:
        eval_keys = json.load(f)
        
    with open(expr_dir / "confidence.json") as f:
        confidence = json.load(f)
        
    # Pick t_ref
    best_t_ref = eval_keys[0]
    best_score = -1.0
    for k in eval_keys:
        if confidence[k]["mask_score"] >= 0.97 and confidence[k]["kp_inframe"] >= 1.0:
            score = confidence[k]["bbox_scale"]
            if score > best_score:
                best_score = score
                best_t_ref = k
                
    frames_dir = resolve_frames_dir(args)

    source_image = str(frames_dir / best_t_ref)
    
    gc.collect()
    torch.cuda.empty_cache()
    model, estimator, hr_image_size, device = init_model_and_estimator(args, gpus[0] if gpus else None)

    recon = reconstruct_subject(model, estimator, video_stem, source_image, args, hr_image_size, device)
    if recon is None:
        raise ValueError("Failed to reconstruct avatar.")
        
    gs_tpose, avatar_tokens = recon
    
    torch.save({
        "gs_tpose": gs_tpose,
        "avatar_tokens": avatar_tokens,
        "t_ref": best_t_ref
    }, expr_dir / "avatar_ref.pth")


def run_strategy(args):
    video_stem = args.video
    strategy = args.strategy
    expr_dir = Path("expr/motion_pose") / video_stem
    
    sam3db_gated = torch.load(expr_dir / "sam3db_gated.pth", map_location="cpu", weights_only=False)
    with open(expr_dir / "confidence.json") as f:
        confidence = json.load(f)
        
    keys = sorted(confidence.keys())
    T = len(keys)
    
    accept_mask = torch.tensor([confidence[k]["accept"] for k in keys], dtype=torch.bool)
    w_t = torch.tensor([confidence[k]["w_t"] for k in keys], dtype=torch.float32)
    
    db_keys = [f"images/{k}" for k in keys]
    pose_cont = torch.stack([torch.as_tensor(sam3db_gated[k]["pred_pose_raw"]) for k in db_keys])
    mhr_params = torch.stack([torch.as_tensor(sam3db_gated[k]["mhr_model_params"]) for k in db_keys]) if "mhr_model_params" in sam3db_gated[db_keys[0]] else None
    shape = torch.stack([sam3db_gated[k]["shape"] for k in db_keys])
    scale = torch.stack([sam3db_gated[k]["scale"] for k in db_keys])
    cam_trans = torch.stack([sam3db_gated[k]["cam_trans"] for k in db_keys])
    pred_cam_t = torch.stack([sam3db_gated[k]["pred_cam_t"] for k in db_keys])
    face = torch.stack([sam3db_gated[k]["face"] for k in db_keys])
    cam_rot = torch.stack([sam3db_gated[k]["cam_rot"] for k in db_keys])
    
    if strategy == "s5":
        raise NotImplementedError(
            "S5 (silhouette term) is a plan stretch goal, gated on S4 being within 0.2 dB of S2. "
            "It is deliberately not a silent alias of S4."
        )

    if strategy == "s0":
        pass
    
    if strategy in ["s1", "s4", "s5"]:
        accepted_shape = shape[accept_mask]
        accepted_scale = scale[accept_mask]
        if len(accepted_shape) > 0:
            med_shape = accepted_shape.median(dim=0).values
            med_scale = accepted_scale.median(dim=0).values
            shape = med_shape.unsqueeze(0).expand(T, -1).clone()
            scale = med_scale.unsqueeze(0).expand(T, -1).clone()
            
    if strategy in ["s2", "s4", "s5"]:
        # S4/S5 pre-smooth only lightly (the fitter does the real work); S2 smooths for keeps.
        beta = args.beta if args.beta >= 0 else (0.005 if strategy in ["s4", "s5"] else 0.05)
        f_min = args.f_min
        print(f"One-Euro: f_min={f_min}, beta={beta}")

        pose_cont = one_euro_filter_track(pose_cont, min_cutoff=f_min, beta=beta)
        cam_trans = one_euro_filter_track(cam_trans, min_cutoff=f_min, beta=beta)
        pred_cam_t = one_euro_filter_track(pred_cam_t, min_cutoff=f_min, beta=beta)
        face = one_euro_filter_track(face, min_cutoff=f_min, beta=beta)
        cam_rot = slerp_smooth_rotations(cam_rot, min_cutoff=f_min, beta=beta)
        # mhr_model_params is what render_motion_frame actually consumes, so it must be
        # smoothed too — filtering pred_pose_raw alone would leave the render untouched.
        # [:6] is identically zero after factorization; filtering it is a no-op.
        if mhr_params is not None:
            mhr_params = one_euro_filter_track(mhr_params, min_cutoff=f_min, beta=beta)

    if strategy in ["s3", "s4", "s5"]:
        # Build fit inputs through the shared helper so kp2d_target is the crop-space
        # reprojection of pred_keypoints_3d (K_proc space), not the raw-image kp2d.
        fit_entries = []
        for i, k in enumerate(db_keys):
            entry = dict(sam3db_gated[k])
            entry["img_key"] = k
            entry["pred_pose_raw"] = pose_cont[i]
            entry["shape"] = shape[i]
            entry["scale"] = scale[i]
            entry["face"] = face[i]
            entry["cam_rot"] = cam_rot[i]
            entry["cam_trans"] = cam_trans[i]
            entry["pred_cam_t"] = pred_cam_t[i]
            if mhr_params is not None:
                entry["mhr_model_params"] = mhr_params[i]
            fit_entries.append(entry)

        gpus = parse_gpus(args.gpus)
        gc.collect()
        torch.cuda.empty_cache()
        model, estimator, hr_image_size, device = init_model_and_estimator(args, gpus[0] if gpus else None)
        unwrapped_model = get_unwrapped_model(model)

        inputs = prepare_pose_fit_inputs(fit_entries, device)
        w_t = w_t.to(device)
        accept_mask = accept_mask.to(device)

        # head_pose is the full-body MHR head. head_pose_hand is the hand-refinement head:
        # it zeroes every non-hand parameter and every non-right-hand keypoint, so fitting
        # against it produces a hand-only mesh.
        result = fit_motion_windows(
            mhr_model=unwrapped_model.head_pose,
            inputs=inputs,
            frame_weights=w_t,
            frame_accept_mask=accept_mask,
            window_size=256,
            overlap=32,
            n_iters=args.fit_iters,
        )

        for i, k in enumerate(db_keys):
            b = result["per_view_body"][k]
            c = result["per_view"][k]
            if mhr_params is not None:
                mhr_params[i] = b["mhr_model_params"]
            pose_cont[i] = b["pred_pose_raw"]
            shape[i] = b["shape"]
            scale[i] = b["scale"]
            face[i] = b["face"]
            cam_trans[i] = c["cam_trans"]
            pred_cam_t[i] = c["pred_cam_t"]
            cam_rot[i] = c["cam_rot"]
            if "hand" in b:
                sam3db_gated[k]["hand"] = b["hand"]
                
    for i, k in enumerate(db_keys):
        if mhr_params is not None:
            sam3db_gated[k]["mhr_model_params"] = mhr_params[i]
        sam3db_gated[k]["pred_pose_raw"] = pose_cont[i]
        sam3db_gated[k]["shape"] = shape[i]
        sam3db_gated[k]["scale"] = scale[i]
        sam3db_gated[k]["cam_trans"] = cam_trans[i]
        sam3db_gated[k]["pred_cam_t"] = pred_cam_t[i]
        sam3db_gated[k]["face"] = face[i]
        sam3db_gated[k]["cam_rot"] = cam_rot[i]
        
    out_path = expr_dir / f"sam3db_{strategy}{args.out_suffix}.pth"
    torch.save(sam3db_gated, out_path)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    args = get_args()
    gpus = parse_gpus(args.gpus)
    if args.stage in ["0a", "all"]:
        stage_0a_track(args, gpus)
    if args.stage in ["0b", "all"]:
        stage_0b_confidence(args)
    if args.stage in ["0c", "all"]:
        stage_0c_gated_track(args)
    if args.stage in ["0d", "all"]:
        stage_0d_eval_frames(args)
    if args.stage in ["0e", "all"]:
        stage_0e_avatar(args, gpus)
    if args.stage in ["strategy", "all"]:
        run_strategy(args)
