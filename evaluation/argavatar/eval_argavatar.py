#!/usr/bin/env python3
"""Evaluate ARGNetBase on the validation set, reporting per-dataset PSNR and LPIPS.

Usage (single GPU):
python evaluation/argavatar/eval_argavatar.py \
    --config configs/ARGAvatar-Final.yaml \
    --checkpoint_path checkpoints/ARGAvatar-Final.pth \
    --data_config configs/_shared/data_val_list.yaml \
    --eval_views 6 \
    --output_folder "expr/visualize_model/argavatar"

Usage (multi-GPU with DDP):
    accelerate launch --multi_gpu --num_processes 6 evaluation/argavatar/eval_argavatar.py \
        --config configs/ARGAvatar-Final.yaml \
        --checkpoint_path checkpoints/ARGAvatar-Final.pth

The --data_config flag (default: configs/_shared/data_val_list.yaml) controls which
validation datasets are evaluated.  The YAML must have a top-level 'data.val' section.

All eval outputs (metrics JSON, dataset_config.yaml, optional vis/) are written under
``expr/results/<project_name>/<ckpt_stem>[_n<num_input_views>]/`` so every checkpoint's
results are in one place and different view-count runs don't collide.

Timing and peak GPU memory are written into the per-dataset eval_results.json as a
``timing`` block alongside the metric values.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf
import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from xlib.utils import load_config, construct_class_by_name
from xlib.train.train_helper import (
    setup_accelerator,
    setup_model_sam3d,
    get_unwrapped_model,
    resolve_sam3d_cfg,
    render_posed_gs,
    build_gs_camera,
    prepare_all_pose_outputs,
)
from xlib.basetype import TensorDict
from xlib.dataset.collate import collate_with_pil_images

from xlib.train.metrics import (
    compute_psnr,
    compute_ssim,
    compute_lpips_masked,
)
from xlib.render.gaussian_splatting import set_gaussian_type
from xlib.model.mhr_animatable_gaussians import animate_gs
from xlib.model_input_utils import slice_batch
from xlib.dataset.common import binary_mask_from_pil
from xlib.utils import AsyncExecutor

from evaluation.common.caching import save_results_json, load_cached_dataset_results, save_sample_result
from evaluation.common.results import (
    print_eval_table,
    chw_tensor_to_uint8,
    mask_tensor_to_uint8,
    save_rgb,
    save_rgba,
    encoded_artifact_name,
)
from evaluation.common.timing import StageTimer


def _snapshot_render_entries(render_entries):
    """Convert render entries' GPU tensors to numpy — call synchronously before async dispatch."""
    return [
        {
            "render_np": chw_tensor_to_uint8(e["render_result"]["body"]["image"][0]),
            "render_mask_np": (
                mask_tensor_to_uint8(e["render_result"]["body"]["mask"][0])
                if e["render_result"]["body"].get("mask") is not None else None
            ),
            "gt_np": chw_tensor_to_uint8(e["gt_image"]),
            "image_path": e.get("image_path"),
        }
        for e in render_entries
    ]


def _write_vis_snapshots(vis_dir, sample_name, snapshots):
    """Write pre-converted numpy snapshots to disk (runs in background thread)."""
    sample_dir = os.path.join(vis_dir, sample_name)
    render_subdir = os.path.join(sample_dir, "render")
    gt_subdir = os.path.join(sample_dir, "render_human_input")
    os.makedirs(render_subdir, exist_ok=True)
    os.makedirs(gt_subdir, exist_ok=True)
    for snap in snapshots:
        fname = encoded_artifact_name(snap["image_path"], suffix=".png")
        if snap["render_mask_np"] is not None:
            save_rgba(os.path.join(render_subdir, fname), snap["render_np"], snap["render_mask_np"])
        else:
            save_rgb(os.path.join(render_subdir, fname), snap["render_np"])
        save_rgb(os.path.join(gt_subdir, fname), snap["gt_np"])


def results_keys_in_eval_order(eval_order, all_results):
    return [k for k in eval_order if k in all_results]


def _get_val_ds_keys(train_args):
    """Return the ordered list of dataset keys from the val config without building datasets."""
    val_cfg = train_args.data.val
    if hasattr(val_cfg, "keys") and "ds_list" in val_cfg and "ds_keys" in val_cfg:
        return list(val_cfg.ds_keys)
    return [getattr(val_cfg, "name", "val")]


def build_val_dataloaders(train_args, num_workers=2, pin_memory=False, ds_overrides=None):
    val_cfg = train_args.data.val

    if hasattr(val_cfg, "keys") and "ds_list" in val_cfg and "ds_keys" in val_cfg:
        ds_keys = list(val_cfg.ds_keys)
        ds_cfgs = list(val_cfg.ds_list)
    else:
        ds_keys = [getattr(val_cfg, "name", "val")]
        ds_cfgs = [val_cfg]

    dataset_cfg = getattr(train_args, "dataset", None)
    hr_image_size = list(dataset_cfg.hr_image_size) if dataset_cfg is not None and hasattr(dataset_cfg, "hr_image_size") else None

    val_loaders = []
    for ds_key, ds_cfg in zip(ds_keys, ds_cfgs):
        ds_cfg = dict(ds_cfg)
        if hr_image_size is not None and "hr_image_size" not in ds_cfg:
            ds_cfg["hr_image_size"] = hr_image_size
        if ds_overrides:
            ds_cfg.update(ds_overrides)
        ds = construct_class_by_name(**ds_cfg)
        val_dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_with_pil_images,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        val_loaders.append((ds_key, val_dl))
    return val_loaders


def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(i) for i in obj]
    return obj


def render_in_raw_space(model, batch, gs_tpose_all, offset_pose_all, device, max_render_height=1024):
    """Render all views in original (raw) image space using K_raw intrinsics."""
    unwrapped_model = get_unwrapped_model(model)
    animatable_gs = unwrapped_model.mhr_animatable_gs
    bg_color = batch["bg_color"].float().to(device)
    B, V = batch["images"].shape[:2]

    input_pose_dic = unwrapped_model.get_input_pose_dic(batch)
    factorized_pose_dic = unwrapped_model._ensure_factorized_input_pose_dic(dict(input_pose_dic))
    all_pose_outputs = prepare_all_pose_outputs(batch, factorized_pose_dic)

    raw_images = batch.get("raw_images")
    raw_masks = batch.get("raw_masks")
    image_paths = batch.get("image_paths")

    # G1: One-shot CPU copy of per-view metadata avoids V×2 per-view GPU→CPU scalar syncs.
    raw_sizes_lst = batch["raw_sizes"].cpu().tolist()
    bg_color_cpu = bg_color.cpu()

    renders = []
    for b_idx in range(B):
        gs_tpose_b = slice_batch(gs_tpose_all, b_idx)
        _sample_v = next((v for v in gs_tpose_b.values() if isinstance(v, torch.Tensor)), None)
        if _sample_v is not None and _sample_v.device.type == "cpu":
            gs_tpose_b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in gs_tpose_b.items()}

        # G5: Compute LBS cache for all views in one batched call instead of V per-view calls.
        pose_dics_b = [slice_batch(all_pose_outputs[v_idx]["pose_dic"], b_idx) for v_idx in range(V)]
        lbs_cache_b = animatable_gs.precompute_lbs_cache(pose_dics_b)

        for v_idx in range(V):
            # G1: Read from pre-copied CPU list; no per-view .item() GPU→CPU syncs.
            H_raw = int(raw_sizes_lst[b_idx][v_idx][0])
            W_raw = int(raw_sizes_lst[b_idx][v_idx][1])
            K_raw_v = batch["K_raw"][b_idx, v_idx].clone()

            scale = min(max_render_height / H_raw, 1.0) if max_render_height > 0 else 1.0
            H_v = int(round(H_raw * scale))
            W_v = int(round(W_raw * scale))
            if scale < 1.0:
                K_raw_v = K_raw_v.clone()
                K_raw_v[0] *= scale
                K_raw_v[1] *= scale

            pred_cam_v = slice_batch(all_pose_outputs[v_idx]["pred_camera"], b_idx)
            cam_rot = pred_cam_v.get("cam_rot")
            cam_trans = pred_cam_v.get("cam_trans")
            cam_v, _ = build_gs_camera(
                pred_cam_v["pred_cam_t"][0],
                K_raw_v, H_v, W_v,
                cam_rot=cam_rot[0] if cam_rot is not None else None,
                model_trans=cam_trans[0] if cam_trans is not None else None,
            )

            # G5: Use per-view slice from the batched LBS cache.
            offset_pose = offset_pose_all[v_idx:v_idx + 1] if offset_pose_all is not None else None
            if offset_pose is not None and offset_pose.device.type == "cpu":
                offset_pose = offset_pose.to(device)
            gs_posed, _ = animate_gs(
                animatable_gs, gs_tpose_b, None,
                lbs_cache=lbs_cache_b, pose_indices=[v_idx],
                to_colmap=True, offset_pose=offset_pose,
            )
            render_result = render_posed_gs(
                gs_posed, b_idx=0, cam=cam_v,
                bg_color=bg_color[b_idx, v_idx],
            )
            del gs_posed
            # G1: Keep render on GPU — drop the .cpu() round-trip.
            rendered_img_gpu = render_result["body"]["image"]
            rendered_mask_gpu = render_result["body"].get("mask")
            del render_result

            if raw_images is not None and raw_masks is not None:
                pil_img = raw_images[b_idx][v_idx]
                pil_mask = raw_masks[b_idx][v_idx]
                # CPU decode is unavoidable; upload once and do all remaining ops on GPU.
                arr = np.ascontiguousarray(np.array(pil_img.convert("RGB")))
                mask_np = binary_mask_from_pil(pil_mask)
                gt_t = torch.from_numpy(arr).to(device, non_blocking=True).permute(2, 0, 1).float().div_(255.0)
                mask_t_2d = torch.from_numpy(np.ascontiguousarray(mask_np[..., 0])).to(device, non_blocking=True).float().div_(255.0)
                if scale < 1.0:
                    gt_t = F.interpolate(gt_t.unsqueeze(0), size=(H_v, W_v), mode="bilinear", align_corners=False)[0]
                    mask_t_2d = F.interpolate(mask_t_2d.unsqueeze(0).unsqueeze(0), size=(H_v, W_v), mode="bilinear", align_corners=False)[0, 0]
                bg_v = bg_color[b_idx, v_idx].view(3, 1, 1)
                raw_img_t = gt_t * mask_t_2d.unsqueeze(0) + (1.0 - mask_t_2d.unsqueeze(0)) * bg_v
                raw_mask_t = mask_t_2d.unsqueeze(0)
            else:
                raw_img_t = batch["images"][b_idx, v_idx]
                raw_mask_t = batch["masks"][b_idx, v_idx]

            img_path = image_paths[v_idx][b_idx]
            renders.append({
                "render_result": {"body": {"image": rendered_img_gpu, "mask": rendered_mask_gpu}},
                "gt_image": raw_img_t,
                "gt_mask": raw_mask_t,
                "bg_color": bg_color_cpu[b_idx, v_idx].reshape(-1),
                "image_path": img_path,
            })

        del lbs_cache_b

    return renders


def _results_dir_for_nviews(args, ckpt_step_base: str, n_views, views_list) -> str:
    """Compute results directory for a given n_views value."""
    suffix = f"_n{n_views}" if n_views is not None else ""
    if args.output_folder:
        if len(views_list) > 1:
            return args.output_folder + suffix
        return args.output_folder
    return os.path.join("expr", "results", args._project_name, ckpt_step_base + suffix)


def _eval_all_views(
    args,
    views_list,
    ckpt_step_base: str,
    eval_order,
    val_loaders,
    model,
    lpips_model,
    accelerator,
    device,
):
    results_dirs = {n: _results_dir_for_nviews(args, ckpt_step_base, n, views_list) for n in views_list}
    output_jsons = {n: os.path.join(d, "eval_results.json") for n, d in results_dirs.items()}

    for ds_key, val_dl in val_loaders:
        cached = {n: load_cached_dataset_results(output_jsons[n]) for n in views_list}
        pending_views = [n for n in views_list if ds_key not in cached[n]]

        if not pending_views:
            if accelerator.is_main_process:
                print(f"Skipping {ds_key} (all view counts cached)")
            continue

        if accelerator.is_main_process:
            for n in pending_views:
                os.makedirs(results_dirs[n], exist_ok=True)
        for n in pending_views:
            os.makedirs(os.path.join(results_dirs[n], "vis", ds_key), exist_ok=True)

        val_dl = accelerator.prepare(val_dl)

        sums = {
            n: {k: torch.tensor(0.0, device=device) for k in ("psnr_input", "psnr_cross", "lpips_input", "lpips_cross", "ssim_input", "ssim_cross")}
            for n in pending_views
        }
        counts = {
            n: {"input": torch.tensor(0.0, device=device), "cross": torch.tensor(0.0, device=device)}
            for n in pending_views
        }
        timers = {n: StageTimer(sync_cuda=getattr(args, "sync_cuda", False)) for n in pending_views}
        vis_executor = AsyncExecutor()

        n_samples = args.n_samples
        total = len(val_dl) if n_samples <= 0 else math.ceil(n_samples / max(accelerator.num_processes, 1))

        pbar = tqdm(
            val_dl,
            total=total,
            desc=f"Evaluating {ds_key}",
            disable=not accelerator.is_main_process,
        )
        count = 0

        for batch_raw in pbar:
            if isinstance(batch_raw, list) and len(batch_raw) == 2:
                _, batch_raw = batch_raw
            raw_id = batch_raw.get("sample_identifier", [None])[0]
            sample_name = (
                str(raw_id).replace("/", "_").replace(os.sep, "_")
                if raw_id is not None
                else f"sample{count:04d}"
            )

            # Preprocess batch once; per-view input_view_mask is overridden inside the views loop.
            batch = TensorDict.from_dict(batch_raw).to(device)
            batch["bg_color"] = torch.ones_like(batch["bg_color"])
            batch["images"] = batch["images"] * batch["masks"] + (1 - batch["masks"]) * batch["bg_color"].view(1, -1, 3, 1, 1)
            if "hr_images" in batch:
                B, V, _, H, W = batch["masks"].shape
                H_r, W_r = batch["hr_images"].shape[-2:]
                sr_mask = (
                    F.interpolate(
                        batch["masks"].view(B * V, 1, H, W),
                        size=(H_r, W_r),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .view(B, V, 1, H_r, W_r)
                )
                batch["hr_images"] = batch["hr_images"] * sr_mask + (1 - sr_mask) * batch["bg_color"].view(1, -1, 3, 1, 1)

            _ivm_orig = batch.get("input_view_mask")

            for n_views in pending_views:
                if _ivm_orig is not None:
                    V_total = _ivm_orig.shape[-1]
                    n_input = V_total if n_views < 0 else min(n_views, V_total)
                    _ivm_new = torch.zeros(V_total, dtype=torch.bool, device=_ivm_orig.device)
                    _ivm_new[:n_input] = True
                    batch["input_view_mask"] = _ivm_new.unsqueeze(0).expand_as(_ivm_orig) if _ivm_orig.ndim == 2 else _ivm_new

                timers[n_views].begin_sample()

                with timers[n_views].stage("reconstruction"):
                    with accelerator.autocast():
                        result = get_unwrapped_model(model).two_phase_model_forward(batch)

                _pass = result["passes"][-1]
                _gs_tpose = _pass["output_canonical_gs_dic"]
                _offset_pose = _pass.get("offset_pose")
                del result, _pass
                torch.cuda.empty_cache()

                with timers[n_views].stage("render"):
                    render_entries = render_in_raw_space(model, batch, _gs_tpose, _offset_pose, device, args.max_render_height)

                V = batch["images"].shape[1]
                _ivm = batch.get("input_view_mask")
                _ivm_1d = (_ivm[0] if _ivm.ndim == 2 else _ivm).bool() if _ivm is not None else None

                with timers[n_views].stage("metrics"):
                    for i, entry in enumerate(render_entries):
                        pred = entry["render_result"]["body"]["image"][0].to(device)
                        gt = entry["gt_image"].to(device)
                        mask = entry["gt_mask"].to(device)
                        bg = entry["bg_color"].to(device)
                        psnr_val = compute_psnr(pred, gt, mask)
                        ssim_val = compute_ssim(pred, gt, mask)
                        lpips_val = compute_lpips_masked(pred, gt, mask, bg, lpips_model)

                        v_idx = i % V
                        is_input = bool(_ivm_1d[v_idx].item()) if _ivm_1d is not None else (v_idx == 0)
                        if is_input:
                            sums[n_views]["psnr_input"] += psnr_val
                            sums[n_views]["lpips_input"] += lpips_val
                            sums[n_views]["ssim_input"] += ssim_val
                            counts[n_views]["input"] += 1.0
                        else:
                            sums[n_views]["psnr_cross"] += psnr_val
                            sums[n_views]["lpips_cross"] += lpips_val
                            sums[n_views]["ssim_cross"] += ssim_val
                            counts[n_views]["cross"] += 1.0

                sample_timing = timers[n_views].end_sample()

                if accelerator.is_main_process:
                    npy_path = os.path.join(results_dirs[n_views], "sample_results.npy")
                    save_sample_result(npy_path, sample_name, {"timing": sample_timing})
                global_idx = accelerator.process_index + count * accelerator.num_processes
                do_vis = (args.n_vis < 0) or (global_idx < args.n_vis)
                if do_vis:
                    vis_dir = os.path.join(results_dirs[n_views], "vis", ds_key)
                    snapshots = _snapshot_render_entries(render_entries)
                    vis_executor(_write_vis_snapshots, vis_dir, sample_name, snapshots)

                del render_entries, _gs_tpose, _offset_pose
                torch.cuda.empty_cache()

            del batch
            count += 1
            if count >= total:
                break

        vis_executor.clean_sync()

        for n_views in pending_views:
            gathered = {k: accelerator.gather(sums[n_views][k]) for k in sums[n_views]}
            gin = accelerator.gather(counts[n_views]["input"])
            gcross = accelerator.gather(counts[n_views]["cross"])
            n_in = int(gin.sum().item())
            n_cross = int(gcross.sum().item())

            ds_result = {
                "psnr_input": (gathered["psnr_input"].sum() / gin.sum().clamp(min=1)).item(),
                "psnr_cross": (gathered["psnr_cross"].sum() / gcross.sum().clamp(min=1)).item(),
                "lpips_input": (gathered["lpips_input"].sum() / gin.sum().clamp(min=1)).item(),
                "lpips_cross": (gathered["lpips_cross"].sum() / gcross.sum().clamp(min=1)).item(),
                "ssim_input": (gathered["ssim_input"].sum() / gin.sum().clamp(min=1)).item(),
                "ssim_cross": (gathered["ssim_cross"].sum() / gcross.sum().clamp(min=1)).item(),
                "n_input": n_in,
                "n_cross": n_cross,
                "timing": timers[n_views].to_dict(),
            }

            if accelerator.is_main_process:
                existing = load_cached_dataset_results(output_jsons[n_views])
                existing[ds_key] = ds_result
                save_results_json(output_jsons[n_views], existing, results_keys_in_eval_order(eval_order, existing))
                if ds_result["psnr_cross"] < 25.0 or ds_result["psnr_input"] < 25.0:
                    print(
                        f"WARNING: {ds_key} [n={n_views}] PSNR below 25 dB "
                        f"(input={ds_result['psnr_input']:.2f}, cross={ds_result['psnr_cross']:.2f})"
                    )
                print(f"[n={n_views}] Finished {ds_key}. Saved to: {output_jsons[n_views]}")

    if accelerator.is_main_process:
        for n_views in views_list:
            payload = load_cached_dataset_results(output_jsons[n_views])
            if payload:
                print(f"\n=== n_input_views={n_views} ===")
                print_eval_table({k: payload[k] for k in eval_order if k in payload})


def main(args):
    torch.set_grad_enabled(False)

    train_args = load_config(args.config)
    cfg = resolve_sam3d_cfg(train_args, args.config)
    set_gaussian_type(train_args.model.get("gaussian_type", "3d"))

    if args.checkpoint_path:
        train_args.model["load_checkpoint"] = args.checkpoint_path

    data_cfg = OmegaConf.load(args.data_config)
    OmegaConf.resolve(data_cfg)
    train_args.data = data_cfg.data

    expr_dir = os.path.join(
        getattr(train_args.training, "output_dir", "expr"), train_args.project_name
    )
    ckpt_step_base = os.path.splitext(os.path.basename(args.checkpoint_path))[0] if args.checkpoint_path else "no_ckpt"
    args._project_name = train_args.project_name

    views_list = args.eval_views

    eval_order = _get_val_ds_keys(train_args)

    accelerator = setup_accelerator(train_args, expr_dir)
    device = accelerator.device

    seed = int(train_args.training.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # When any requested view count exceeds 6, load all supervision views from the dataset.
    any_dense = any(v is not None and v > 6 for v in views_list)
    ds_overrides = {"num_sup_views": -1} if any_dense else None
    val_loaders = build_val_dataloaders(
        train_args,
        num_workers=args.num_workers,
        ds_overrides=ds_overrides,
    )

    # Load model once, evaluate all view counts in a single pass per dataset.
    model = setup_model_sam3d(train_args, cfg, device)
    model.eval()
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.requires_grad_(False)
    model = accelerator.prepare(model)

    _eval_all_views(
        args, views_list, ckpt_step_base,
        eval_order, val_loaders, model, lpips_model, accelerator, device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ARGNetBase on validation set")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint_path", default="", help="Path to model checkpoint")
    parser.add_argument(
        "--data_config",
        default="configs/_shared/data_val_list.yaml",
        help="Path to data config YAML (must have a top-level 'data.val' section); "
             "defaults to configs/_shared/data_val_list.yaml",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="Max samples per dataset across all processes (-1 = full dataset)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples that already have eval_results.json",
    )
    parser.add_argument(
        "--n_vis",
        type=int,
        default=6,
        help="Save visualizations for only the first N samples per dataset (-1 = all)",
    )
    parser.add_argument(
        "--max_render_height",
        type=int,
        default=1024,
        help="Use original image size when height is below this value; otherwise scale "
             "height to this value with aspect ratio fixed. Set to 0 for original resolution.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker processes per evaluation rank.",
    )
    parser.add_argument(
        "--eval_views",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6],
        help="Input view counts to evaluate. Each count writes to a separate '_n<N>' subdirectory. "
             "Values > 6 automatically enable dense mode (all supervision views loaded).",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        help="Override the default results directory (expr/results/<model>/<ckpt>/).",
    )
    parser.add_argument("--sync_cuda", action="store_true", help="Synchronize CUDA before/after each stage for accurate per-stage timing.")
    args = parser.parse_args()
    main(args)
