import sys
import logging
import os
import random
import re
import numpy as np

import torch
import torch.nn as nn
from typing import cast, Optional, List, Dict, Any, Tuple
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, GradScalerKwargs, InitProcessGroupKwargs
import torch.nn.functional as F
from PIL import Image
from diffusers.utils.export_utils import export_to_video

sys.path.append('.')
from xlib.dataset.collate import collate_with_pil_images
from xlib.model.mhr_animatable_gaussians import animate_gs
from xlib.render.gaussian_splatting_3d import index_gs
import lpips
from xlib.utils import DictConfig, construct_class_by_name
from xlib.render.camera import GSCamera
from xlib.render.gaussian_splatting import splat_gaussians
from xlib import model_input_utils
from xlib.render.visualizer import _resolve_view0_offset_pose



def estimate_rigid_transform(src_xyz: torch.Tensor, dst_xyz: torch.Tensor) -> tuple:
    """Solve dst ~= src @ R^T + t for row-vector point clouds via SVD."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        src_xyz = src_xyz.float()
        dst_xyz = dst_xyz.float()
        src_mean = src_xyz.mean(dim=0)
        dst_mean = dst_xyz.mean(dim=0)
        cov = (src_xyz - src_mean).T @ (dst_xyz - dst_mean)
        U, _, Vh = torch.linalg.svd(cov)
        rot = Vh.T @ U.T
        if torch.det(rot) < 0:
            Vh[-1] *= -1
            rot = Vh.T @ U.T
        trans = dst_mean - (rot @ src_mean)
    return rot, trans


def factorize_root_rotation(
    animatable_gs,
    model_params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract root rigid transform from model_params and return zeroed params.

    Args:
        animatable_gs: MHRAnimatableGaussians instance.
        model_params: [B, 1, D] or [B, D] model parameters.

    Returns:
        zero_model_params: model_params with root rotation zeroed (same shape as input).
        rigid_rot: [B, 3, 3] rotation matrix in colmap space.
        rigid_trans: [B, 3] translation vector in colmap space.
    """
    zero_model_params = model_params.clone()
    zero_model_params[..., :6] = 0
    mp_b = model_params.reshape(-1, model_params.shape[-1])  # [B, D]
    rigid_rot_b, rigid_trans_b = animatable_gs.get_root_rigid_in_colmap(mp_b)
    return zero_model_params, rigid_rot_b, rigid_trans_b


def build_gs_camera(
    pred_cam_t: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    K_raw: Optional[torch.Tensor] = None,
    raw_size: Optional[torch.Tensor] = None,
    cam_rot: Optional[torch.Tensor] = None,
    model_trans: Optional[torch.Tensor] = None,
) -> Tuple[GSCamera, torch.Tensor]:
    """Build a GSCamera and world2cam matrix with optional root-factored rotation.

    Returns:
        cam: GSCamera ready for splatting.
        world2cam: [4, 4] matrix (needed for face camera construction).
    """
    device = pred_cam_t.device
    world2cam = torch.eye(4, device=device, dtype=torch.float32)
    world2cam[:3, :3] = cam_rot.float()
    world2cam[:3, 3] = pred_cam_t.float()
    if model_trans is not None:
        world2cam[:3, 3] += model_trans.float()        

    full_image_width, full_image_height = None, None
    if raw_size is not None:
        full_image_height = int(raw_size[0].item())
        full_image_width = int(raw_size[1].item())

    cam = GSCamera.from_extrinsics_intrinsics(
        world2cam, K=K,
        full_K=K_raw if K_raw is not None else None,
        full_image_width=full_image_width,
        full_image_height=full_image_height,
        image_width=W, image_height=H,
    )
    return cam, world2cam


def build_pose_camera(
    animatable_gs,
    model_params: torch.Tensor,
    pred_cam_t: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    K_raw: Optional[torch.Tensor] = None,
    raw_size: Optional[torch.Tensor] = None,
    cam_rot: Optional[torch.Tensor] = None,
    model_trans: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, GSCamera, torch.Tensor]:
    """Factorize root rotation from model_params and build camera in one step.

    Args:
        animatable_gs: MHRAnimatableGaussians instance.
        model_params: [1, 1, D] or [1, D] model parameters.
        pred_cam_t: [3] camera translation.
        K: [3, 3] intrinsics.
        H, W: Render dimensions.
        K_raw: [3, 3] raw intrinsics, optional.
        raw_size: [2] raw image size, optional.
        cam_rot: Optional [3, 3] pre-factorized root rotation.
        model_trans: Optional [3] pre-factorized root translation.

    Returns:
        zero_model_params: model_params with root rotation zeroed.
        cam: GSCamera with root rotation baked in.
        world2cam: [4, 4] matrix (for face camera construction).
    """
    if cam_rot is not None and model_trans is not None:
        zero_model_params = model_params
        rigid_rot = cam_rot
        rigid_trans = model_trans
    else:
        zero_model_params, rigid_rot, rigid_trans = factorize_root_rotation(
            animatable_gs, model_params)
        if rigid_rot.ndim == 3:
            rigid_rot = rigid_rot[0]
        if rigid_trans.ndim == 2:
            rigid_trans = rigid_trans[0]
    cam, world2cam = build_gs_camera(
        pred_cam_t, K, H, W,
        K_raw=K_raw, raw_size=raw_size,
        cam_rot=rigid_rot, model_trans=rigid_trans,
    )
    return zero_model_params, cam, world2cam


def render_posed_gs(
    gs_posed,
    b_idx: int,
    cam: GSCamera,
    bg_color: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Render already-posed GaussianParameters with a pre-built camera.

    Args:
        gs_posed: GaussianParameters [B, 1, L, ...] — already animated/deformed GS.
        b_idx: Batch index to render.
        cam: Pre-built GSCamera for body rendering.
        bg_color: Background color [3].
    """
    gs_view = index_gs(gs_posed, b_idx, 0)
    result = splat_gaussians(gs_view, [cam], bg_color.float())
    return {"body": result}


def store_render_to_buffer(
    entry: Dict,
    p_idx: int,
    b_idx: int,
    v_idx: int,
    all_pass_rendered_image: torch.Tensor,
) -> None:
    """Write rendered image into the visualization buffer."""
    all_pass_rendered_image[p_idx, b_idx, v_idx] = entry["render_result"]["body"]["image"][0].detach().cpu()


def prepare_all_pose_outputs(batch, input_pose_dic):
    total_views = batch["images"].shape[1]
    all_pose_outputs = []

    def _view_tensor(value, v_idx):
        if not isinstance(value, torch.Tensor):
            return value
        if value.ndim >= 2 and value.shape[1] == total_views:
            return value[:, v_idx]
        return value

    for v_idx in range(total_views):
        shape_params = _view_tensor(input_pose_dic["shape"], v_idx)
        model_params = _view_tensor(input_pose_dic["mhr_model_params"], v_idx)
        expr_params = _view_tensor(input_pose_dic["face"], v_idx)
        if model_params.ndim == 2:
            model_params = model_params.unsqueeze(1)
        if expr_params.ndim == 2:
            expr_params = expr_params.unsqueeze(1)

        pose_dic = {
            "shape_params": shape_params,
            "model_params": model_params,
            "expr_params": expr_params,
        }
        pred_camera = {
            "pred_cam_t": _view_tensor(input_pose_dic["pred_cam_t"], v_idx),
        }
        if "cam_rot" in input_pose_dic:
            pred_camera["cam_rot"] = _view_tensor(input_pose_dic["cam_rot"], v_idx)
        if "cam_trans" in input_pose_dic:
            pred_camera["cam_trans"] = _view_tensor(input_pose_dic["cam_trans"], v_idx)
        all_pose_outputs.append({
            "mhr": {k: _view_tensor(val, v_idx) for k, val in input_pose_dic.items() if isinstance(val, torch.Tensor)},
            "pose_dic": pose_dic,
            "pred_camera": pred_camera,
            "pred_cam_t": pred_camera["pred_cam_t"],
        })
    return all_pose_outputs


@torch.no_grad()
def prepare_pose_lbs(
    model,
    batch: Dict,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute target_poses and lbs_cache for all views from a batch.

    Returns poses dict (model/shape/expr_params stacked over B*V) and lbs_cache.
    """
    unwrapped = get_unwrapped_model(model)
    device = batch["images"].device
    input_pose_dic = unwrapped.get_input_pose_dic(batch)
    factorized_pose_dic = unwrapped._ensure_factorized_input_pose_dic(dict(input_pose_dic))
    all_pose_outputs = prepare_all_pose_outputs(batch, factorized_pose_dic)
    all_pose_dics = [view_output["pose_dic"] for view_output in all_pose_outputs]
    all_lbs_cache = unwrapped.mhr_animatable_gs.precompute_lbs_cache(all_pose_dics)

    B, V = batch["images"].shape[:2]
    all_model_params: List[torch.Tensor] = []
    all_shape_params: List[torch.Tensor] = []
    all_expr_params: List[torch.Tensor] = []
    for v_idx in range(V):
        for b_idx in range(B):
            pose_v = model_input_utils.slice_batch(all_pose_outputs[v_idx]["pose_dic"], b_idx)
            all_model_params.append(pose_v["model_params"].reshape(1, -1))
            all_shape_params.append(pose_v["shape_params"].reshape(1, -1))
            all_expr_params.append(pose_v["expr_params"].reshape(1, -1))

    poses = {
        "model_params": torch.cat(all_model_params, dim=0).to(device),
        "shape_params": torch.cat(all_shape_params, dim=0).to(device),
        "expr_params": torch.cat(all_expr_params, dim=0).to(device),
    }
    return poses, all_lbs_cache


def get_visualization_items(
    passes: List,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fields required by ``visualize_nps_step`` extracted from ``passes``.

    Returns:
        uv_encoding_passes: One entry per non-token-GS pass for UV Gaussian-attribute dumps.
        last_pass_viz: ``output_canonical_gs_dic`` and ``view0_offset_pose`` for the final pass
            (``passes[-1]``).
    """
    uv_encoding_passes: List[Dict[str, Any]] = []
    for p_idx, pass_output in enumerate(passes):
        if pass_output.get("is_token_gs"):
            continue
        uv_encoding_passes.append({
            "pass_index": p_idx,
            "output_canonical_gs_dic": pass_output.get("output_canonical_gs_dic"),
            "offset_pose": pass_output.get("offset_pose"),
        })
    last = passes[-1]
    last_pass_viz = {
        "output_canonical_gs_dic": last.get("output_canonical_gs_dic"),
        "view0_offset_pose": _resolve_view0_offset_pose(
            deform_dic=last.get("deform_dic"),
            offset_pose=last.get("offset_pose"),
        ),
    }
    return uv_encoding_passes, last_pass_viz


def build_precomputed_pose_camera_cache(
    unwrapped_model,
    batch: Dict,
    H_r: int,
    W_r: int,
    scale_h: float = 1.0,
    scale_w: float = 1.0,
) -> Dict[Tuple[int, int], Tuple[Dict[str, torch.Tensor], GSCamera, torch.Tensor]]:
    """Build a per-(batch, view) cache of zero-root pose and render camera.

    The source of truth is the normalized input pose dictionary:
    1. populate missing per-view poses via model.get_input_pose_dic()
    2. factorize root rotation via _ensure_factorized_input_pose_dic()
    3. convert to pose_dic/pred_camera pairs via prepare_all_pose_outputs()
    """
    input_pose_dic = unwrapped_model.get_input_pose_dic(batch)
    factorized_pose_dic = unwrapped_model._ensure_factorized_input_pose_dic(dict(input_pose_dic))
    all_pose_outputs = prepare_all_pose_outputs(batch, factorized_pose_dic)

    B, V = batch["images"].shape[:2]
    precomputed = {}
    for b_idx in range(B):
        for v_idx in range(V):
            pose_v = model_input_utils.slice_batch(all_pose_outputs[v_idx]["pose_dic"], b_idx)
            pred_cam_v = model_input_utils.slice_batch(all_pose_outputs[v_idx]["pred_camera"], b_idx)

            K_v = batch["K_proc"][b_idx, v_idx]
            if scale_h != 1.0 or scale_w != 1.0:
                K_v = K_v.clone()
                K_v[0, 0] *= scale_w
                K_v[0, 2] *= scale_w
                K_v[1, 1] *= scale_h
                K_v[1, 2] *= scale_h

            cam_rot_v = pred_cam_v["cam_rot"]
            cam_trans_v = pred_cam_v["cam_trans"]
            cam_v, w2c_v = build_gs_camera(
                pred_cam_v["pred_cam_t"][0],
                K_v, H_r, W_r,
                K_raw=batch["K_raw"][b_idx, v_idx],
                raw_size=batch["raw_sizes"][b_idx, v_idx],
                cam_rot=cam_rot_v[0],
                model_trans=cam_trans_v[0],
            )
            precomputed[(b_idx, v_idx)] = (pose_v, cam_v, w2c_v)
    return precomputed


def get_unwrapped_model(model):
    if hasattr(model, "module"):
        return model.module
    return model

def report_trainable_param_grads(model, step: int, max_list: int = 100) -> None:
    """After backward, report model parameters with ``requires_grad`` that still have ``grad`` None.

    Use with ``training.debug_param_grads=true`` to find trainable tensors disconnected
    from the loss graph.
    """
    unwrapped = get_unwrapped_model(model)
    id_to_name = {id(p): n for n, p in unwrapped.named_parameters()}
    missing = []
    n_trainable = 0
    for p in unwrapped.parameters():
        if not p.requires_grad:
            continue
        n_trainable += 1
        if p.grad is None:
            name = id_to_name.get(id(p), f"<unknown_param id={id(p)}>")
            missing.append(name)
    if missing:
        shown = missing[:max_list]
        extra = "" if len(missing) <= max_list else f" ... and {len(missing) - max_list} more"
        logging.warning(
            "[debug_param_grads] step=%s: %d / %d requires_grad params have grad=None:%s\n  %s",
            step,
            len(missing),
            n_trainable,
            extra,
            "\n  ".join(shown),
        )
    else:
        logging.debug(
            "[debug_param_grads] step=%s: all %d requires_grad params have a grad tensor.",
            step,
            n_trainable,
        )

    debug_tensors = getattr(unwrapped, "_token_gs_debug_tensors", None)
    if debug_tensors:
        for name, tensor in debug_tensors.items():
            grad = tensor.grad
            if grad is None:
                logging.warning(
                    "[token_gs_debug_bwd] step=%s %s grad=None requires_grad=%s grad_fn=%s",
                    step,
                    name,
                    tensor.requires_grad,
                    tensor.grad_fn,
                )
            else:
                logging.warning(
                    "[token_gs_debug_bwd] step=%s %s grad_norm=%.6e grad_absmax=%.6e",
                    step,
                    name,
                    grad.norm().item(),
                    grad.abs().max().item(),
                )


def resolve_sam3d_cfg(train_args, config_path: str = ""):
    """Resolve the SAM3DBody yacs config from a training config object."""
    from xlib.model.sam_3d_body.utils.config import get_config, get_config_from_sections
    if "sam3dbody" in train_args.model:
        return get_config_from_sections(train_args.model, getattr(train_args, "training", None))
    sam3d_config_path = train_args.model.get("sam3d_config", config_path)
    return get_config(sam3d_config_path)


def _get_input_image_paths_for_sample(batch: Dict, b_idx: int) -> Optional[List[str]]:
    """Return list of V input image paths for batch index b_idx, or None if not in batch."""
    if "image_paths" not in batch:
        return None
    image_paths = batch["image_paths"]
    B, V = batch["images"].shape[0], batch["images"].shape[1]
    paths = []
    for v_idx in range(V):
        if len(image_paths) == B and len(image_paths[b_idx]) > v_idx:
            paths.append(image_paths[b_idx][v_idx])
        elif len(image_paths) == V and len(image_paths[v_idx]) > b_idx:
            paths.append(image_paths[v_idx][b_idx])
        else:
            return None
    return paths


# ==================== SAM3D Model Setup Functions ====================


def setup_accelerator(args, expr_dir: str) -> Accelerator:
    project_config = ProjectConfiguration(
        project_dir=expr_dir,
        logging_dir=os.path.join(expr_dir, "logs"),
    )
    # static_graph supports DDP when the same parameters are used in multiple
    # checkpointed forward passes per iteration (e.g. flow matching loops the
    # shared pixel_dit over every supervision view, then does one backward),
    # which otherwise trips DDP's "mark a variable ready only once" check.
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=bool(getattr(args.training, "find_unused_parameters", False)),
        static_graph=bool(getattr(args.training, "ddp_static_graph", False)),
    )

    # Use lower GradScaler init_scale to prevent gradient overflow in rasterizer backward
    # Default is 65536 which can cause overflow when gradients go through CUDA rasterizer
    kwargs_handlers = [ddp_kwargs]
    accelerator = Accelerator(
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        mixed_precision=args.training.mixed_precision,
        log_with="wandb" if args.training.use_wandb else None,
        project_config=project_config,
        kwargs_handlers=kwargs_handlers,
    )

    if accelerator.is_main_process:
        os.makedirs(expr_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(expr_dir, "train.log")),
                logging.StreamHandler(),
            ],
        )

    return accelerator


def setup_data_sam3d(args) -> Tuple[DataLoader, DataLoader]:
    """Setup data loaders for SAM3D training using construct_class_by_name for ConcatDataset format."""

    # Use construct_class_by_name to create datasets from config
    train_ds = construct_class_by_name(**args.data.train)
    val_ds = construct_class_by_name(**args.data.val)

    seed = int(args.training.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    val_num_workers = args.training["val_num_workers"] if "val_num_workers" in args.training else args.training.num_workers

    train_dl = DataLoader(
        train_ds,
        batch_size=args.training.batch_size,
        shuffle=args.training.get("shuffle", True),
        num_workers=args.training.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_with_pil_images,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True,
        persistent_workers=val_num_workers > 0,
        collate_fn=collate_with_pil_images,
        worker_init_fn=_seed_worker,
    )

    return train_dl, val_dl


def override_dataset_views(args):
    """Normalize dataset view-count overrides onto `args.dataset`.

    Applies `args.dataset` recursively to every dataset config under
    `args.data.train` and `args.data.val`. This keeps view-count ownership in
    the main config instead of duplicating it inside shared data YAMLs.
    """
    def _apply_dataset_overrides(dataset_cfg, overrides, split=None):
        if not overrides:
            return dataset_cfg

        if hasattr(dataset_cfg, "keys") and "ds_list" in dataset_cfg:
            for ds_cfg in dataset_cfg["ds_list"]:
                _apply_dataset_overrides(ds_cfg, overrides, split=split)
            return dataset_cfg

        if hasattr(dataset_cfg, "__setitem__"):
            for key, value in overrides.items():
                dataset_cfg[key] = value
            if split == "val":
                if "num_input_views" in args.training:
                    dataset_cfg["num_input_views"] = args.training.num_input_views
                elif "num_input_views" in overrides:
                    dataset_cfg["num_input_views"] = overrides["num_input_views"]
                if "n_val_sup_views" not in args.training:
                    raise KeyError("training.n_val_sup_views must be set in the model config for validation-list datasets")
                dataset_cfg["num_sup_views"] = args.training["n_val_sup_views"]
        return dataset_cfg

    dataset_cfg = args.get("dataset", None) if hasattr(args, "get") else None
    if dataset_cfg is None:
        args["dataset"] = OmegaConf.create({})
        dataset_cfg = args.dataset

    if "num_input_views" not in dataset_cfg and "num_input_views" in args.training:
        dataset_cfg["num_input_views"] = args.training.num_input_views
    if "num_sup_views" not in dataset_cfg and "num_sup_views" in args.training:
        dataset_cfg["num_sup_views"] = args.training.num_sup_views

    has_input = "num_input_views" in dataset_cfg
    has_sup = "num_sup_views" in dataset_cfg

    if has_input != has_sup:
        raise ValueError("dataset.num_input_views and dataset.num_sup_views must be set together")

    if len(dataset_cfg) > 0:
        _apply_dataset_overrides(args.data.train, dataset_cfg, split="train")
        _apply_dataset_overrides(args.data.val, dataset_cfg, split="val")

    return args


def setup_model_sam3d(args, cfg, device):
    """Setup model (ARGNetBase, ARGNetBaseUnified, etc.) using construct_class_by_name.

    Args:
        args: Training arguments/config
        cfg: SAM3DBody model config (yacs format)
        device: Device to use

    Returns:
        model: Initialized model
    """

    model = construct_class_by_name(
        class_name=args.model['class_name'],
        cfg=cfg,
        avatar_cfg=args.model.avatar
    )
    model.to(device)

    # Load SAM3DBody pretrained weights (backbone + pose heads) from cfg.CHECKPOINT_PATH
    if hasattr(model, 'load_pretrained_weights'):
        model.load_pretrained_weights()

    encoder_checkpoint = args.model.get('encoder_checkpoint')
    resume_checkpoint = args.model.get('load_checkpoint')
    if encoder_checkpoint and not resume_checkpoint:
        if not hasattr(model, 'load_checkpoint'):
            raise AttributeError(
                f"{type(model).__name__} does not support model.encoder_checkpoint"
            )
        missing, unexpected = model.load_checkpoint(encoder_checkpoint)
        logging.info(
            "Loaded encoder warm-start checkpoint from %s. Missing: %d, Unexpected: %d",
            encoder_checkpoint,
            len(missing),
            len(unexpected),
        )

    # Load additional training checkpoint if specified (for resuming training)
    if resume_checkpoint:
        if hasattr(model, 'load_checkpoint'):
            missing, unexpected = model.load_checkpoint(resume_checkpoint)
        else:
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
        logging.info(
            "Loaded training checkpoint from %s. Missing: %d, Unexpected: %d",
            resume_checkpoint,
            len(missing),
            len(unexpected),
        )

    if hasattr(model, 'setup_avatar_training'):
        model.setup_avatar_training()

    return model


def setup_models_and_data_sam3d(args, cfg, device):
    """Setup models and data for SAM3D training."""
    # Use existing functions
    train_dl, val_dl = setup_data_sam3d(args)
    model = setup_model_sam3d(args, cfg, device)
    
    # Setup LPIPS
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.requires_grad_(False)
    
    return train_dl, val_dl, model, lpips_model


def visualize_nps_train_images(
    batch: Dict,
    all_pass_rendered_image: torch.Tensor,
    save_dir: str,
    step: int,
    process_index: int,
    all_pass_rendered_face: Optional[torch.Tensor] = None,
    all_pass_canonical_rendered_image: Optional[torch.Tensor] = None,
    all_pass_canonical_rendered_face: Optional[torch.Tensor] = None,
    face_gt: Optional[torch.Tensor] = None,
) -> None:
    """Save one grid per batch item with shared GT rows and one rendered row per pass."""
    B, V = batch["images"].shape[:2]
    target_h, target_w = all_pass_rendered_image.shape[-2:]
    gt_images = batch["images"]
    if "hr_images" in batch and batch["hr_images"].shape[-2:] == (target_h, target_w):
        gt_images = batch["hr_images"]
    num_passes = all_pass_rendered_image.shape[0]

    for b in range(B):
        # Save image paths once per batch item (same for all passes)
        input_paths = _get_input_image_paths_for_sample(batch, b)
        if input_paths is not None:
            paths_txt_path = os.path.join(save_dir, f"mpg_step{step}_p{process_index}_b{b}_paths.txt")
            with open(paths_txt_path, "w") as f:
                f.write("\n".join(input_paths))

        imgs = []

        # Row 1: shared GT body images
        for v in range(V):
            gt_img = gt_images[b, v].detach().cpu()
            if gt_img.shape[-2:] != (target_h, target_w):
                gt_img = F.interpolate(
                    gt_img.unsqueeze(0), size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            imgs.append(gt_img)

        # Row 2: GT masks
        gt_masks = batch["masks"]
        for v in range(V):
            gt_mask = gt_masks[b, v].detach().cpu()
            if gt_mask.shape[-2:] != (target_h, target_w):
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(0), size=(target_h, target_w),
                    mode="nearest",
                ).squeeze(0)
            if gt_mask.shape[0] == 1:
                gt_mask = gt_mask.repeat(3, 1, 1)
            imgs.append(gt_mask)

        # One rendered body row per pass, including recurrent cascade intermediates.
        for p in range(num_passes):
            for v in range(V):
                imgs.append(all_pass_rendered_image[p, b, v])
        if all_pass_canonical_rendered_image is not None:
            for p in range(num_passes):
                for v in range(V):
                    imgs.append(all_pass_canonical_rendered_image[p, b, v])

        # Shared GT face row, followed by one rendered face row per pass.
        if face_gt is not None and all_pass_rendered_face is not None:
            for v in range(V):
                face_img = face_gt[b, v]
                if face_img.shape[-2:] != (target_h, target_w):
                    face_img = F.interpolate(
                        face_img.unsqueeze(0), size=(target_h, target_w),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)
                imgs.append(face_img)

            for p in range(num_passes):
                for v in range(V):
                    face_img = all_pass_rendered_face[p, b, v]
                    if face_img.shape[-2:] != (target_h, target_w):
                        face_img = F.interpolate(
                            face_img.unsqueeze(0), size=(target_h, target_w),
                            mode="bilinear", align_corners=False,
                        ).squeeze(0)
                    imgs.append(face_img)
            if all_pass_canonical_rendered_face is not None:
                for p in range(num_passes):
                    for v in range(V):
                        face_img = all_pass_canonical_rendered_face[p, b, v]
                        if face_img.shape[-2:] != (target_h, target_w):
                            face_img = F.interpolate(
                                face_img.unsqueeze(0), size=(target_h, target_w),
                                mode="bilinear", align_corners=False,
                            ).squeeze(0)
                        imgs.append(face_img)

        grid = vutils.make_grid(torch.stack(imgs), nrow=V, padding=2)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        im.save(os.path.join(save_dir, f"mpg_step{step}_p{process_index}_b{b}.png"))


def visualize_nps_train_intermediate_images(
    batch: Dict,
    all_pass_rendered_image: torch.Tensor,
    save_dir: str,
    step: int,
    process_index: int,
    all_pass_rendered_face: Optional[torch.Tensor] = None,
    face_gt: Optional[torch.Tensor] = None,
) -> None:
    """Save intermediate-pass renders to mpg_step{step}_p{p}_b{b}_intermediate.png.

    Only writes a file when there are intermediate passes (num_passes > 1).
    Each file contains: one GT row, then one rendered row per intermediate pass.
    The final pass is intentionally excluded — it belongs in the main PNG.
    """
    num_passes = all_pass_rendered_image.shape[0]
    if num_passes <= 1:
        return

    B, V = batch["images"].shape[:2]
    target_h, target_w = all_pass_rendered_image.shape[-2:]
    gt_images = batch["images"]
    if "hr_images" in batch and batch["hr_images"].shape[-2:] == (target_h, target_w):
        gt_images = batch["hr_images"]

    intermediate_indices = list(range(num_passes - 1))

    for b in range(B):
        imgs = []

        # Row 1: GT body images
        for v in range(V):
            gt_img = gt_images[b, v].detach().cpu()
            if gt_img.shape[-2:] != (target_h, target_w):
                gt_img = F.interpolate(
                    gt_img.unsqueeze(0), size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            imgs.append(gt_img)

        # Row 2: GT masks
        gt_masks = batch["masks"]
        for v in range(V):
            gt_mask = gt_masks[b, v].detach().cpu()
            if gt_mask.shape[-2:] != (target_h, target_w):
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(0), size=(target_h, target_w),
                    mode="nearest",
                ).squeeze(0)
            if gt_mask.shape[0] == 1:
                gt_mask = gt_mask.repeat(3, 1, 1)
            imgs.append(gt_mask)

        # One rendered row per intermediate pass
        for p in intermediate_indices:
            for v in range(V):
                imgs.append(all_pass_rendered_image[p, b, v])

        # GT face row + intermediate rendered face rows
        if face_gt is not None and all_pass_rendered_face is not None:
            for v in range(V):
                face_img = face_gt[b, v]
                if face_img.shape[-2:] != (target_h, target_w):
                    face_img = F.interpolate(
                        face_img.unsqueeze(0), size=(target_h, target_w),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)
                imgs.append(face_img)
            for p in intermediate_indices:
                for v in range(V):
                    face_img = all_pass_rendered_face[p, b, v]
                    if face_img.shape[-2:] != (target_h, target_w):
                        face_img = F.interpolate(
                            face_img.unsqueeze(0), size=(target_h, target_w),
                            mode="bilinear", align_corners=False,
                        ).squeeze(0)
                    imgs.append(face_img)

        grid = vutils.make_grid(torch.stack(imgs), nrow=V, padding=2)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        Image.fromarray(ndarr).save(
            os.path.join(save_dir, f"mpg_step{step}_p{process_index}_b{b}_intermediate.png")
        )


CHECKPOINT_STEP_RE = re.compile(r"(?:^|[/\\\\])PT_(\d+)\.pth$")


def infer_resume_global_step(args) -> int:
    checkpoint_path = args.model.get("load_checkpoint")
    if not checkpoint_path:
        return 0

    match = CHECKPOINT_STEP_RE.search(str(checkpoint_path))
    if match is None:
        logging.warning(
            "Could not infer global step from checkpoint path %s; starting from step 0",
            checkpoint_path,
        )
        return 0
    return int(match.group(1))


def _get_offset_pose_tensor(deform_dic: Optional[Dict]) -> Optional[torch.Tensor]:
    if deform_dic is None:
        return None
    offset_pose = deform_dic.get("offset_pose")
    return offset_pose if isinstance(offset_pose, torch.Tensor) else None


def _canonical_animation_enabled(pass_output: Dict, loss_weight_dic: Dict[str, float]) -> bool:
    if loss_weight_dic["loss_canonical_animation"] <= 0:
        return False
    return _get_offset_pose_tensor(pass_output.get("deform_dic")) is not None


def _build_loss_weight_dict(loss_args) -> Dict[str, float]:
    return {
        "loss_face": float(getattr(loss_args.loss, "loss_face", 0.0)),
        "loss_uv_smooth": float(getattr(loss_args.loss, "loss_uv_smooth", 0.0)),
        "loss_offset_norm": float(getattr(loss_args.loss, "loss_offset_norm", 0.0)),
        "loss_normal_direction_reg": float(getattr(loss_args.loss, "loss_normal_direction_reg", 100.0)),
        # Backward compatibility: PoseDep configs commonly set only loss_offset_reg.
        "loss_poseoffset_reg": float(
            getattr(loss_args.loss, "loss_poseoffset_reg", getattr(loss_args.loss, "loss_offset_reg", 0.0))
        ),
        "loss_token_pose_strain": float(getattr(loss_args.loss, "loss_token_pose_strain", 0.0)),
        "loss_pose_shell_transfer": float(getattr(loss_args.loss, "loss_pose_shell_transfer", 0.0)),
        "loss_agg_pixel_gs_reg": float(getattr(loss_args.loss, "loss_agg_pixel_gs_reg", 0.0)),
        "loss_offset_reg": float(getattr(loss_args.loss, "loss_offset_reg", 0.0)),
        "loss_token_offset_consistency": float(
            getattr(loss_args.loss, "loss_token_offset_consistency", 0.0)
        ),
        "loss_canonical_animation": float(
            getattr(loss_args.loss, "loss_canonical_animation", 0.0)
        ),
        "loss_scale_reg": float(getattr(loss_args.loss, "loss_scale_reg", 0.0)),
        "loss_uv_pixel_strain": float(getattr(loss_args.loss, "loss_uv_pixel_strain", 0.0)),
    }
