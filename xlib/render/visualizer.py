# pylint: skip-file
"""All kinds of visualizations."""
import logging
import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn
seaborn.set_theme()
import torchvision.utils as vutils
import wandb
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Any, Dict, Tuple
from diffusers.utils.export_utils import export_to_video
from accelerate import Accelerator
import torch.nn.functional as F

from xlib.render.gaussian_splatting import splat_gaussians, GaussianParameters, SH2RGB
from xlib.render.gaussian_splatting_3d import index_gs
from xlib.render.camera import get_lookat_camera, GSCamera
from xlib.render.camera_utils import horizontal_turntable_angles
from xlib.model.mhr_animatable_gaussians import animate_gs
from xlib.misc.spark_utils import export_spark_ply, export_delta_offset_xyz
from xlib.utils import AsyncExecutor
from xlib.geometry.uv_utils import scatter_feature_uv

#################################
######## Creating Images ########
#################################


def add_text_to_tensor_images(image_batch: torch.Tensor, text: str, text_color: tuple = (0, 0, 0)) -> torch.Tensor:
    """Add text to the upper center of batched tensor images (N, C, H, W format)."""
    if image_batch.dim() == 3:
        # Single image (C, H, W), add batch dimension
        image_batch = image_batch.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    N, C, H, W = image_batch.shape
    font_size = int(min(H, W) * 0.05)
    result_batch = []
    
    for i in range(N):
        # Convert tensor to PIL Image
        image_pil = Image.fromarray((image_batch[i].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default(size=font_size)
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position (upper center)
        img_width, img_height = image_pil.size
        x = (img_width - text_width) // 2
        y = int(img_height * 0.01)
        
        # Draw text (no background)
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Convert back to tensor
        result_tensor = torch.from_numpy(np.array(image_pil)).float() / 255.0
        result_batch.append(result_tensor.permute(2, 0, 1))  # Back to C, H, W format
    
    result = torch.stack(result_batch, dim=0)  # N, C, H, W
    return result.squeeze(0) if squeeze_output else result



def plot_dict(dic, fpath=None, x=None, N_row=None, N_col=None):
    if N_row is None:
        N = len(dic.keys())
        N_row = math.ceil(math.sqrt(N))
        N_col = math.ceil(N / N_row)

    fig = plt.figure(figsize=(6 * N_col, 4 * N_row))
    for i, (k, v) in enumerate(dic.items()):
        ax = plt.subplot(N_row, N_col, i + 1)
        if type(v) is dict:  # multiple lines with legend
            for iv in v.values():
                if x is not None:
                    ax.plot(x, iv)
                else:
                    ax.plot(iv)
            ax.legend(list(v.keys()))
        else:
            if x is not None:
                ax.plot(x, v)
            else:
                ax.plot(v)
        ax.set_title(k)
    plt.tight_layout()

    if fpath:
        plt.savefig(fpath)
        plt.close()
    return fig


def save_image_grid(img, fname=None, drange=[0, 1], grid_size=[4, 4]):
    """Save a grid of images.
    Args:
        img: Grid of images (np.array) to save.
        fname: File name to save the images.
        drange: Data range.
        grid_size: Size of the image grid.
    Returns:
        numpy image grid.
    """
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo + 1e-9))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        image = Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        image = Image.fromarray(img, 'RGB').save(fname)
    return img


#################################
######## Creating Videos ########
#################################



def render_rotate(
        gaussians: GaussianParameters,
        bg_color: torch.Tensor,
        fov: float = 0.2443,
        radius: float = 2.7,
        elev_low=-math.pi/4,
        elev_high=math.pi/4,
        opacity_key='opacity',
        n_frames: int = 60,
        resolution=512,
        clamp_image=True
    ) -> List[torch.Tensor]:
    """
    Args:
        gaussians: GaussianParameters, the gaussian parameters.
        bg_color: The background color.
        fov: The field of view.
        radius: The radius of the camera.
        n_frames: The number of frames in the video.
    Returns:
        images: List of torch.Tensor, N x (1, 3, H, W), the rendered images, [0, 1]
        depths: List of torch.Tensor, N x (1, 1, H, W), the rendered depths, not normalized.
    """

    azims, elevs = horizontal_turntable_angles(n_frames, 1, elev_low, elev_high)
    device = bg_color.device

    common_kwargs = {
        'fov': fov,
        'radius': radius,
        'device': device,
        'H': resolution, 'W': resolution
    }

    images, depths = [], []
    with torch.no_grad():
        for i in range(len(azims)):
            cam = get_lookat_camera(azims=azims[i:i+1], elevs=elevs[i:i+1], **common_kwargs)
            res = splat_gaussians([gaussians], cam, bg_color,
                                    opacity_key=opacity_key,
                                    use_gen_size=False)
            img = res['image']
            if clamp_image:
                img = img.clamp(0, 1)
            images.append(img.cpu())
            depths.append(res['depth'].cpu())
    image = torch.cat(images, dim=0)
    depth = torch.cat(depths, dim=0).repeat(1, 3, 1, 1)
    return image, depth


def make_render_rotate_grid(*args, grid_size=None):
    """Make a grid of images for the rotating camera.
    Args:
        args: List of tensor in shape (N, C, H, W)
    """
    n_frames = args[0].shape[0]
    n_items = len(args)
    # generating permuting indices: avoid using too much contiguous()
    indices = torch.arange(n_items * n_frames).view(n_items, -1)
    indices = indices.permute(1, 0).contiguous().view(-1)
    frames = torch.stack([args[i // n_frames][i % n_frames] for i in indices])
    frames = frames.view(n_frames, -1, *frames.shape[1:])
    if grid_size is None:
        grid_size = math.ceil(math.sqrt(frames.shape[1]))
    fn = lambda x: (x.permute(1, 2, 0) * 255).numpy().astype("uint8")
    frames = [fn(vutils.make_grid(f, nrow=grid_size)) for f in frames]
    return frames


def async_write_video(
        async_executor: AsyncExecutor,
        accelerator: Accelerator,
        video_path: str,
        global_step: int,
        frames: list[Image.Image]) -> None:
    """
    Handle visualization of model predictions during training. Execute in async executor.

    Args:
        async_executor: AsyncExecutor instance
        accelerator: Accelerator instance
        video_path: Path to save video
        global_step: Current global step
        frames: Frames to save
    """
    async_executor.clean_sync()
    def worker():
        export_to_video([Image.fromarray(f) for f in frames], video_path, fps=12)
        if accelerator.is_main_process:
            accelerator.log({"frames": wandb.Video(video_path)}, step=global_step)
    async_executor(worker)


def centerize_g(g: GaussianParameters):
    """Center Gaussian parameters by subtracting mean z-coordinate.

    Args:
        g: GaussianParameters to center

    Returns:
        Centered GaussianParameters (cloned to avoid in-place operations)
    """
    # Clone to avoid in-place operations on shared memory
    g_cloned = g.clone()

    if g_cloned.xyz.dim() == 3:
        xyz_mean = g_cloned.xyz.mean(dim=1, keepdim=True)  # [B, 1, 3]
        g_cloned.xyz -= xyz_mean  # Use slice to maintain dimension
    elif g_cloned.xyz.dim() == 2:  # 2D case after split(1) - [N, 3] or other dimensions
        xyz_mean = g_cloned.xyz.mean(dim=0, keepdim=True)  # [1, 3]
        g_cloned.xyz -= xyz_mean
    else:
        print(f"Warning: Unexpected dimensions for xyz: {g_cloned.xyz.dim()}, skip centering")
    return g_cloned


@torch.no_grad()
def visualize_canonical_rotation_frames(
    animatable_gs,
    gs_tpose_b,
    zero_pose,
    bg_color: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    """Render canonical rotation frames as [N, 3, H, W]."""
    torch.cuda.empty_cache()
    canonical_pose = {
        "shape_params": zero_pose["shape_params"],
        "model_params": torch.zeros_like(zero_pose["model_params"]),
        "expr_params": zero_pose["expr_params"],
    }
    gs_canonical, _ = animate_gs(
        animatable_gs,
        gs_tpose_b,
        canonical_pose,
        pose_indices=[0],
        to_colmap=True,
    )
    gs_view = index_gs(gs_canonical, 0, 0)
    gs_view = centerize_g(gs_view)
    frames, _ = render_rotate(gs_view, radius=8, bg_color=bg_color.float(), resolution=resolution)
    return frames.detach().cpu().clamp(0, 1)


def normalize_to_unit(
    x: torch.Tensor,
    eps: float = 1e-6,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize each channel to [0, 1] over spatial dimensions.

    If mask is provided, min/max are computed only on valid pixels and output is
    zeroed on invalid pixels.
    """
    if mask is None:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    if mask.dim() == 2:
        valid = mask.view(1, 1, *mask.shape)
    elif mask.dim() == 4:
        valid = mask
    else:
        raise ValueError(f"mask must be [H, W] or [B|1, 1|C, H, W], got {tuple(mask.shape)}")
    valid = valid.to(device=x.device) > 0

    x_min = x.masked_fill(~valid, float("inf")).amin(dim=(-2, -1), keepdim=True)
    x_max = x.masked_fill(~valid, float("-inf")).amax(dim=(-2, -1), keepdim=True)
    out = (x - x_min) / (x_max - x_min + eps)
    return out.masked_fill(~valid, 0.0)


#### UV Visualization #####


def reconstruct_component_to_full(
    c_img: torch.Tensor,
    comp: "UVComponentInfo",
    uv_size: int,
) -> torch.Tensor:
    """Reconstruct a component map to the full UV space."""
    def _component_valid_extent(comp: "UVComponentInfo") -> Tuple[int, int]:
        valid_rows = torch.where(comp.pixel_mask.any(dim=1))[0]
        valid_cols = torch.where(comp.pixel_mask.any(dim=0))[0]
        if valid_rows.numel() == 0 or valid_cols.numel() == 0:
            return comp.scaled_size
        return int(valid_rows[-1].item()) + 1, int(valid_cols[-1].item()) + 1

    B, C = c_img.shape[:2]
    full_recon = torch.zeros(B, C, uv_size, uv_size, device=c_img.device, dtype=c_img.dtype)
    r_min, c_min, r_max, c_max = comp.bbox
    h_orig, w_orig = r_max - r_min, c_max - c_min
    valid_h, valid_w = _component_valid_extent(comp)
    resized = F.interpolate(
        c_img[:, :, :valid_h, :valid_w],
        size=(h_orig, w_orig),
        mode="bilinear",
        align_corners=False,
    )
    mask = comp.orig_pixel_mask.view(1, 1, h_orig, w_orig).to(resized)
    orig_content = full_recon[:, :, r_min:r_max, c_min:c_max]
    full_recon[:, :, r_min:r_max, c_min:c_max] = resized * mask + orig_content * (1 - mask)
    return full_recon


def gs_uv_attr_images_masked(
    comp_map: torch.Tensor,
    spatial_mask: torch.Tensor,
    *,
    pose_debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """RGB-style panels for Gaussian splatting channels in a [B, C, H, W] UV map.

    spatial_mask is [H, W] or [1, 1, H, W] (broadcast over batch and channels).
    When pose_debug is True and C >= 20, includes offset_pose and canonical_xyz+offset panels.
    """
    if spatial_mask.dim() == 2:
        m = spatial_mask.view(1, 1, *spatial_mask.shape)
    elif spatial_mask.dim() == 4:
        m = spatial_mask
    else:
        raise ValueError(f"spatial_mask must be [H, W] or [1, 1, H, W], got {tuple(spatial_mask.shape)}")
    m = m.to(device=comp_map.device, dtype=comp_map.dtype)
    vis: Dict[str, torch.Tensor] = {
        "offset": normalize_to_unit(comp_map[:, 0:3], mask=m),
        "scale": normalize_to_unit(comp_map[:, 3:6], mask=m),
        "sh": comp_map[:, 11:14],
        "opacity": comp_map[:, 10:11],
    }
    quat = comp_map[:, 6:10]
    quat_rgb = quat[:, 0:3] / (quat[:, 3:4].abs() + 1e-6)
    vis["rotation"] = quat_rgb * m
    if pose_debug and comp_map.shape[1] >= 20:
        vis["offset-pose"] = normalize_to_unit(comp_map[:, 14:17], mask=m)
        vis["offset-final"] = normalize_to_unit(comp_map[:, 17:20], mask=m)
    return vis


@torch.no_grad()
def _visualize_uv_encoding_connected_components(
    uv_enc: "UVEncoding",
    features: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Visualize UV encoding in connected_components mode from per-patch features [B, L, C]."""
    device = uv_enc.patch_positions.device
    B, _, C = features.shape
    comp_images: List[torch.Tensor] = []
    for comp in uv_enc.components:
        st = comp.pixel_global_offset
        H, W = comp.scaled_size
        comp_feat = features[:, st : st + comp.n_valid_pixels]
        dense = torch.zeros(B, H * W, C, device=device, dtype=features.dtype)
        dense[:, comp.pixel_mask.view(-1), :] = comp_feat
        comp_images.append(dense.permute(0, 2, 1).view(B, C, H, W))

    if len(comp_images) != len(uv_enc.components):
        missing_component_ids = [comp.component_id for comp in uv_enc.components[len(comp_images):]]
        logging.warning(
            "visualize_uv_encoding received %s component tensors for %s registered UV components; missing component_ids=%s",
            len(comp_images),
            len(uv_enc.components),
            missing_component_ids,
        )

    B, C = comp_images[0].shape[:2]
    full_recon = torch.zeros(
        B, C, uv_enc.uv_size, uv_enc.uv_size, device=device, dtype=comp_images[0].dtype
    )
    for comp, c_img in zip(uv_enc.components, comp_images):
        full_recon += reconstruct_component_to_full(c_img, comp, uv_enc.uv_size)

    full_mask = torch.zeros(1, 1, uv_enc.uv_size, uv_enc.uv_size, device=device, dtype=full_recon.dtype)
    for comp in uv_enc.components:
        r_min, c_min, r_max, c_max = comp.bbox
        comp_mask = comp.orig_pixel_mask.view(1, 1, r_max - r_min, c_max - c_min).to(device=device, dtype=full_recon.dtype)
        full_mask[:, :, r_min:r_max, c_min:c_max] = torch.maximum(
            full_mask[:, :, r_min:r_max, c_min:c_max], comp_mask
        )

    return {
        "full_reconstructed": full_recon,
        "full_mask": full_mask,
    }


@torch.no_grad()
def visualize_uv_encoding(
    uv_enc: "UVEncoding",
    features: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Visualize UV layout from flat per-patch features [B, L, C].

    Returns:
        Dict containing:
            'full_reconstructed': [B, C, H_uv, W_uv] tensor
            'full_mask': [1, 1, H_uv, W_uv] tensor
    """
    assert uv_enc is not None, "uv_enc must be provided"
    assert uv_enc.is_setup, "uv_enc must be setup"

    if uv_enc.mode == "naive":
        full_recon = scatter_feature_uv(features, uv_enc.uv_mask, uv_size=uv_enc.uv_size)
        full_mask = uv_enc.uv_mask.view(1, 1, uv_enc.uv_size, uv_enc.uv_size).to(
            device=full_recon.device, dtype=full_recon.dtype
        )
        return {
            "full_reconstructed": full_recon,
            "full_mask": full_mask,
        }
    if uv_enc.mode == "connected_components":
        return _visualize_uv_encoding_connected_components(uv_enc=uv_enc, features=features)
    raise NotImplementedError(f"Unknown mode: {uv_enc.mode}")


@torch.no_grad()
def save_uv_encoding_visualizations(
    uv_enc,
    gs_tpose_dic,
    offset_pose: Optional[torch.Tensor],
    save_dir: str,
    step: int,
    process_index: int,
    pass_index: int = 0,
) -> None:
    """Save UV encoding visualizations for the given UV encoding, canonical GS attributes, and pose-dependent deform.
    Args:
        uv_enc: UV encoding
        gs_tpose_dic: Canonical GS attributes
        offset_pose: Pose-dependent per-Gaussian offset (optional); first batch index is used for viz
        save_dir: Directory to save the visualizations
        step: Current step
        process_index: Current process index
        pass_index: Current pass index
    Returns:
        None
    """

    # Scatter canonical 3DGS attrs plus pose-dependent debug channels back to UV space in one pass.
    attrs = [gs_tpose_dic["offset_xyz"], gs_tpose_dic["scale"], gs_tpose_dic["rotation"], gs_tpose_dic["opacity"], SH2RGB(gs_tpose_dic["sh"][..., :1, :].squeeze(-2))]
    gs_feat = torch.cat(attrs, dim=-1)[0]
    offset_pose_ch = torch.zeros_like(gs_feat[:, :, 0:3])
    if offset_pose is not None:
        offset_pose_ch = offset_pose[:1]  # take the first pose
    device, dtype = gs_feat.device, gs_feat.dtype
    offset_pose_ch = offset_pose_ch.to(device=device, dtype=dtype)

    final_xyz_offset = gs_feat[:, :, 0:3] + offset_pose_ch
    vis_feat = torch.cat([gs_feat, offset_pose_ch, final_xyz_offset], dim=-1)  # [B, L, 20]
    vis = visualize_uv_encoding(uv_enc=uv_enc, features=vis_feat)
    full_reconstructed = vis['full_reconstructed'].detach().float().cpu()
    full_mask = vis["full_mask"].detach().float().cpu()

    full_visualizations = gs_uv_attr_images_masked(full_reconstructed, full_mask, pose_debug=True)

    for key, image in full_visualizations.items():
        prefix = f'uvgsattr_{key}'
        for b_idx in range(image.shape[0]):
            fpath = os.path.join(save_dir, f"{prefix}_step{step}_p{process_index}_b{b_idx}_v0_pass{pass_index}.png")
            vutils.save_image(image[b_idx], fpath)


# ---------------------------------------------------------------------------
# Pass-output visualization helpers (shared by train.py & visualize_checkpoints.py)
# ---------------------------------------------------------------------------

def _tpose_dict_to_render_gs(d: dict) -> GaussianParameters:
    sh = d.get("sh")
    return GaussianParameters(
        xyz=d["xyz"][0, 0],
        scale=d["scale"][0, 0],
        rotation=d["rotation"][0, 0],
        opacity=d["opacity"][0, 0],
        sh=sh[0, 0] if sh is not None else None,
        active_sh_degree=0,
    )


def _merge_all_offsets_into_offset_xyz(gs_dic: dict, extra_offsets: Optional[List[torch.Tensor]] = None) -> dict:
    merged = {**gs_dic}
    base = merged.get("offset_xyz", torch.zeros_like(merged["xyz"]))
    offsets = [base]
    if "offset_pose" in merged:
        offsets.append(merged["offset_pose"])
    if extra_offsets:
        offsets.extend(o for o in extra_offsets if o is not None)
    merged["offset_xyz"] = sum(offsets[1:], offsets[0]) if offsets else base
    merged.pop("offset_pose", None)
    return merged


def _offset_pose_tensor_for_view_idx(offset_pose: torch.Tensor, view_idx: int) -> torch.Tensor:
    if offset_pose.ndim == 5:
        # [B, V, 1, N, 3] → [B, 1, N, 3]
        return offset_pose[:, view_idx]
    if offset_pose.ndim == 3 and offset_pose.shape[0] > 1:
        return offset_pose[view_idx:view_idx + 1]
    return offset_pose


def _view_offset_from_pass(pass_result: dict, view_idx: int) -> Optional[torch.Tensor]:
    deform_dic = pass_result.get("deform_dic")
    if deform_dic is not None:
        offset_pose = deform_dic.get("offset_pose")
        if isinstance(offset_pose, torch.Tensor):
            return _offset_pose_tensor_for_view_idx(offset_pose, view_idx)
        return offset_pose
    return None


def _resolve_view0_offset_pose(
    *,
    view0_offset_pose: Optional[torch.Tensor] = None,
    deform_dic: Optional[dict] = None,
    offset_pose: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if view0_offset_pose is not None:
        return view0_offset_pose
    if deform_dic is not None:
        deform_view0 = _view_offset_from_pass({"deform_dic": deform_dic}, 0)
        if deform_view0 is not None:
            return deform_view0
    if offset_pose is not None and isinstance(offset_pose, torch.Tensor):
        return _offset_pose_tensor_for_view_idx(offset_pose, 0)
    return None


def _save_video(frames: torch.Tensor, save_path: str, fps: int = 30) -> None:
    frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    export_to_video([Image.fromarray(f) for f in frames_np], save_path, fps=fps)


def _render_rotation_from_gs(
    gs: GaussianParameters,
    save_path: str,
    bg_color: torch.Tensor,
    resolution: int = 512,
) -> None:
    frames, _ = render_rotate(gs, fov=1.0, radius=1.8, resolution=resolution, bg_color=bg_color)
    _save_video(frames, save_path)

def build_posed_gs(
    gs_tpose: dict,
    pose_dic: dict,
    animatable_gs,
    offset_pose: Optional[torch.Tensor] = None,
) -> GaussianParameters:
    gs_tpose_anim = _merge_all_offsets_into_offset_xyz(gs_tpose)
    gs_posed_ij, _ = animate_gs(
        animatable_gs, gs_tpose_anim, pose_dic,
        pose_indices=[0], to_colmap=True, offset_pose=offset_pose,
    )
    return centerize_g(index_gs(gs_posed_ij, 0, 0))


def _gs_from_tpose(gs_tpose: dict) -> GaussianParameters:
    return GaussianParameters(
        xyz=gs_tpose["xyz"][0, 0],
        scale=gs_tpose["scale"][0, 0],
        rotation=gs_tpose["rotation"][0, 0],
        opacity=gs_tpose["opacity"][0, 0],
        sh=gs_tpose["sh"][0, 0] if gs_tpose.get("sh") is not None else None,
    )

@torch.no_grad()
def export_spark(
    output_canonical_gs_dic: dict,
    pose_dic: dict,
    animatable_gs,
    output_dir: str,
    step_tag: str,
    *,
    view0_offset_pose: Optional[torch.Tensor] = None,
) -> GaussianParameters:
    """Export Spark PLYs and per-Gaussian offset ``.bin`` files.

    Writes canonical deltas from ``output_canonical_gs_dic["offset_xyz"]`` via
    ``{step_tag}_canonical_offset_xyz.bin``. When ``view0_offset_pose`` is set, also writes
    ``{step_tag}_view0_offset_pose.bin`` and ``{step_tag}_posed_delta.bin``.

    Returns posed Gaussian parameters (with view-0 ``offset_pose`` when present) for
    :func:`export_rotate_video`.
    """
    canonical_gs = output_canonical_gs_dic

    export_delta_offset_xyz(
        {"offset_xyz": output_canonical_gs_dic["offset_xyz"]},
        os.path.join(output_dir, f"{step_tag}_canonical_offset_xyz.bin"),
    )
    if view0_offset_pose is not None:
        export_delta_offset_xyz(
            view0_offset_pose,
            os.path.join(output_dir, f"{step_tag}_view0_offset_pose.bin"),
        )

    export_spark_ply(
        _gs_from_tpose(canonical_gs),
        os.path.join(output_dir, f"{step_tag}_canonical.ply"),
    )

    posed_gs = build_posed_gs(
        canonical_gs,
        pose_dic,
        animatable_gs,
        offset_pose=None,
    )
    export_spark_ply(
        posed_gs,
        os.path.join(output_dir, f"{step_tag}_posed.ply"),
    )
    if view0_offset_pose is not None:
        posed_gs_with_offset = build_posed_gs(
            canonical_gs,
            pose_dic,
            animatable_gs,
            offset_pose=view0_offset_pose,
        )
        export_delta_offset_xyz(
            posed_gs_with_offset.xyz - posed_gs.xyz,
            os.path.join(output_dir, f"{step_tag}_posed_delta.bin"),
        )
    else:
        posed_gs_with_offset = posed_gs

    return posed_gs_with_offset


@torch.no_grad()
def export_rotate_video(
    canonical_gs: dict,
    posed_gs_with_offset: GaussianParameters,
    bg_color: torch.Tensor,
    output_dir: str,
    step_tag: str,
    resolution: int = 512,
    *,
    skip_videos: bool = False,
) -> None:
    """Export canonical and posed turntable MP4s."""
    if skip_videos:
        logging.info("  skip_videos=True — skipping rotation MP4 exports")
        return

    canonical_rotation_path = os.path.join(output_dir, f"{step_tag}_canonical_rotation.mp4")
    _render_rotation_from_gs(
        centerize_g(_tpose_dict_to_render_gs(canonical_gs)),
        canonical_rotation_path,
        bg_color,
        resolution=resolution,
    )
    logging.info(f"  canonical rotation -> {canonical_rotation_path}")

    posed_rotation_path = os.path.join(output_dir, f"{step_tag}_posed_rotation.mp4")
    _render_rotation_from_gs(
        posed_gs_with_offset,
        posed_rotation_path,
        bg_color,
        resolution=resolution,
    )
    logging.info(f"  posed rotation     -> {posed_rotation_path}")
