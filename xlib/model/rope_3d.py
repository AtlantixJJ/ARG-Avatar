from typing import Any, Dict, List, Optional, Tuple

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import trimesh

from xlib.render.gaussian_splatting import splat_gaussians, RGB2SH
from xlib.misc.spark_utils import GaussianParameters as SparkGS, export_spark_ply, export_delta_offset_xyz


def save_ply(path, points, colors=None):
    """Save [N, 3] tensor as PLY file with optional [N, 3] RGB colors."""
    points_np = points.detach().cpu().numpy()
    if colors is not None:
        colors_np = colors.detach().cpu().numpy()
        if colors_np.shape[0] != points_np.shape[0]:
            raise ValueError(f"Color count {colors_np.shape[0]} must match point count {points_np.shape[0]}")
    
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points_np)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, p in enumerate(points_np):
            if colors is not None:
                c = colors_np[i]
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


def _prepare_uint8_colors(colors: torch.Tensor) -> torch.Tensor:
    if colors.dtype == torch.uint8:
        return colors
    if colors.is_floating_point():
        if colors.max() <= 1.0 and colors.min() >= 0.0:
            colors = colors * 255.0
        colors = colors.clamp(0.0, 255.0)
    return colors.to(torch.uint8)


def _scaled_camera_intrinsics(
    cameras: List,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return per-camera intrinsics scaled to the active image/render size."""
    scaled = []
    for cam in cameras:
        K = cam.K.to(device=device, dtype=dtype).clone()
        full_w = float(getattr(cam, "full_image_width", cam.image_width))
        full_h = float(getattr(cam, "full_image_height", cam.image_height))
        cur_w = float(cam.image_width)
        cur_h = float(cam.image_height)
        if full_w > 0 and full_h > 0 and (abs(full_w - cur_w) > 1e-6 or abs(full_h - cur_h) > 1e-6):
            K[0] *= cur_w / full_w
            K[1] *= cur_h / full_h
        scaled.append(K)
    return torch.stack(scaled, dim=0)


def _project_camera_points(
    posed: torch.Tensor,
    cameras: List,
    render_depth: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world points with the same rasterizer path used to render depth."""
    world2cam_t = torch.stack([cam.world_view_transform.to(posed) for cam in cameras], dim=0)
    proj_t = torch.stack([cam.full_proj_transform.to(posed) for cam in cameras], dim=0)
    points_cam = posed @ world2cam_t[:, :3, :] + world2cam_t[:, 3:4, :]
    points_clip = posed @ proj_t[:, :3, :] + proj_t[:, 3:4, :]
    points_ndc = points_clip[..., :3] / (points_clip[..., 3:4] + 1e-8)

    h, w = render_depth.shape[-2:]
    pixel_x = ((points_ndc[..., 0] + 1.0) * w - 1.0) / 2.0
    pixel_y = ((points_ndc[..., 1] + 1.0) * h - 1.0) / 2.0
    return points_cam, points_ndc, pixel_x, pixel_y


def _camera_depth_to_feature_scale(
    z_avatar: torch.Tensor,
    cameras: List,
    render_depth: torch.Tensor,
    feat_w: int,
) -> torch.Tensor:
    """Convert camera-space depth deltas into feature-space x units."""
    _, w = render_depth.shape[-2:]
    K = _scaled_camera_intrinsics(cameras, device=z_avatar.device, dtype=z_avatar.dtype)
    scale_x = K[:, None, 0, 0] / z_avatar
    return scale_x * (float(feat_w) / float(w))


def map_values_to_color(values: torch.Tensor) -> torch.Tensor:
    """Map scalar values to colors using red-white-blue colormap.
    
    Positive values: white to red
    Negative values: white to blue
    
    Args:
        values: [N] tensor of scalar values
        
    Returns:
        colors: [N, 3] tensor of RGB colors in [0, 255] range
    """
    POSITIVE_COLOR = matplotlib.colormaps["Reds"]
    NEGATIVE_COLOR = matplotlib.colormaps["Blues"]
    
    # Normalize by absolute max
    abs_max = values.abs().max()
    if abs_max < 1e-8:
        # All values are zero, return white
        return torch.ones(values.shape[0], 3, device=values.device) * 255.0
    
    values_norm = values / abs_max  # Range: [-1, 1]
    values_np = values_norm.detach().cpu().numpy()
    
    # Separate positive and negative values
    pos_mask = values_np > 0
    neg_mask = values_np < 0
    
    # Initialize with white (1.0, 1.0, 1.0)
    colors = np.ones((len(values_np), 3))
    
    # Map positive values to red colormap
    if pos_mask.any():
        pos_colors = POSITIVE_COLOR(values_np[pos_mask])[:, :3]
        colors[pos_mask] = pos_colors
    
    # Map negative values to blue colormap
    if neg_mask.any():
        neg_colors = NEGATIVE_COLOR(-values_np[neg_mask])[:, :3]
        colors[neg_mask] = neg_colors
    
    # Convert to [0, 255] range and return as tensor
    colors_tensor = torch.from_numpy(colors).to(values.device).float() * 255.0
    return colors_tensor


def _export_canonical_dic_as_spark(
    gs_dic: Dict[str, torch.Tensor],
    path: str,
) -> None:
    """Export a canonical GS dict as a Spark PLY, composing xyz + offset_xyz."""

    def _to_2d(t: torch.Tensor) -> torch.Tensor:
        while t.ndim > 2:
            t = t[0]
        return t.detach().float().cpu()

    def _to_3d(t: torch.Tensor) -> torch.Tensor:
        while t.ndim > 3:
            t = t[0]
        return t.detach().float().cpu()

    xyz = _to_2d(gs_dic["xyz"])
    off = gs_dic.get("offset_xyz")
    if off is not None:
        xyz = xyz + _to_2d(off)
    n_pts = xyz.shape[0]

    scale = _to_2d(gs_dic["scale"]) if "scale" in gs_dic else torch.full((n_pts, 3), 0.005)
    rotation = (
        _to_2d(gs_dic["rotation"])
        if "rotation" in gs_dic
        else torch.cat([torch.ones(n_pts, 1), torch.zeros(n_pts, 3)], dim=1)
    )
    opacity = _to_2d(gs_dic["opacity"]) if "opacity" in gs_dic else torch.ones(n_pts, 1)
    sh = _to_3d(gs_dic["sh"]) if "sh" in gs_dic else torch.zeros(n_pts, 1, 3)

    export_spark_ply(
        SparkGS(xyz=xyz, scale=scale, rotation=rotation, opacity=opacity, sh=sh),
        path,
        centerize=False,
    )


def _make_pseudo_spark_gs(xyz: torch.Tensor, colors_rgb01: Optional[torch.Tensor] = None, default_scale: float = math.log(0.005)) -> SparkGS:
    """Create a minimal SparkGS for debug export with identity attrs and given xyz [N, 3]."""
    N = xyz.shape[0]
    scale = torch.full((N, 3), math.exp(default_scale), dtype=torch.float32)
    rotation = torch.zeros(N, 4, dtype=torch.float32)
    rotation[:, 0] = 1.0  # identity quaternion (w=1)
    opacity = torch.ones(N, 1, dtype=torch.float32)
    if colors_rgb01 is not None:
        # SH DC: color = sigmoid(f_dc * SH_C0 + 0.5) → f_dc = (logit(color) - 0.5) / SH_C0
        f_dc = RGB2SH(colors_rgb01.float())  # [N, 3]
        sh = f_dc.unsqueeze(1)  # [N, 1, 3]
    else:
        sh = None
    sh_arg = sh if sh is not None else torch.zeros(N, 1, 3, dtype=torch.float32)
    return SparkGS(xyz=xyz.float().cpu(), scale=scale, rotation=rotation, opacity=opacity, sh=sh_arg)


def debug_rope_projection(
    batch: Dict[str, torch.Tensor],
    render_result: Dict[str, Any],
    fac_patch: torch.Tensor,
    posed_xyz: torch.Tensor,
    folder: str,
    prefix: str,
    posed_gs: Optional[Any] = None,
    export_image_gs: Optional[bool] = None,
    token_canonical_gs_dic: Optional[Dict[str, torch.Tensor]] = None,
    offset_pose: Optional[torch.Tensor] = None,
) -> None:
    if export_image_gs is None:
        export_image_gs = prefix.startswith("pass0")
    os.makedirs(folder, exist_ok=True)

    import torchvision.utils as vutils

    img = batch["img"][0] if batch["img"].ndim == 5 else batch["img"]
    render_image = render_result["image"]

    if render_image.shape[-2:] != img.shape[-2:]:
        render_image = F.interpolate(
            render_image,
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    depth_disp = render_result["depth"].repeat(1, 3, 1, 1)
    if depth_disp.shape[-2:] != img.shape[-2:]:
        depth_disp = F.interpolate(
            depth_disp,
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    depth_min = depth_disp.min()
    depth_max = depth_disp.max()
    depth_disp = (depth_disp - depth_min) / (depth_max - depth_min).clamp(min=1e-8)

    disp = torch.cat([img, render_image, depth_disp])
    vutils.save_image(disp, os.path.join(folder, f"{prefix}_disp.png"))

    # --- Spark GS exports ---
    # 1. GS used to compute FAC (original posed GS) — shape [B, V, N, ...], B=V=1
    gs_b0v0 = None
    if posed_gs is not None:
        gs_b0v0 = posed_gs.squeeze(0).squeeze(0)  # [N, ...]
        export_spark_ply(gs_b0v0, os.path.join(folder, f"{prefix}_posed_gs.ply"))

    # 2 & 3. FAC GS and (pass0-only) image GS scaled into posed-GS world space.
    fac_xyz = fac_patch[0].detach().float().cpu()  # [L, 3]  pixel space
    pose_xyz_ref = gs_b0v0.xyz.detach().float().cpu() if gs_b0v0 is not None else posed_xyz[0].detach().float().cpu()

    # Scale/offset that maps pixel space → world space so FAC overlays on posed_gs.
    fac_center = fac_xyz.mean(dim=0)
    fac_extent = (fac_xyz - fac_center).abs().max().item()
    pose_center = pose_xyz_ref.mean(dim=0)
    pose_extent = (pose_xyz_ref - pose_center).abs().max().item()
    pix_to_world = pose_extent / max(fac_extent, 1e-6)

    def _pix_to_world(pts):
        return (pts - fac_center) * pix_to_world + pose_center

    fac_xyz = _pix_to_world(fac_xyz)

    if export_image_gs:
        img_b = batch["img"]
        if img_b.ndim == 5:
            img_b = img_b[0]
        img_b = img_b[0]  # [C, H, W]
        H_img, W_img = img_b.shape[-2:]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H_img, dtype=torch.float32),
            torch.arange(W_img, dtype=torch.float32),
            indexing="ij",
        )
        img_xyz = torch.stack(
            [x_grid.flatten() + 0.5, y_grid.flatten() + 0.5, torch.zeros(H_img * W_img)],
            dim=-1,
        )  # [H*W, 3]
        img_colors = img_b.float().permute(1, 2, 0).reshape(-1, 3).clamp(0, 1).cpu()

        # Apply the same pixel→world transform so image GS aligns with posed_gs.
        img_xyz = _pix_to_world(img_xyz)

        log_scale = math.log(0.002)
        spark_img = _make_pseudo_spark_gs(img_xyz, colors_rgb01=img_colors, default_scale=log_scale)
        export_spark_ply(spark_img, os.path.join(folder, f"{prefix}_image_gs.ply"), centerize=False)

    log_scale = math.log(0.01)
    spark_fac = _make_pseudo_spark_gs(fac_xyz, default_scale=log_scale)
    export_spark_ply(spark_fac, os.path.join(folder, f"{prefix}_fac_gs.ply"), centerize=False)

    if posed_gs is not None and offset_pose is not None:
        export_delta_offset_xyz(offset_pose, os.path.join(folder, f"{prefix}_posed_delta.bin"))

    if token_canonical_gs_dic is not None:
        _export_canonical_dic_as_spark(
            token_canonical_gs_dic,
            os.path.join(folder, f"{prefix}_token_canonical_gs.ply"),
        )
        export_delta_offset_xyz(
            token_canonical_gs_dic,
            os.path.join(folder, f"{prefix}_token_canonical_delta.bin"),
        )


def compute_continuous_3d_cis(
    coords: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Compute complex RoPE frequencies for continuous 3D coordinates."""
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    d_half = dim // 2
    d_per_axis = dim // 6

    if d_per_axis == 0:
        return torch.ones(
            *coords.shape[:-1], d_half, device=coords.device, dtype=torch.complex64
        )

    freq_dtype = torch.float32
    freqs = 1.0 / (
        theta ** (torch.arange(d_per_axis, device=coords.device, dtype=freq_dtype) / d_per_axis)
    )

    x = coords[..., 0:1].to(freq_dtype)
    y = coords[..., 1:2].to(freq_dtype)
    z = coords[..., 2:3].to(freq_dtype)

    cis_x = torch.polar(torch.ones_like(x * freqs), x * freqs)
    cis_y = torch.polar(torch.ones_like(y * freqs), y * freqs)
    cis_z = torch.polar(torch.ones_like(z * freqs), z * freqs)

    freqs_cis = torch.cat([cis_x, cis_y, cis_z], dim=-1)
    if freqs_cis.shape[-1] < d_half:
        pad = d_half - freqs_cis.shape[-1]
        freqs_cis = torch.cat(
            [
                freqs_cis,
                torch.ones(
                    *freqs_cis.shape[:-1], pad, device=freqs_cis.device, dtype=freqs_cis.dtype
                ),
            ],
            dim=-1,
        )
    return freqs_cis


def apply_rotary_enc_3d(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis_q: torch.Tensor,
    freqs_cis_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D RoPE on q and k.
    
    Args:
        xq: [B, H, N_q, D]
        xk: [B, H, N_kv, D]
        freqs_cis_q: [B, N_q, D/2] or [N_q, D/2] complex
        freqs_cis_k: [B, N_kv, D/2] or [N_kv, D/2] complex
    """
    if xq.shape[-1] % 2 != 0 or xk.shape[-1] % 2 != 0:
        raise ValueError("RoPE expects even head dimension.")

    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    def reshape_cis(cis, x):
        if cis.ndim == 2: # [N, D/2] -> [1, 1, N, D/2]
            return cis.unsqueeze(0).unsqueeze(1).to(x.dtype)
        if cis.ndim == 3: # [B, N, D/2] -> [B, 1, N, D/2]
            return cis.unsqueeze(1).to(x.dtype)
        return cis.to(x.dtype)

    q_cis = reshape_cis(freqs_cis_q, xq_complex)
    xq_complex = xq_complex * q_cis

    k_cis = reshape_cis(freqs_cis_k, xk_complex)
    xk_complex = xk_complex * k_cis

    xq_out = torch.view_as_real(xq_complex).flatten(-2).to(xq.dtype)
    xk_out = torch.view_as_real(xk_complex).flatten(-2).to(xk.dtype)
    return xq_out, xk_out


class RoPE3DCrossAttention(nn.Module):
    """
    3D Rotary Position Embedding (RoPE3D) for cross-attention.
    
    This layer applies rotary position encodings to queries and keys based 
    on their 3D coordinates. In this codebase, it's primarily used for:
    - Queries: Avatar tokens (at their animated 3D positions).
    - Keys/Values: Image features (projected into 3D using their 2D pixel
      coordinates and the rendered/relative depth at those pixels).
    
    This mechanism provides strong spatial grounding for the cross-attention,
    making avatar token refinement aware of the 3D geometry and alignment.
    
    Args:
        avatar_dim: Dimension of query tokens.
        cond_dim: Dimension of conditioning/context (keys/values).
        num_heads: Number of attention heads.
        rope_theta: Base value for RoPE frequency computation.
    """

    def __init__(
        self,
        avatar_dim: int,
        cond_dim: int,
        num_heads: int = 8,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if avatar_dim % num_heads != 0:
            raise ValueError(
                f"avatar_dim ({avatar_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = avatar_dim // num_heads
        self.rope_theta = rope_theta
        self.dropout_p = dropout

        self.q_proj = nn.Linear(avatar_dim, avatar_dim)
        self.k_proj = nn.Linear(cond_dim, avatar_dim)
        self.v_proj = nn.Linear(cond_dim, avatar_dim)
        self.out_proj = nn.Linear(avatar_dim, avatar_dim)

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, self.num_heads, c // self.num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_coords: torch.Tensor,
        kv_coords: torch.Tensor,
    ) -> torch.Tensor:
        q_proj = self._separate_heads(self.q_proj(q))
        k_proj = self._separate_heads(self.k_proj(kv))
        v_proj = self._separate_heads(self.v_proj(kv))

        freqs_q = compute_continuous_3d_cis(
            q_coords, dim=self.head_dim, theta=self.rope_theta
        )
        freqs_k = compute_continuous_3d_cis(
            kv_coords, dim=self.head_dim, theta=self.rope_theta
        )

        q_rot, k_rot = apply_rotary_enc_3d(
            q_proj, k_proj, freqs_q, freqs_k
        )

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q_rot, k_rot, v_proj, dropout_p=dropout_p)
        out = self._recombine_heads(out)
        return self.out_proj(out)


def project_points_to_feature_coords(
    posed: torch.Tensor,           # [B, N, 3]
    cameras: List,
    render_depth: torch.Tensor,    # [B, 1, H, W]
    feat_h: int, feat_w: int,
    coord_h: Optional[int] = None,
    coord_w: Optional[int] = None,
    depth_scale: float = 1.0,
    differentiable: bool = False,
    z_grad: bool = False,
    depth_grad: bool = False,
    z_clamp_min: float = 0.05,
) -> torch.Tensor:
    """Project avatar-token 3D points to shared image-space coordinates with relative depth.

    This function handles the projection of animated avatar points into the
    coordinate system used by image tokens, including a relative depth
    component (z_rel = z_avatar - z_rendered_depth) for 3D RoPE.

    differentiable=False (default) keeps the historical no-grad behavior.
    With differentiable=True, gradients flow through the x/y projection;
    z_grad additionally opens the relative-depth channel (1/z terms clamped
    by z_clamp_min) and depth_grad the rasterized-depth backward.
    """
    bsz = render_depth.shape[0]
    if len(cameras) != bsz:
        raise ValueError(f"Expected {bsz} cameras, got {len(cameras)}")

    # Ensure we're in fp32 for projection math; no-grad unless the caller opts
    # into the differentiable-FAC path (never force-enables grad under eval).
    grad_on = differentiable and torch.is_grad_enabled()
    with torch.set_grad_enabled(grad_on), torch.amp.autocast(device_type='cuda', enabled=False):
        if grad_on and not depth_grad:
            # Detached-visibility: occlusion depth is a constant per step.
            render_depth = render_depth.detach()
        # posed is [B, N, 3]
        points_cam, points_ndc, pixel_x, pixel_y = _project_camera_points(posed, cameras, render_depth)
        h, w = render_depth.shape[-2:]
        # 1/z and 1/z^2 backward terms need a real clamp in differentiable mode
        z_avatar = points_cam[..., 2].clamp(min=z_clamp_min if grad_on else 1e-6)

        # Sample depth at projected points
        grid = points_ndc[..., :2].unsqueeze(2) # [B, N, 1, 2]
        z_render = F.grid_sample(
            render_depth.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(1).squeeze(-1) # [B, N]

        # Compute relative depth: surface points have z_rel ≈ 0
        z_rel = z_avatar - z_render

        if coord_h is None:
            coord_h = feat_h
        if coord_w is None:
            coord_w = feat_w

        # Scale projected pixel coordinates into the same image-space system as the KV coords.
        coord_x = (pixel_x + 0.5) * (float(coord_w) / float(w))
        coord_y = (pixel_y + 0.5) * (float(coord_h) / float(h))

        # Perspective-consistent depth scaling in the same x/y unit system.
        z_norm = z_rel * _camera_depth_to_feature_scale(
            z_avatar=z_avatar,
            cameras=cameras,
            render_depth=render_depth,
            feat_w=coord_w,
        ) * depth_scale

        if grad_on and not z_grad:
            # x/y-only gradient stage: relative depth is ill-conditioned (1/z^2)
            z_norm = z_norm.detach()

        return torch.stack([coord_x, coord_y, z_norm], dim=-1)


def compute_image_grid_coords(feat_h: int, feat_w: int, device: torch.device) -> torch.Tensor:
    """Return image-grid coordinates [feat_h * feat_w, 3] with z=0."""
    y, x = torch.meshgrid(
        torch.arange(feat_h, device=device),
        torch.arange(feat_w, device=device),
        indexing="ij",
    )
    z = torch.zeros_like(x)
    return torch.stack(
        [x.flatten().float() + 0.5, y.flatten().float() + 0.5, z.flatten().float()],
        dim=-1,
    )

def compute_fac(
    posed_gs: Any,
    cameras: List[Any],
    feat_h: int,
    feat_w: int,
    depth_scale: float,
    coord_h: Optional[int] = None,
    coord_w: Optional[int] = None,
    uv_encoding: Optional[Any] = None,
    render_result: Optional[Dict[str, Any]] = None,
    differentiable: bool = False,
    z_grad: bool = False,
    depth_grad: bool = False,
    z_clamp_min: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Computes Front-Aligned Canonical (FAC) features for a single view.
    
    Args:
        posed_gs: GaussianParameters or TensorDict with .xyz or ["xyz"]
        cameras: List of GSCamera for this view
        feat_h: Feature map height
        feat_w: Feature map width
        coord_h: Coordinate space height
        coord_w: Coordinate space width
        depth_scale: Scale factor for depth projection
        uv_encoding: Optional aggregator (e.g., UVEncoding)
        render_result: Optional pre-computed splat_gaussians result
        
    Returns:
        patch_fac_i: Projected and optionally aggregated feature coordinates [B, L, 3]
        patch_posed_xyz: Raw posed world coordinates [B, L, 3]
        render_result: The dictionary returned by splat_gaussians
    """
    
    device = posed_gs["xyz"].device if isinstance(posed_gs, dict) else posed_gs.xyz.device

    # Read the projection input BEFORE splatting: GaussianParameters.squeeze()
    # mutates in place, and a view created inside the no-grad splat below would
    # silently detach the xyz used for the differentiable projection.
    posed_xyz = posed_gs.xyz[:, 0] if posed_gs.xyz.ndim == 4 else posed_gs.xyz

    if render_result is None:
        # In differentiable mode without depth_grad, the depth render only
        # provides (detached) visibility — skip the rasterizer backward graph.
        splat_no_grad = differentiable and not depth_grad
        with torch.set_grad_enabled(torch.is_grad_enabled() and not splat_no_grad):
            render_result = splat_gaussians(
                posed_gs.squeeze(1) if posed_gs.xyz.ndim == 4 else posed_gs,
                cameras,
                bg_color=torch.ones(3, device=device, dtype=torch.float32),
            )

    fac_i = project_points_to_feature_coords(
        posed_xyz,
        cameras,
        render_result["depth"],
        feat_h=feat_h,
        feat_w=feat_w,
        coord_h=coord_h,
        coord_w=coord_w,
        depth_scale=depth_scale,
        differentiable=differentiable,
        z_grad=z_grad,
        depth_grad=depth_grad,
        z_clamp_min=z_clamp_min,
    )
    
    if uv_encoding is not None:
        patch_fac_posed, _ = uv_encoding.aggregate_patch_points(
            torch.cat([fac_i, posed_xyz], dim=-1)
        )
        patch_fac_i, patch_posed_xyz = patch_fac_posed[..., :3], patch_fac_posed[..., 3:]
    else:
        patch_fac_i, patch_posed_xyz = fac_i, posed_xyz
        
    return patch_fac_i, patch_posed_xyz, render_result
