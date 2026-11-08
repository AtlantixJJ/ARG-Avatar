import os
import struct
import logging
import torch
import numpy as np


# standalone GaussianParameters class for Spark export
class GaussianParameters:
    def __init__(self, xyz, scale, rotation, opacity, sh, offset_xyz=None):
        self.xyz = xyz.detach().float().cpu() if xyz is not None else None
        self.scale = scale.detach().float().cpu()
        self.rotation = rotation.detach().float().cpu()
        self.opacity = opacity.detach().float().cpu()
        self.sh = sh.detach().float().cpu() if sh is not None else None
        self.offset_xyz = offset_xyz.detach().float().cpu() if offset_xyz is not None else None

    
    def clone(self):
        return GaussianParameters(
            xyz=self.xyz.clone() if self.xyz is not None else None,
            scale=self.scale.clone(),
            rotation=self.rotation.clone(),
            opacity=self.opacity.clone(),
            sh=self.sh.clone() if self.sh is not None else None,
            offset_xyz=self.offset_xyz.clone() if self.offset_xyz is not None else None,
        )


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


PLY_FLOAT_PROPS = (
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "opacity",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
)


def _to_spark_ready_gs(gs: GaussianParameters, centerize: bool = True) -> GaussianParameters:
    """Return a GaussianParameters object suitable for Spark export."""
    if centerize:
        gs = centerize_g(gs)
    return GaussianParameters(
        xyz=gs.xyz.detach().float().cpu(),
        scale=gs.scale.detach().float().cpu(),
        rotation=gs.rotation.detach().float().cpu(),
        opacity=gs.opacity.detach().float().cpu(),
        sh=None if gs.sh is None else gs.sh.detach().float().cpu(),
    )

def export_spark_ply(gs: GaussianParameters, save_path: str, centerize: bool = True):
    """Export a binary little-endian PLY that Spark can read as a 3DGS asset."""
    gs = _to_spark_ready_gs(gs, centerize=centerize)
    xyz = gs.xyz.reshape(-1, 3)
    scale = gs.scale.reshape(-1, 3).clamp_min(1e-12)
    rotation = torch.nn.functional.normalize(gs.rotation.reshape(-1, 4), dim=-1, eps=1e-8)
    opacity = gs.opacity.reshape(-1, 1).clamp(1e-6, 1 - 1e-6)
    opacity_raw = torch.logit(opacity)

    if gs.sh is not None and gs.sh.numel() > 0:
        f_dc = gs.sh.reshape(-1, gs.sh.shape[-2], 3)[:, 0]
    else:
        f_dc = torch.zeros((xyz.shape[0], 3), dtype=torch.float32)

    rows = torch.cat(
        [
            xyz,
            scale.log(),
            rotation,
            opacity_raw,
            f_dc,
        ],
        dim=-1,
    ).numpy()

    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment argavatar_rotation_order xyzw",
        f"element vertex {rows.shape[0]}",
    ]
    header.extend([f"property float {name}" for name in PLY_FLOAT_PROPS])
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(header_bytes)
        for row in rows:
            f.write(struct.pack("<14f", *row.tolist()))


def export_delta_offset_xyz(gs_tpose: dict | GaussianParameters | torch.Tensor, save_path: str):
    """Export raw little-endian float32 offset_xyz values with shape [N, 3]."""
    if isinstance(gs_tpose, dict):
        delta = gs_tpose["offset_xyz"][0, 0].detach().float().cpu()
    elif isinstance(gs_tpose, torch.Tensor):
        delta = gs_tpose.detach().float().cpu()
    else:
        delta = gs_tpose.offset_xyz.detach().float().cpu()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    delta.reshape(-1, 3).contiguous().numpy().astype(np.float32).tofile(save_path)
