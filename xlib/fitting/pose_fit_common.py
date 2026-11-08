
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import roma
import torch
import torch.nn.functional as F
from torch import Tensor

from xlib.model.sam_3d_body.models.modules.mhr_utils import (
    compact_cont_to_model_params_body,
    mhr_param_hand_mask,
)
from xlib.utils import DictRecorder
from xlib.render.visualizer import plot_dict


def make_img_key(image_path: str) -> str:
    """Return 'dir/base.ext' key for an image path."""
    return os.path.join(os.path.basename(os.path.dirname(image_path)), os.path.basename(image_path))


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert [C, H, W] float tensor [0, 1] to [H, W, C] uint8 numpy array."""
    return (tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def perspective_projection(x: Tensor, K: Tensor) -> Tensor:
    """Project camera-space 3D points with intrinsics K."""
    y = x / x[:, :, -1:]
    K = K.to(dtype=y.dtype, device=y.device)
    y = torch.einsum("bij,bkj->bki", K, y)
    return y[:, :, :2]


def rot6d_to_rotmat(x: Tensor) -> Tensor:
    """Convert fitting-local 6D rotation representation to a rotation matrix."""
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(x: Tensor) -> Tensor:
    """Convert a rotation matrix to the column-major 6D convention above."""
    batch_dim = x.size()[:-2]
    return x[..., :, :2].transpose(-1, -2).contiguous().reshape(batch_dim + (6,))

@dataclass
class PoseFitInputs:
    view_keys: List[str]
    pose_raw: Tensor  # [V, 266]
    shape: Tensor     # [V, 45]
    scale: Tensor     # [V, 28]
    hand: Tensor      # [V, 108]
    face: Tensor      # [V, 72]
    cam_t: Tensor     # [V, 3]
    cam_rot: Tensor   # [V, 3, 3]
    cam_trans: Tensor # [V, 3]
    K_fit: Tensor     # [V, 3, 3]
    fit_size: Tensor  # [V, 2]
    kp2d_target: Tensor  # [V, 70, 2]
    kp3d_target: Optional[Tensor] = None  # [V, 70, 3] — 3D positions behind kp2d_target
    original_per_view_mhr_model_params: Optional[Tensor] = None  # [V, 204]
    
    def to(self, device):
        return PoseFitInputs(
            view_keys=self.view_keys,
            pose_raw=self.pose_raw.to(device) if self.pose_raw is not None else None,
            shape=self.shape.to(device) if self.shape is not None else None,
            scale=self.scale.to(device) if self.scale is not None else None,
            hand=self.hand.to(device) if self.hand is not None else None,
            face=self.face.to(device) if self.face is not None else None,
            cam_t=self.cam_t.to(device) if self.cam_t is not None else None,
            cam_rot=self.cam_rot.to(device) if self.cam_rot is not None else None,
            cam_trans=self.cam_trans.to(device) if self.cam_trans is not None else None,
            K_fit=self.K_fit.to(device) if self.K_fit is not None else None,
            fit_size=self.fit_size.to(device) if self.fit_size is not None else None,
            kp2d_target=self.kp2d_target.to(device) if self.kp2d_target is not None else None,
            kp3d_target=self.kp3d_target.to(device) if self.kp3d_target is not None else None,
            original_per_view_mhr_model_params=self.original_per_view_mhr_model_params.to(device) if self.original_per_view_mhr_model_params is not None else None,
        )


def as_view_tensor(entry: Dict, key: str, device: torch.device, shape: Tuple) -> Tensor:
    return torch.as_tensor(entry[key], device=device, dtype=torch.float32).view(*shape)


def decode_pose_raw(pose_raw_shared: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Decodes the raw pose representation into euler angles for MHR.
    Mirroring logic from xlib/model/sam_3d_body/models/heads/mhr_head.py
    """
    # pose_raw_shared: [B, 266]
    global_rot_6d = pose_raw_shared[:, :6]
    global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
    global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)

    pred_pose_cont = pose_raw_shared[:, 6:]
    pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)

    # Zero-out hands and jaw as in mhr_head.py
    pred_pose_euler[:, mhr_param_hand_mask] = 0
    pred_pose_euler[:, -3:] = 0

    return global_rot_euler, pred_pose_euler


def medoid_index(features: Tensor) -> int:
    """Return the feature row with the smallest mean distance to all rows."""
    flat = features.float().flatten(1)
    pairwise_l1 = torch.cdist(flat, flat, p=1)
    scores = pairwise_l1.mean(dim=1)
    return int(scores.argmin().item())


def project_keypoints_3d(kp3d_cam: Tensor, K: Tensor) -> Tensor:
    """Simple wrapper for perspective_projection."""
    return perspective_projection(kp3d_cam, K)


def sample_mesh_depth_at_points(
    verts_cam_v: Tensor,
    faces: Tensor,
    query_xy_v: Tensor,
    K_v: Tensor,
    image_size_v: Tensor,
    nvdiffrast_context,
) -> Tensor:
    """Render mesh depth with Kaolin and sample it at sparse query pixels.

    Args:
        verts_cam_v: [V, N, 3] mesh vertices in camera space.
        faces: [F, 3] mesh faces.
        query_xy_v: [V, K, 2] query pixels in processed-image coordinates.
        K_v: [V, 3, 3] processed-image intrinsics.
        image_size_v: [V, 2] image sizes as (H, W).

    Returns:
        [V, K] nearest rendered mesh depth at each query pixel, inf if uncovered.
    """
    import kaolin as kal
    from xlib.geometry.kal_ops import render_mesh

    V, _, _ = verts_cam_v.shape
    faces = faces.to(device=verts_cam_v.device, dtype=torch.long)
    mesh_depth_v = []

    for v_idx in range(V):
        verts_cam = verts_cam_v[v_idx].float()
        query_xy = query_xy_v[v_idx].float()
        H_v = int(image_size_v[v_idx, 0].item())
        W_v = int(image_size_v[v_idx, 1].item())
        K_cur = K_v[v_idx].float()

        single_mesh = kal.rep.SurfaceMesh(
            vertices=verts_cam,
            faces=faces,
            vertex_features=torch.ones(
                verts_cam.shape[0], 3,
                device=verts_cam.device,
                dtype=verts_cam.dtype,
            ),
        )
        mesh = kal.rep.SurfaceMesh.cat([single_mesh])

        view_matrix = torch.eye(4, device=verts_cam.device, dtype=torch.float32)
        view_matrix[[1, 2]] *= -1
        camera = kal.render.camera.Camera.from_args(
            view_matrix=view_matrix[None],
            focal_x=K_cur[0, 0],
            focal_y=K_cur[1, 1],
            x0=K_cur[0, 2] - W_v / 2.0,
            y0=-(K_cur[1, 2] - H_v / 2.0),
            width=W_v,
            height=H_v,
            device=verts_cam.device,
            dtype=torch.float32,
        )
        render_result = render_mesh(camera, mesh, nvdiffrast_context)
        depth_img = -render_result["depth"][0, :, :, 0].to(verts_cam.dtype)
        face_idx = render_result["face_idx"][0]

        px = query_xy[:, 0].round().long()
        py = query_xy[:, 1].round().long()
        in_image = (px >= 0) & (px < W_v) & (py >= 0) & (py < H_v)
        px = px.clamp(0, W_v - 1)
        py = py.clamp(0, H_v - 1)
        sampled_depth = depth_img[py, px]
        # rast0 in kal_ops already has torch.flip(dims=(1,)), so depth_img is in image-Y convention
        covered = in_image & (face_idx[py, px] >= 0) & (sampled_depth > 0.1)
        sampled_depth = torch.where(
            covered,
            sampled_depth,
            torch.full_like(sampled_depth, torch.inf),
        )
        mesh_depth_v.append(sampled_depth)

    return torch.stack(mesh_depth_v, dim=0)


def pad_mhr_model_params(params: Tensor, target_dim: int = 204) -> Tensor:
    """Pad MHR model params to target_dim (usually 204 for SMPL-X)."""
    if params.shape[-1] < target_dim:
        padding = torch.zeros(*params.shape[:-1], target_dim - params.shape[-1], device=params.device, dtype=params.dtype)
        return torch.cat([params, padding], dim=-1)
    return params


# MHR 70-keypoint skeleton edges derived from skeleton_info in mhr70.py.
BODY_KP_CONNECTIONS: List[Tuple[int, int]] = [
    # Body
    (13, 11), (11, 9), (14, 12), (12, 10), (9, 10),
    (5, 9), (6, 10), (5, 6), (5, 7), (6, 8), (7, 62), (8, 41),
    # Face
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    # Feet
    (13, 15), (13, 16), (13, 17), (14, 18), (14, 19), (14, 20),
    # Left hand (wrist=62): thumb, index, middle, ring, pinky
    (62, 45), (45, 44), (44, 43), (43, 42),
    (62, 49), (49, 48), (48, 47), (47, 46),
    (62, 53), (53, 52), (52, 51), (51, 50),
    (62, 57), (57, 56), (56, 55), (55, 54),
    (62, 61), (61, 60), (60, 59), (59, 58),
    # Right hand (wrist=41): thumb, index, middle, ring, pinky
    (41, 24), (24, 23), (23, 22), (22, 21),
    (41, 28), (28, 27), (27, 26), (26, 25),
    (41, 32), (32, 31), (31, 30), (30, 29),
    (41, 36), (36, 35), (35, 34), (34, 33),
    (41, 40), (40, 39), (39, 38), (38, 37),
]


def draw_keypoints_on_image(
    img_np: np.ndarray,
    kp2d: np.ndarray,
    visible: np.ndarray,
    connections: List[Tuple[int, int]],
    point_radius: int = 4,
    line_thickness: int = 1,
) -> np.ndarray:
    """Draw skeleton lines and per-joint circles on a copy of img_np (RGB uint8)."""
    out = img_np.copy()
    H, W = out.shape[:2]

    def _pt(k):
        x, y = float(kp2d[k, 0]), float(kp2d[k, 1])
        return (int(round(x)), int(round(y))), np.isfinite(x) and np.isfinite(y)

    for i, j in connections:
        if i >= len(kp2d) or j >= len(kp2d):
            continue
        pi, fi = _pt(i)
        pj, fj = _pt(j)
        if not (fi and fj):
            continue
        in_img = (0 <= pi[0] < W and 0 <= pi[1] < H) or (0 <= pj[0] < W and 0 <= pj[1] < H)
        if in_img:
            cv2.line(out, pi, pj, (180, 180, 180), line_thickness, cv2.LINE_AA)

    for k in range(len(kp2d)):
        pt, finite = _pt(k)
        if not finite:
            continue
        color = (0, 255, 0) if bool(visible[k]) else (255, 0, 0)
        cv2.circle(out, pt, point_radius, color, -1, cv2.LINE_AA)

    return out


def save_loss_record_plot(loss_record: DictRecorder, output_path: str) -> None:
    if not loss_record:
        return

    def _as_numpy(key: str):
        vals = loss_record.get(key, [])
        if isinstance(vals, torch.Tensor):
            return vals.detach().cpu().float().numpy().reshape(-1)
        if not vals:
            return np.asarray([], dtype=np.float32)
        return np.asarray([
            float(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else float(v)
            for v in vals
        ], dtype=np.float32)

    steps = _as_numpy("step")
    if steps.size == 0:
        first_key = next(iter(loss_record.keys()))
        steps = np.arange(_as_numpy(first_key).shape[0])

    plot_dic = {
        "loss": {
            key: _as_numpy(key)
            for key in ("loss", "loss_kp", "loss_prior", "loss_consistency")
            if _as_numpy(key).size
        },
        "optimization_state": {
            key: _as_numpy(key)
            for key in ("smpl_visible_fraction", "pose_delta", "cam_delta")
            if _as_numpy(key).size
        },
    }
    plot_dic = {key: val for key, val in plot_dic.items() if val}
    plot_dict(plot_dic, fpath=output_path, x=steps, N_row=len(plot_dic), N_col=1)


def prepare_pose_fit_inputs(
    per_view_entries: List[Dict],
    device: torch.device,
    animatable_gs=None,
) -> PoseFitInputs:
    """Gather per-view tensors and prepare inputs for pose fitting."""
    V = len(per_view_entries)
    view_keys = [entry["img_key"] for entry in per_view_entries]

    pose_raw = torch.stack([as_view_tensor(e, "pred_pose_raw", device, (266,)) for e in per_view_entries])
    shape = torch.stack([as_view_tensor(e, "shape", device, (45,)) for e in per_view_entries])
    scale = torch.stack([as_view_tensor(e, "scale", device, (28,)) for e in per_view_entries])
    hand = torch.stack([as_view_tensor(e, "hand", device, (108,)) for e in per_view_entries])
    face = torch.stack([as_view_tensor(e, "face", device, (72,)) for e in per_view_entries])
    cam_t = torch.stack([as_view_tensor(e, "pred_cam_t", device, (3,)) for e in per_view_entries])
    K_fit = torch.stack([as_view_tensor(e, "K_proc", device, (3, 3)) for e in per_view_entries])
    fit_size = torch.stack([as_view_tensor(e, "img_size", device, (2,)) for e in per_view_entries])

    cam_rot_list = []
    cam_trans_list = []
    missing_extrinsic_indices = []
    missing_mhr_params = []
    for v, entry in enumerate(per_view_entries):
        if "cam_rot" in entry and "cam_trans" in entry:
            cam_rot_list.append(torch.as_tensor(entry["cam_rot"], device=device, dtype=torch.float32).view(3, 3))
            cam_trans_list.append(torch.as_tensor(entry["cam_trans"], device=device, dtype=torch.float32).view(3))
        elif animatable_gs is not None and "mhr_model_params" in entry:
            cam_rot_list.append(None)
            cam_trans_list.append(None)
            missing_extrinsic_indices.append(v)
            missing_mhr_params.append(torch.as_tensor(entry["mhr_model_params"], device=device, dtype=torch.float32).view(-1))
        else:
            # Fallback for naive variant which might just use pred_pose_raw[:6]
            cam_rot_list.append(None)
            cam_trans_list.append(None)
            missing_extrinsic_indices.append(v)
            # We'll handle this below if it's still None

    if missing_mhr_params and animatable_gs is not None:
        with torch.no_grad():
            derived_rot, derived_trans = animatable_gs.get_root_rigid_in_colmap(torch.stack(missing_mhr_params))
        for local_idx, v in enumerate(missing_extrinsic_indices):
            cam_rot_list[v] = derived_rot[local_idx]
            cam_trans_list[v] = derived_trans[local_idx]

    # Final fallback for cam_rot/cam_trans if still None (e.g. naive mode without mhr_model_params).
    # pose_raw[:6] gives R_mhr; convert to R_colmap = F @ R_mhr @ F^{-1} where F=diag(1,-1,-1).
    for v in range(V):
        if cam_rot_list[v] is None:
            R_mhr = rot6d_to_rotmat(pose_raw[v, :6][None])[0]
            flip = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=torch.float32)
            cam_rot_list[v] = R_mhr * flip.view(1, 3) * flip.view(3, 1)
        if cam_trans_list[v] is None:
            cam_trans_list[v] = torch.zeros(3, device=device, dtype=torch.float32)

    cam_rot = torch.stack(cam_rot_list)
    cam_trans = torch.stack(cam_trans_list)

    kp3d_target = None
    if all("pred_keypoints_3d" in e for e in per_view_entries):
        kp3d_v = torch.stack([
            torch.as_tensor(e["pred_keypoints_3d"], device=device, dtype=torch.float32).view(-1, 3)[:70]
            for e in per_view_entries
        ])
        kp3d_target = kp3d_v + cam_t[:, None, :]  # [V, 70, 3] — camera-space 3D positions
        kp2d_target = project_keypoints_3d(kp3d_target, K_fit)
    else:
        kp2d_target = torch.stack([
            torch.as_tensor(e["pred_keypoints_2d"], device=device, dtype=torch.float32).view(70, 2)
            for e in per_view_entries
        ])

    orig_params = None
    if all("mhr_model_params" in e for e in per_view_entries):
        orig_params = torch.stack([
            pad_mhr_model_params(torch.as_tensor(e["mhr_model_params"], device=device, dtype=torch.float32).view(-1))
            for e in per_view_entries
        ])

    return PoseFitInputs(
        view_keys=view_keys,
        pose_raw=pose_raw,
        shape=shape,
        scale=scale,
        hand=hand,
        face=face,
        cam_t=cam_t,
        cam_rot=cam_rot,
        cam_trans=cam_trans,
        K_fit=K_fit,
        fit_size=fit_size,
        kp2d_target=kp2d_target,
        kp3d_target=kp3d_target,
        original_per_view_mhr_model_params=orig_params,
    )
