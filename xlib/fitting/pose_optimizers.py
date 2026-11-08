
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from torch import Tensor

from xlib.fitting.pose_fit_common import (
    PoseFitInputs,
    decode_pose_raw,
    medoid_index,
    rot6d_to_rotmat,
    rotmat_to_rot6d,
    project_keypoints_3d,
    pad_mhr_model_params,
    sample_mesh_depth_at_points,
)
from xlib.utils import DictRecorder

class BaseMHRPoseOptimizer(nn.Module):
    """Base class for MHR pose optimizers with shared camera and visibility logic."""
    def __init__(self, mhr_model: nn.Module, inputs: PoseFitInputs):
        super().__init__()
        self.mhr_model = mhr_model
        self.inputs = inputs
        self.V = len(inputs.view_keys)
        self.device = inputs.pose_raw.device

        # 1. Initialize per-view camera parameters
        self.cam_rot_param = nn.Parameter(rotmat_to_rot6d(inputs.cam_rot))
        self.cam_trans_param = nn.Parameter(inputs.cam_trans.clone())
        self.cam_t_param = nn.Parameter(inputs.cam_t.clone())

        # 2. Constants
        self.root_zero_6d = rotmat_to_rot6d(torch.eye(3, device=self.device)[None])[0]
        self.cam_t_init = inputs.cam_t.detach().clone()

        # Mesh faces for visibility
        self.mesh_faces = getattr(mhr_model, "faces", None)
        if self.mesh_faces is not None:
            self.mesh_faces = torch.as_tensor(self.mesh_faces, device=self.device, dtype=torch.long)

        self.nvdiffrast_context = None
        if self.mesh_faces is not None:
            from kaolin.render.mesh.nvdiffrast_context import default_nvdiffrast_context
            self.nvdiffrast_context = default_nvdiffrast_context(device=self.device, raise_error=True)

        self.joint_visibility_weight: Optional[Tensor] = None

    @torch.no_grad()
    def compute_joint_visibility(self, use_mesh_visibility: bool = True) -> None:
        """Compute joint visibility weights once from the initial pose and cache for the full optimization."""
        mhr_output = self.forward_mhr()
        j3d_cam, verts_cam = self.keypoints_camera(mhr_output)
        kp2d_pred = project_keypoints_3d(j3d_cam, self.inputs.K_fit)

        u = kp2d_pred[..., 0] / (self.inputs.fit_size[:, 1:2] - 1).clamp(min=1) * 2 - 1
        v = kp2d_pred[..., 1] / (self.inputs.fit_size[:, 0:1] - 1).clamp(min=1) * 2 - 1

        in_bbox = ((u.abs() <= 1) & (v.abs() <= 1)).float()
        weights = in_bbox

        if use_mesh_visibility and verts_cam is not None and self.mesh_faces is not None and self.nvdiffrast_context is not None:
            mesh_depth_at_kp = sample_mesh_depth_at_points(
                verts_cam,
                self.mesh_faces,
                kp2d_pred,
                self.inputs.K_fit,
                self.inputs.fit_size,
                self.nvdiffrast_context,
            )
            is_mesh_visible = (
                torch.isfinite(mesh_depth_at_kp)
                & (j3d_cam[..., 2] <= mesh_depth_at_kp + 0.2)
            ).float()
            weights = weights * is_mesh_visible

        self.joint_visibility_weight = weights

    def set_camera_requires_grad(self, enabled: bool):
        for p in [self.cam_rot_param, self.cam_trans_param, self.cam_t_param]:
            p.requires_grad_(enabled)

    def forward_mhr(self) -> Dict[str, Tensor]:
        """Runs MHR forward. Must return [V, ...] tensors."""
        raise NotImplementedError

    def keypoints_camera(self, mhr_output: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Transforms 3D keypoints and vertices to camera space for all views."""
        return self._mhr_output_to_camera(
            mhr_output,
            rot6d_to_rotmat(self.cam_rot_param),
            self.cam_trans_param,
            self.cam_t_param,
        )

    def _mhr_output_to_camera(
        self,
        mhr_output: Dict[str, Tensor],
        cam_rot: Tensor,
        cam_trans: Tensor,
        cam_t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        verts_body = mhr_output["verts"].clone()
        verts_body[..., [1, 2]] *= -1
        j3d_body = mhr_output["kp3d"][:, :70].clone()
        j3d_body[..., [1, 2]] *= -1

        verts_cam = torch.einsum("bij,bkj->bki", cam_rot, verts_body) + cam_trans[:, None, :] + cam_t[:, None, :]
        j3d_cam = torch.einsum("bij,bkj->bki", cam_rot, j3d_body) + cam_trans[:, None, :] + cam_t[:, None, :]

        return j3d_cam, verts_cam

    @torch.no_grad()
    def input_view_camera_geometry(self) -> Tuple[Tensor, Tensor]:
        """Return original per-input-view keypoints and mesh vertices in camera space."""
        root_6d_v = self.root_zero_6d[None].expand(self.V, -1)
        pose_raw_v = torch.cat([root_6d_v, self.inputs.pose_raw[:, 6:]], dim=1)
        global_rot_euler, body_pose_params = decode_pose_raw(pose_raw_v)

        mhr_forward_fn = self.mhr_model.mhr_forward if hasattr(self.mhr_model, "mhr_forward") else self.mhr_model
        output = mhr_forward_fn(
            global_trans=torch.zeros_like(global_rot_euler),
            global_rot=global_rot_euler,
            body_pose_params=body_pose_params,
            hand_pose_params=self.inputs.hand,
            scale_params=self.inputs.scale,
            shape_params=self.inputs.shape,
            expr_params=self.inputs.face,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
        )
        mhr_output = {
            "verts": output[0],
            "kp3d": output[1],
            "model_params": output[3],
        }
        return self._mhr_output_to_camera(
            mhr_output,
            self.inputs.cam_rot,
            self.inputs.cam_trans,
            self.inputs.cam_t,
        )

    def reprojection_loss(self, j3d_cam: Tensor) -> Tuple[Tensor, Dict]:
        """Computes 2D reprojection loss using visibility cached from the initial pose."""
        kp2d_pred = project_keypoints_3d(j3d_cam, self.inputs.K_fit)
        w = self.joint_visibility_weight

        loss_kp = (w * (kp2d_pred - self.inputs.kp2d_target).abs().sum(-1)).sum() / (w.sum().clamp(min=1.0))
        return loss_kp, {"loss_kp": loss_kp, "smpl_visible_fraction": w.mean()}

    def _kp_viz_dict(self, kp2d_fitted: Tensor, verts_cam: Optional[Tensor] = None) -> Dict:
        d = {
            "kp2d_fitted": kp2d_fitted.detach().cpu(),
            "kp2d_single": self.inputs.kp2d_target.detach().cpu(),
            "joint_visibility_weight": self.joint_visibility_weight.detach().cpu() > 0.5,
            "view_keys": list(self.inputs.view_keys),
        }
        if verts_cam is not None and self.mesh_faces is not None:
            _, input_verts_cam = self.input_view_camera_geometry()
            d["verts_cam"] = verts_cam.detach().cpu()
            d["input_verts_cam"] = input_verts_cam.detach().cpu()
            d["mesh_faces"] = self.mesh_faces.cpu()
            d["K_fit"] = self.inputs.K_fit.detach().cpu()
            d["fit_size"] = self.inputs.fit_size.detach().cpu()
        return d

    def _per_view_cam_dict(self, cam_rot: Tensor) -> Dict:
        return {
            img_key: {
                "pred_cam_t": self.cam_t_param[v].detach().cpu(),
                "cam_rot": cam_rot[v].detach().cpu(),
                "cam_trans": self.cam_trans_param[v].detach().cpu(),
            } for v, img_key in enumerate(self.inputs.view_keys)
        }

class SharedMHRPoseOptimizer(BaseMHRPoseOptimizer):
    """Optimizer for shared body pose across multiple views."""
    def __init__(self, mhr_model: nn.Module, inputs: PoseFitInputs):
        super().__init__(mhr_model, inputs)

        # 1. Initialize shared body parameters
        # Use medoid as anchor
        medoid_features = torch.cat([
            inputs.pose_raw[:, 6:],
            inputs.shape,
            inputs.scale,
            inputs.hand,
            inputs.face,
        ], dim=1)
        anchor_idx = medoid_index(medoid_features)
        self.anchor_idx = anchor_idx

        self.shared_pose_cont = nn.Parameter(inputs.pose_raw[anchor_idx, 6:].clone())
        self.shared_shape = nn.Parameter(inputs.shape[anchor_idx].clone())
        self.shared_scale = nn.Parameter(inputs.scale[anchor_idx].clone())
        self.shared_hand = nn.Parameter(inputs.hand[anchor_idx].clone())
        self.shared_face = nn.Parameter(inputs.face[anchor_idx].clone())

        # 2. Constants
        self.pose_cont_init = self.shared_pose_cont.detach().clone()
        self.shape_init = self.shared_shape.detach().clone()
        self.scale_init = self.shared_scale.detach().clone()
        self.hand_init = self.shared_hand.detach().clone()

    def forward_mhr(self) -> Dict[str, Tensor]:
        """Runs MHR forward for the shared pose."""
        pose_raw_shared = torch.cat([self.root_zero_6d[None], self.shared_pose_cont[None]], dim=1)
        global_rot_euler, body_pose_params = decode_pose_raw(pose_raw_shared)

        mhr_forward_fn = self.mhr_model.mhr_forward if hasattr(self.mhr_model, "mhr_forward") else self.mhr_model

        output = mhr_forward_fn(
            global_trans=torch.zeros_like(global_rot_euler),
            global_rot=global_rot_euler,
            body_pose_params=body_pose_params,
            hand_pose_params=self.shared_hand[None],
            scale_params=self.shared_scale[None],
            shape_params=self.shared_shape[None],
            expr_params=self.shared_face[None],
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
        )
        return {
            "verts": output[0].expand(self.V, -1, -1),
            "kp3d": output[1].expand(self.V, -1, -1),
            "model_params": output[3].expand(self.V, -1),
        }

    def reprojection_loss(self, j3d_cam: Tensor) -> Tuple[Tensor, Dict]:
        """Computes 2D reprojection loss with prior terms."""
        loss_kp, metrics = super().reprojection_loss(j3d_cam)

        # Prior loss
        loss_prior = (
            0.5 * (self.shared_pose_cont - self.pose_cont_init).pow(2).sum()
            + 1e-2 * (self.cam_t_param - self.cam_t_init).pow(2).sum()
            + 1e-3 * (
                (self.shared_shape - self.shape_init).pow(2).sum()
                + (self.shared_scale - self.scale_init).pow(2).sum()
                + (self.shared_hand - self.hand_init).pow(2).sum()
            )
        )

        loss = loss_kp + loss_prior
        metrics.update({
            "loss_prior": loss_prior,
        })

        return loss, metrics

    @torch.no_grad()
    def snapshot(self, step: int) -> Dict:
        """Generates a snapshot of the current state for evaluation/visualization."""
        mhr_output = self.forward_mhr()
        j3d_cam, verts_cam = self.keypoints_camera(mhr_output)
        kp2d_fitted = project_keypoints_3d(j3d_cam, self.inputs.K_fit)

        mhr_model_params = mhr_output["model_params"][0].clone()
        mhr_model_params[:6] = 0

        pose_raw_shared = torch.cat([self.root_zero_6d[None], self.shared_pose_cont[None]], dim=1)

        return {
            "step": step,
            "body": {
                "pred_pose_raw": pose_raw_shared[0].detach().cpu(),
                "shape": self.shared_shape.detach().cpu(),
                "scale": self.shared_scale.detach().cpu(),
                "hand": self.shared_hand.detach().cpu(),
                "face": self.shared_face.detach().cpu(),
                "mhr_model_params": pad_mhr_model_params(mhr_model_params).detach().cpu(),
            },
            "per_view": self._per_view_cam_dict(rot6d_to_rotmat(self.cam_rot_param)),
            "kp_viz": self._kp_viz_dict(kp2d_fitted, verts_cam),
        }

    def fit(
        self,
        n_iters: int,
        use_mesh_visibility: bool = True,
        optimize_cam_trans: bool = True,
    ) -> Tuple[DictRecorder, List]:
        """Run the optimization loop.

        Args:
            use_mesh_visibility: pass verts_cam to reprojection_loss for depth-based visibility.
            optimize_cam_trans: include cam_trans_param in the optimizer.
        Returns:
            (loss_record, snapshots)
        """
        if optimize_cam_trans:
            param_groups = [
                {'params': [self.shared_pose_cont], 'lr': 1e-4},
                {'params': [self.shared_shape, self.shared_scale, self.shared_hand, self.shared_face], 'lr': 1e-3},
                {'params': [self.cam_rot_param, self.cam_trans_param], 'lr': 1e-2},
                {'params': [self.cam_t_param], 'lr': 5e-3},
            ]
        else:
            param_groups = [
                {'params': [self.shared_pose_cont], 'lr': 1e-4},
                {'params': [self.shared_shape, self.shared_scale, self.shared_hand, self.shared_face], 'lr': 1e-3},
                {'params': [self.cam_rot_param, self.cam_t_param], 'lr': 5e-3},
            ]

        optimizer = torch.optim.Adam(param_groups, eps=1e-8, betas=(0.9, 0.999))
        loss_record = DictRecorder()
        self.compute_joint_visibility(use_mesh_visibility)
        snapshots = [self.snapshot(0)]

        with torch.enable_grad(), torch.amp.autocast(device_type=self.device.type, enabled=False):
            for step in range(n_iters):
                optimizer.zero_grad()
                mhr_output = self.forward_mhr()
                j3d_cam, _ = self.keypoints_camera(mhr_output)
                loss, metrics = self.reprojection_loss(j3d_cam)
                metrics["step"] = torch.as_tensor(step, device=self.device)
                metrics["pose_delta"] = (self.shared_pose_cont - self.pose_cont_init).norm()
                metrics["cam_delta"] = (self.cam_t_param - self.cam_t_init).norm()
                loss_record.add_dict(metrics)
                loss.backward()
                optimizer.step()
                step_idx = step + 1
                if step_idx % 10 == 0 or step_idx == n_iters:
                    snapshots.append(self.snapshot(step_idx))

        return loss_record, snapshots

    def result(self) -> Dict:
        """Final result format compatible with existing callers."""
        snap = self.snapshot(-1)
        return {
            "body": snap["body"],
            "per_view": snap["per_view"],
            "meta": {
                "view_keys": self.inputs.view_keys,
                "anchor_view_key": self.inputs.view_keys[self.anchor_idx],
                "version": "refactored-shared-2",
            },
        }

class PerViewMHRPoseOptimizer(BaseMHRPoseOptimizer):
    """Optimizer for per-view body pose with consistency regularization."""
    def __init__(self, mhr_model: nn.Module, inputs: PoseFitInputs, consistency_weight: float = 1.0):
        super().__init__(mhr_model, inputs)
        self.consistency_weight = consistency_weight

        # 1. Initialize per-view body parameters from their own predictions
        self.pv_pose_cont = nn.Parameter(inputs.pose_raw[:, 6:].clone())
        self.pv_shape = nn.Parameter(inputs.shape.clone())
        self.pv_scale = nn.Parameter(inputs.scale.clone())
        self.pv_hand = nn.Parameter(inputs.hand.clone())
        self.pv_face = nn.Parameter(inputs.face.clone())

        # 2. Constants
        self.pose_cont_init = self.pv_pose_cont.detach().clone()
        self.shape_init = self.pv_shape.detach().clone()
        self.scale_init = self.pv_scale.detach().clone()
        self.hand_init = self.pv_hand.detach().clone()

    def forward_mhr(self) -> Dict[str, Tensor]:
        """Runs MHR forward for all per-view poses."""
        root_6d_v = self.root_zero_6d[None].expand(self.V, -1)
        pose_raw_v = torch.cat([root_6d_v, self.pv_pose_cont], dim=1)
        g_euler, body_params = decode_pose_raw(pose_raw_v)

        mhr_forward_fn = self.mhr_model.mhr_forward if hasattr(self.mhr_model, "mhr_forward") else self.mhr_model

        output = mhr_forward_fn(
            global_trans=torch.zeros_like(g_euler),
            global_rot=g_euler,
            body_pose_params=body_params,
            hand_pose_params=self.pv_hand,
            scale_params=self.pv_scale,
            shape_params=self.pv_shape,
            expr_params=self.pv_face,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
        )
        return {
            "verts": output[0],
            "kp3d": output[1],
            "model_params": output[3],
        }

    def consistency_loss(self) -> Tensor:
        """Computes cross-view consistency loss."""
        mean_pose = self.pv_pose_cont.mean(dim=0, keepdim=True)
        mean_shape = self.pv_shape.mean(dim=0, keepdim=True)
        mean_scale = self.pv_scale.mean(dim=0, keepdim=True)
        mean_hand  = self.pv_hand.mean(dim=0, keepdim=True)
        mean_face  = self.pv_face.mean(dim=0, keepdim=True)

        loss = self.consistency_weight * (
            (self.pv_pose_cont - mean_pose).pow(2).mean()
            + (self.pv_shape   - mean_shape).pow(2).mean()
            + (self.pv_scale   - mean_scale).pow(2).mean()
            + (self.pv_hand    - mean_hand).pow(2).mean()
            + (self.pv_face    - mean_face).pow(2).mean()
        )
        return loss

    def prior_loss(self) -> Tensor:
        """Computes anchor prior loss."""
        loss = (
            0.5 * (self.pv_pose_cont - self.pose_cont_init).pow(2).mean()
            + 1e-2 * (self.cam_t_param - self.cam_t_init).pow(2).sum()
            + 1e-3 * (
                (self.pv_shape - self.inputs.shape).pow(2).mean()
                + (self.pv_scale - self.inputs.scale).pow(2).mean()
                + (self.pv_hand  - self.inputs.hand).pow(2).mean()
            )
        )
        return loss

    def total_loss(self, mhr_output: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        j3d_cam, _ = self.keypoints_camera(mhr_output)
        loss_kp, metrics = self.reprojection_loss(j3d_cam)
        loss_cons = self.consistency_loss()
        loss_prior = self.prior_loss()

        loss = loss_kp + loss_cons + loss_prior
        metrics.update({
            "loss_consistency": loss_cons,
            "loss_prior": loss_prior,
            "loss": loss,
        })
        return loss, metrics

    def _consensus_idx(self) -> int:
        features = torch.cat([self.pv_pose_cont, self.pv_shape, self.pv_scale, self.pv_hand, self.pv_face], dim=1)
        return medoid_index(features)

    def _snapshot_mhr_model_params(self, step: int, mhr_output: Dict[str, Tensor]) -> Tensor:
        if step == 0 and self.inputs.original_per_view_mhr_model_params is not None:
            params = self.inputs.original_per_view_mhr_model_params.to(
                device=self.device,
                dtype=mhr_output["model_params"].dtype,
            ).clone()
        else:
            params = mhr_output["model_params"].clone()
        params = pad_mhr_model_params(params)
        params[:, :6] = 0
        return params

    def _snapshot_cam_rot(self, step: int) -> Tensor:
        if step == 0:
            return self.inputs.cam_rot
        return rot6d_to_rotmat(self.cam_rot_param)

    @torch.no_grad()
    def snapshot(self, step: int) -> Dict:
        """Generates a snapshot for evaluation/visualization."""
        mhr_output = self.forward_mhr()
        j3d_cam, verts_cam = self.keypoints_camera(mhr_output)
        kp2d_fitted = project_keypoints_3d(j3d_cam, self.inputs.K_fit)

        cidx = self._consensus_idx()
        pv_mhr_params = self._snapshot_mhr_model_params(step, mhr_output)
        mhr_model_params_c = pv_mhr_params[cidx]

        root_6d_v = self.root_zero_6d[None].expand(self.V, -1)
        pose_raw_c = torch.cat([root_6d_v[cidx:cidx+1], self.pv_pose_cont[cidx:cidx+1]], dim=1)[0]
        cam_rot_snapshot = self._snapshot_cam_rot(step)

        return {
            "step": step,
            "body": {
                "pred_pose_raw":    pose_raw_c.detach().cpu(),
                "shape":            self.pv_shape[cidx].detach().cpu(),
                "scale":            self.pv_scale[cidx].detach().cpu(),
                "hand":             self.pv_hand[cidx].detach().cpu(),
                "face":             self.pv_face[cidx].detach().cpu(),
                "mhr_model_params": pad_mhr_model_params(mhr_model_params_c).detach().cpu(),
            },
            "per_view_body": {
                img_key: {
                    "pred_pose_raw":    torch.cat([root_6d_v[v:v+1], self.pv_pose_cont[v:v+1]], dim=1)[0].detach().cpu(),
                    "shape":            self.pv_shape[v].detach().cpu(),
                    "scale":            self.pv_scale[v].detach().cpu(),
                    "hand":             self.pv_hand[v].detach().cpu(),
                    "face":             self.pv_face[v].detach().cpu(),
                    "mhr_model_params": pad_mhr_model_params(pv_mhr_params[v]).detach().cpu(),
                } for v, img_key in enumerate(self.inputs.view_keys)
            },
            "per_view": self._per_view_cam_dict(cam_rot_snapshot),
            "kp_viz": self._kp_viz_dict(kp2d_fitted, verts_cam),
        }

    def fit(self, n_iters: int, freeze_cam_iters: int = 0, use_mesh_visibility: bool = True) -> Tuple[DictRecorder, List]:
        """Run the optimization loop.

        Args:
            freeze_cam_iters: freeze camera parameters for the first N iterations.
            use_mesh_visibility: use depth-based mesh occlusion to weight keypoints.
        Returns:
            (loss_record, snapshots)
        """
        base_lr = 0.01
        body_param_groups = [
            {'params': [self.pv_pose_cont], 'lr': base_lr},
            {'params': [self.pv_shape, self.pv_scale, self.pv_hand, self.pv_face], 'lr': base_lr * 0.01},
        ]

        cam_frozen = freeze_cam_iters > 0
        if cam_frozen:
            self.set_camera_requires_grad(False)
            optimizer = torch.optim.Adam(body_param_groups, eps=1e-8, betas=(0.9, 0.999))
        else:
            cam_param_groups = [
                {'params': [self.cam_rot_param, self.cam_trans_param], 'lr': base_lr * 0.1},
                {'params': [self.cam_t_param], 'lr': base_lr * 0.1},
            ]
            optimizer = torch.optim.Adam(body_param_groups + cam_param_groups, eps=1e-8, betas=(0.9, 0.999))

        loss_record = DictRecorder()
        self.compute_joint_visibility(use_mesh_visibility)
        snapshots = [self.snapshot(0)]

        with torch.enable_grad(), torch.amp.autocast(device_type=self.device.type, enabled=False):
            for step in range(n_iters):
                if cam_frozen and step == freeze_cam_iters:
                    self.set_camera_requires_grad(True)
                    optimizer.add_param_group({'params': [self.cam_rot_param, self.cam_trans_param], 'lr': 1e-2})
                    optimizer.add_param_group({'params': [self.cam_t_param], 'lr': 5e-3})
                    cam_frozen = False

                optimizer.zero_grad()
                mhr_output = self.forward_mhr()
                loss, metrics = self.total_loss(mhr_output)
                metrics["step"] = torch.as_tensor(step, device=self.device)
                metrics["pose_delta"] = (self.pv_pose_cont - self.pose_cont_init).norm()
                metrics["cam_delta"] = (self.cam_t_param - self.cam_t_init).norm()
                loss_record.add_dict(metrics)
                loss.backward()
                optimizer.step()

                step_idx = step + 1
                if step_idx % 10 == 0 or step_idx == n_iters:
                    snapshots.append(self.snapshot(step_idx))

        return loss_record, snapshots

    def result(self) -> Dict:
        snap = self.snapshot(-1)
        return {
            "body": snap["body"],
            "per_view_body": snap["per_view_body"],
            "per_view": snap["per_view"],
            "meta": {
                "view_keys": self.inputs.view_keys,
                "anchor_view_key": self.inputs.view_keys[self._consensus_idx()],
                "version": "refactored-pv-3",
            },
        }
