import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from torch import Tensor
import math

from xlib.fitting.pose_optimizers import BaseMHRPoseOptimizer
from xlib.fitting.pose_fit_common import (
    PoseFitInputs,
    decode_pose_raw,
    rotmat_to_rot6d,
    rot6d_to_rotmat,
    pad_mhr_model_params,
    project_keypoints_3d,
)
from xlib.utils import DictRecorder

def geodesic_distance_sq(R1: Tensor, R2: Tensor) -> Tensor:
    """Computes the squared geodesic distance between two rotation matrices."""
    dR = torch.bmm(R1.transpose(1, 2), R2)
    tr = torch.diagonal(dR, dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6))
    return theta.pow(2)

class TemporalMHRPoseOptimizer(BaseMHRPoseOptimizer):
    def __init__(
        self, 
        mhr_model: nn.Module, 
        inputs: PoseFitInputs, 
        frame_weights: Tensor,
        frame_accept_mask: Tensor,
        lambda_v: float = 1.0,
        lambda_a: float = 1.0,
    ):
        super().__init__(mhr_model, inputs)
        
        self.T = self.V
        self.frame_weights = frame_weights.to(self.device).view(self.T, 1, 1)
        self.frame_accept_mask = frame_accept_mask.to(self.device)
        self.lambda_v = lambda_v
        self.lambda_a = lambda_a
        
        # 1. Initialize per-frame body parameters from their own predictions
        self.pose_cont = nn.Parameter(inputs.pose_raw[:, 6:].clone()) # [T, 260]
        
        # 2. Shared shape and scale over the window
        accepted_shape = inputs.shape[self.frame_accept_mask]
        accepted_scale = inputs.scale[self.frame_accept_mask]
        
        if len(accepted_shape) > 0:
            init_shape = accepted_shape.median(dim=0).values
            init_scale = accepted_scale.median(dim=0).values
        else:
            init_shape = inputs.shape.mean(dim=0)
            init_scale = inputs.scale.mean(dim=0)
            
        self.shape = nn.Parameter(init_shape) # [45]
        self.scale = nn.Parameter(init_scale) # [28]
        
        self.hand = nn.Parameter(inputs.hand.clone()) # [T, 108]
        self.face = nn.Parameter(inputs.face.clone()) # [T, 72]
        
        # 3. Constants for priors
        self.pose_cont_init = self.pose_cont.detach().clone()
        self.shape_init = self.shape.detach().clone()
        self.scale_init = self.scale.detach().clone()
        
    @torch.no_grad()
    def compute_joint_visibility(self, use_mesh_visibility: bool = True) -> None:
        super().compute_joint_visibility(use_mesh_visibility)
        # Scale visibility weights by Stage 0 frame_weights (which bakes in the gating criteria)
        # self.joint_visibility_weight is [V, K] or [V, K, 1], frame_weights is [V, 1, 1]
        w = self.frame_weights.view(self.T, 1)
        self.joint_visibility_weight = self.joint_visibility_weight * w

    def forward_mhr(self) -> Dict[str, Tensor]:
        root_6d_t = self.root_zero_6d[None].expand(self.T, -1)
        pose_raw_t = torch.cat([root_6d_t, self.pose_cont], dim=1)
        g_euler, body_params = decode_pose_raw(pose_raw_t)
        
        mhr_forward_fn = self.mhr_model.mhr_forward if hasattr(self.mhr_model, "mhr_forward") else self.mhr_model
        
        output = mhr_forward_fn(
            global_trans=torch.zeros_like(g_euler),
            global_rot=g_euler,
            body_pose_params=body_params,
            hand_pose_params=self.hand,
            scale_params=self.scale[None].expand(self.T, -1),
            shape_params=self.shape[None].expand(self.T, -1),
            expr_params=self.face,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
        )
        return {
            "verts": output[0],
            "kp3d": output[1],
            "model_params": output[3],
        }

    def total_loss(self, mhr_output: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        j3d_cam, _ = self.keypoints_camera(mhr_output)
        loss_kp, metrics = self.reprojection_loss(j3d_cam)
        
        # Identity prior
        loss_id = 1e-3 * (
            (self.shape - self.shape_init).pow(2).sum() +
            (self.scale - self.scale_init).pow(2).sum()
        )
        
        # Pose prior on accepted frames
        accepted = self.frame_accept_mask
        if accepted.any():
            loss_pose_prior = 0.5 * (self.pose_cont[accepted] - self.pose_cont_init[accepted]).pow(2).mean()
        else:
            loss_pose_prior = torch.tensor(0.0, device=self.device)
            
        # Velocity and Acceleration
        loss_vel = torch.tensor(0.0, device=self.device)
        loss_acc = torch.tensor(0.0, device=self.device)
        
        if self.T > 1:
            # Velocity
            v_pose = (self.pose_cont[1:] - self.pose_cont[:-1]).pow(2).mean()
            v_cam_trans = (self.cam_trans_param[1:] - self.cam_trans_param[:-1]).pow(2).mean()
            v_cam_t = (self.cam_t_param[1:] - self.cam_t_param[:-1]).pow(2).mean()
            
            R_t = rot6d_to_rotmat(self.cam_rot_param)
            v_rot = geodesic_distance_sq(R_t[:-1], R_t[1:]).mean()
            
            loss_vel = self.lambda_v * (v_pose + v_cam_trans + v_cam_t + v_rot)
            
            # Acceleration
            if self.T > 2:
                a_pose = (self.pose_cont[2:] - 2*self.pose_cont[1:-1] + self.pose_cont[:-2]).pow(2).mean()
                a_cam_trans = (self.cam_trans_param[2:] - 2*self.cam_trans_param[1:-1] + self.cam_trans_param[:-2]).pow(2).mean()
                a_cam_t = (self.cam_t_param[2:] - 2*self.cam_t_param[1:-1] + self.cam_t_param[:-2]).pow(2).mean()
                a_rot = (R_t[2:] - 2*R_t[1:-1] + R_t[:-2]).pow(2).mean() # Frobenius norm approx
                
                loss_acc = self.lambda_a * (a_pose + a_cam_trans + a_cam_t + a_rot)
                
        loss = loss_kp + loss_id + loss_pose_prior + loss_vel + loss_acc
        
        metrics.update({
            "loss_kp": loss_kp,
            "loss_id": loss_id,
            "loss_pose_prior": loss_pose_prior,
            "loss_vel": loss_vel,
            "loss_acc": loss_acc,
            "loss": loss,
        })
        return loss, metrics

    def fit(self, n_iters: int, freeze_cam_iters: int = 100, use_mesh_visibility: bool = True) -> Tuple[DictRecorder, List]:
        # Using the same scheduling as described in the plan
        base_lr = 0.01
        
        body_param_groups = [
            {'params': [self.pose_cont], 'lr': base_lr},
            {'params': [self.shape, self.scale, self.hand, self.face], 'lr': base_lr * 0.01},
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
        # snapshots disabled to save memory, or we just save the final result
        
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
                loss_record.add_dict(metrics)
                loss.backward()
                optimizer.step()

        return loss_record, []
        
    def result(self) -> Dict:
        """Fetch the final parameters from the optimizer."""
        mhr_output = self.forward_mhr()
        j3d_cam, verts_cam = self.keypoints_camera(mhr_output)
        
        pv_mhr_params = mhr_output["model_params"].clone()
        pv_mhr_params = pad_mhr_model_params(pv_mhr_params)
        pv_mhr_params[:, :6] = 0
        
        root_6d_v = self.root_zero_6d[None].expand(self.T, -1)
        
        cam_rot_snapshot = rot6d_to_rotmat(self.cam_rot_param)
        
        return {
            "per_view_body": {
                img_key: {
                    "pred_pose_raw": torch.cat([root_6d_v[v:v+1], self.pose_cont[v:v+1]], dim=1)[0].detach().cpu(),
                    "shape": self.shape.detach().cpu(),
                    "scale": self.scale.detach().cpu(),
                    "hand": self.hand[v].detach().cpu(),
                    "face": self.face[v].detach().cpu(),
                    "mhr_model_params": pv_mhr_params[v].detach().cpu(),
                } for v, img_key in enumerate(self.inputs.view_keys)
            },
            "per_view": {
                img_key: {
                    "pred_cam_t": self.cam_t_param[v].detach().cpu(),
                    "cam_rot": cam_rot_snapshot[v].detach().cpu(),
                    "cam_trans": self.cam_trans_param[v].detach().cpu(),
                } for v, img_key in enumerate(self.inputs.view_keys)
            }
        }

def get_cosine_blend_weights(window_size: int, overlap: int, is_first: bool, is_last: bool, device: torch.device) -> Tensor:
    weights = torch.ones(window_size, device=device)
    if not is_first and overlap > 0:
        x = torch.linspace(0, math.pi, overlap, device=device)
        weights[:overlap] = (1 - torch.cos(x)) / 2.0
    if not is_last and overlap > 0:
        x = torch.linspace(0, math.pi, overlap, device=device)
        weights[-overlap:] = (1 + torch.cos(x)) / 2.0
    return weights

def fit_motion_windows(
    mhr_model: nn.Module,
    inputs: PoseFitInputs,
    frame_weights: Tensor,
    frame_accept_mask: Tensor,
    window_size: int = 256,
    overlap: int = 32,
    n_iters: int = 300,
    lambda_v: float = 1.0,
    lambda_a: float = 1.0
) -> Dict[str, Dict]:
    T = inputs.pose_raw.shape[0]
    stride = window_size - overlap
    num_windows = math.ceil((T - overlap) / stride) if T > overlap else 1
    
    # 1. Global shape/scale init using subset
    device = inputs.pose_raw.device
    global_shape = None
    global_scale = None
    
    accepted_indices = torch.where(frame_accept_mask)[0]
    if len(accepted_indices) > 0:
        step = max(1, len(accepted_indices) // 200)
        subset_indices = accepted_indices[::step][:200]
        
        subset_shape = inputs.shape[subset_indices]
        subset_scale = inputs.scale[subset_indices]
        global_shape = subset_shape.median(dim=0).values
        global_scale = subset_scale.median(dim=0).values
    
    results = []
    
    for i in range(num_windows):
        start = i * stride
        end = min(start + window_size, T)
        curr_window_size = end - start
        
        print(f"Fitting window {i+1}/{num_windows} [{start}:{end}]")
        
        # Extract inputs for this window
        win_inputs = PoseFitInputs(
            view_keys=inputs.view_keys[start:end],
            pose_raw=inputs.pose_raw[start:end],
            shape=inputs.shape[start:end],
            scale=inputs.scale[start:end],
            hand=inputs.hand[start:end],
            face=inputs.face[start:end],
            K_fit=inputs.K_fit[start:end],
            fit_size=inputs.fit_size[start:end],
            kp2d_target=inputs.kp2d_target[start:end],
            cam_rot=inputs.cam_rot[start:end],
            cam_trans=inputs.cam_trans[start:end],
            cam_t=inputs.cam_t[start:end],
            original_per_view_mhr_model_params=inputs.original_per_view_mhr_model_params[start:end] if inputs.original_per_view_mhr_model_params is not None else None,
        )
        
        win_weights = frame_weights[start:end]
        win_accept = frame_accept_mask[start:end]
        
        opt = TemporalMHRPoseOptimizer(
            mhr_model, win_inputs, win_weights, win_accept, lambda_v, lambda_a
        )
        if global_shape is not None:
            opt.shape.data.copy_(global_shape)
            opt.scale.data.copy_(global_scale)
            
        opt.fit(n_iters, freeze_cam_iters=min(100, n_iters // 2), use_mesh_visibility=True)
        results.append((start, end, opt.result()))
        
    # Blending
    final_result = {
        "per_view_body": {},
        "per_view": {}
    }
    
    blend_weights = torch.zeros(T, device=device)
    blended_mhr_params = torch.zeros(T, 204, device=device)
    blended_cam_t = torch.zeros(T, 3, device=device)
    blended_cam_trans = torch.zeros(T, 3, device=device)
    blended_pred_pose_raw = torch.zeros(T, 266, device=device)
    blended_hand = torch.zeros(T, 108, device=device)
    blended_face = torch.zeros(T, 72, device=device)
    
    # We also need to blend cam_rot using quats to be safe, or assume small differences
    blended_cam_rot_quats = torch.zeros(T, 4, device=device)
    
    from xlib.fitting.motion_filters import rotmat_to_quat, quat_to_rotmat
    
    for i, (start, end, res) in enumerate(results):
        curr_window_size = end - start
        is_first = (i == 0)
        is_last = (i == num_windows - 1)
        w = get_cosine_blend_weights(curr_window_size, overlap, is_first, is_last, device)
        
        for v_idx, img_key in enumerate(res["per_view"].keys()):
            global_idx = start + v_idx
            wv = w[v_idx]
            
            blend_weights[global_idx] += wv
            
            body = res["per_view_body"][img_key]
            cam = res["per_view"][img_key]
            
            blended_mhr_params[global_idx] += body["mhr_model_params"].to(device) * wv
            blended_pred_pose_raw[global_idx] += body["pred_pose_raw"].to(device) * wv
            blended_hand[global_idx] += body["hand"].to(device) * wv
            blended_face[global_idx] += body["face"].to(device) * wv
            
            blended_cam_t[global_idx] += cam["pred_cam_t"].to(device) * wv
            blended_cam_trans[global_idx] += cam["cam_trans"].to(device) * wv
            
            quat = rotmat_to_quat(cam["cam_rot"].to(device).unsqueeze(0))[0]
            # align quats to avoid negating
            if i > 0 and v_idx < overlap:
                # check dot product with existing
                existing_quat = blended_cam_rot_quats[global_idx] / blend_weights[global_idx].clamp(min=1e-6)
                if torch.dot(existing_quat, quat) < 0:
                    quat = -quat
            blended_cam_rot_quats[global_idx] += quat * wv
            
            if "shape" not in final_result:
                final_result["shape"] = body["shape"]
                final_result["scale"] = body["scale"]

    # Normalize
    for t in range(T):
        bw = blend_weights[t]
        if bw > 0:
            blended_mhr_params[t] /= bw
            blended_pred_pose_raw[t] /= bw
            blended_hand[t] /= bw
            blended_face[t] /= bw
            blended_cam_t[t] /= bw
            blended_cam_trans[t] /= bw
            blended_cam_rot_quats[t] /= bw
            
            # normalize quat
            blended_cam_rot_quats[t] = blended_cam_rot_quats[t] / torch.norm(blended_cam_rot_quats[t])

    blended_cam_rot = quat_to_rotmat(blended_cam_rot_quats)
    
    # Pack into result
    for t in range(T):
        img_key = inputs.view_keys[t]
        final_result["per_view_body"][img_key] = {
            "mhr_model_params": blended_mhr_params[t].cpu(),
            "pred_pose_raw": blended_pred_pose_raw[t].cpu(),
            "hand": blended_hand[t].cpu(),
            "face": blended_face[t].cpu(),
            "shape": final_result["shape"],
            "scale": final_result["scale"]
        }
        final_result["per_view"][img_key] = {
            "pred_cam_t": blended_cam_t[t].cpu(),
            "cam_trans": blended_cam_trans[t].cpu(),
            "cam_rot": blended_cam_rot[t].cpu()
        }
        
    return final_result
