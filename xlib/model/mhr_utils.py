from typing import Optional, Tuple

import torch

from xlib.geometry.ops import quaternion_to_matrix


MHR_PARAM_LIST = [
    "root_tx",
    "root_ty",
    "root_tz",
    "root_rx",
    "root_ry",
    "root_rz",
    "spine0_rx_flexible",
    "spine_twist0",
    "spine0_ry_flexible",
    "spine_lean0",
    "spine0_rz_flexible",
    "spine_bend0",
    "spine1_rx_flexible",
    "spine_twist1",
    "spine1_ry_flexible",
    "spine_lean1",
    "spine1_rz_flexible",
    "spine_bend1",
    "spine2_rx_flexible",
    "spine2_ry_flexible",
    "spine2_rz_flexible",
    "spine3_rx_flexible",
    "spine3_ry_flexible",
    "spine3_rz_flexible",
    "neck_twist",
    "neck_lean",
    "neck_bend",
    "head_twist",
    "head_lean",
    "head_bend",
    "r_clavicle_rx",
    "r_clavicle_ry",
    "r_clavicle_rz",
    "r_uparm_twist",
    "r_uparm_ry",
    "r_uparm_rz",
    "r_elbow_bend",
    "r_lowarm_twist",
    "r_wrist_ry",
    "r_wrist_rz",
    "l_clavicle_rx",
    "l_clavicle_ry",
    "l_clavicle_rz",
    "l_uparm_twist",
    "l_uparm_ry",
    "l_uparm_rz",
    "l_elbow_bend",
    "l_lowarm_twist",
    "l_wrist_ry",
    "l_wrist_rz",
    "r_upleg_twist",
    "r_upleg_ry",
    "r_upleg_rz",
    "r_knee_bend",
    "r_lowleg_twist",
    "r_foot_bend",
    "r_foot_lean0",
    "r_foot_lean1",
    "r_ball_bend",
    "l_upleg_twist",
    "l_upleg_ry",
    "l_upleg_rz",
    "l_knee_bend",
    "l_lowleg_twist",
    "l_foot_bend",
    "l_foot_lean0",
    "l_foot_lean1",
    "l_ball_bend",
    "r_thumb0_ry",
    "r_thumb0_rz",
    "r_thumb1_rx",
    "r_thumb1_ry",
    "r_thumb1_rz",
    "r_thumb2_rz",
    "r_thumb3_rz",
    "r_index1_ry",
    "r_ring1_ry",
    "r_pinky1_ry",
    "r_middle1_ry",
    "r_index1_rz",
    "r_index2_rz",
    "r_index3_rz",
    "r_middle1_rz",
    "r_middle2_rz",
    "r_middle3_rz",
    "r_ring1_rz",
    "r_ring2_rz",
    "r_ring3_rz",
    "r_pinky1_rz",
    "r_pinky2_rz",
    "r_pinky3_rz",
    "r_index1_rx",
    "r_ring1_rx",
    "r_pinky1_rx",
    "r_middle1_rx",
    "l_thumb0_ry",
    "l_thumb0_rz",
    "l_thumb1_rx",
    "l_thumb1_ry",
    "l_thumb1_rz",
    "l_thumb2_rz",
    "l_thumb3_rz",
    "l_index1_ry",
    "l_ring1_ry",
    "l_pinky1_ry",
    "l_middle1_ry",
    "l_index1_rz",
    "l_index2_rz",
    "l_index3_rz",
    "l_middle1_rz",
    "l_middle2_rz",
    "l_middle3_rz",
    "l_ring1_rz",
    "l_ring2_rz",
    "l_ring3_rz",
    "l_pinky1_rz",
    "l_pinky2_rz",
    "l_pinky3_rz",
    "l_index1_rx",
    "l_ring1_rx",
    "l_pinky1_rx",
    "l_middle1_rx",
    "l_foot_ry_flexible",
    "l_subtalar_rz_flexible",
    "l_talocrural_rx_flexible",
    "l_ball_rx_flexible",
    "r_foot_ry_flexible",
    "r_subtalar_rz_flexible",
    "r_talocrural_rx_flexible",
    "r_ball_rx_flexible",
    "spine_length_flexible",
    "neck_length_flexible",
    "shoulder_width_flexible",
    "arm_length_flexible",
    "hip_width_flexible",
    "leg_length_flexible",
    "scale_eye_width",
    "scale_eye_height",
    "scale_eye_depth",
    "scale_spine_length",
    "scale_neck_length",
    "scale_shoulder_width",
    "scale_uparms",
    "scale_lowarms",
    "scale_r_hands",
    "scale_l_hands",
    "scale_hip_width",
    "scale_hip_height",
    "scale_hip_depth",
    "scale_uplegs",
    "scale_lowlegs",
    "scale_knee_knock",
    "scale_ankle_height",
    "scale_foot_length",
    "scale_r_index1_length",
    "scale_r_middle1_length",
    "scale_r_ring1_length",
    "scale_r_pinky1_length",
    "scale_r_thumb1_length",
    "scale_r_index1_offset",
    "scale_r_middle1_offset",
    "scale_r_ring1_offset",
    "scale_r_pinky1_offset",
    "scale_r_thumb1_offset",
    "scale_r_index2_length",
    "scale_r_middle2_length",
    "scale_r_ring2_length",
    "scale_r_pinky2_length",
    "scale_r_thumb2_length",
    "scale_r_index3_length",
    "scale_r_middle3_length",
    "scale_r_ring3_length",
    "scale_r_pinky3_length",
    "scale_r_thumb3_length",
    "scale_r_index_null_tx",
    "scale_r_middle_null_tx",
    "scale_r_ring_null_tx",
    "scale_r_pinky_null_tx",
    "scale_r_thumb_null_tx",
    "scale_l_index1_length",
    "scale_l_middle1_length",
    "scale_l_ring1_length",
    "scale_l_pinky1_length",
    "scale_l_thumb1_length",
    "scale_l_index1_offset",
    "scale_l_middle1_offset",
    "scale_l_ring1_offset",
    "scale_l_pinky1_offset",
    "scale_l_thumb1_offset",
    "scale_l_index2_length",
    "scale_l_middle2_length",
    "scale_l_ring2_length",
    "scale_l_pinky2_length",
    "scale_l_thumb2_length",
    "scale_l_index3_length",
    "scale_l_middle3_length",
    "scale_l_ring3_length",
    "scale_l_pinky3_length",
    "scale_l_thumb3_length",
    "scale_l_index_null_tx",
    "scale_l_middle_null_tx",
    "scale_l_ring_null_tx",
    "scale_l_pinky_null_tx",
    "scale_l_thumb_null_tx"
]


def _normalize_pose_param_dims(
    model_params: torch.Tensor,
    shape_params: Optional[torch.Tensor] = None,
    expr_params: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Normalize pose tensors to [B, V, D] for consistent conditioning logic."""
    if model_params.ndim == 2:
        model_params = model_params.unsqueeze(1)
    if shape_params is not None and shape_params.ndim == 2:
        shape_params = shape_params.unsqueeze(1)
    if expr_params is not None and expr_params.ndim == 2:
        expr_params = expr_params.unsqueeze(1)
    return model_params, shape_params, expr_params


def build_raw_pose_condition(
    model_params: torch.Tensor,
    shape_params: Optional[torch.Tensor] = None,
    expr_params: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Concatenate MHR pose components into [B, V, D_total] conditioning tensor."""
    model_params, shape_params, expr_params = _normalize_pose_param_dims(
        model_params=model_params,
        shape_params=shape_params,
        expr_params=expr_params,
    )
    parts = [model_params]
    if shape_params is not None:
        parts.append(shape_params)
    if expr_params is not None:
        parts.append(expr_params)
    return torch.cat(parts, dim=-1)


def mhr_model_params_to_joint_rot6d(
    model_params: torch.Tensor,
    mhr_animatable_gs,
) -> torch.Tensor:
    """Convert model params [B, D] to flattened per-joint 6D rotations [B, J*6]."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        model_params_f32 = model_params.float()
        joint_params = mhr_animatable_gs._compute_joint_parameters(model_params_f32)
        skel_state = mhr_animatable_gs._compute_skeleton_state(joint_params)
        quats = skel_state[..., 3:7]
        rot_mats = quaternion_to_matrix(quats)
        rot_6d = rot_mats[..., :2, :].reshape(model_params.shape[0], -1)
    return rot_6d.to(model_params.dtype)


def mhr_model_params_to_joint_pos_rot6d(
    model_params: torch.Tensor,
    mhr_animatable_gs,
) -> torch.Tensor:
    """Convert model params [B, D] to per-joint [pos(m), rot6d] features [B, J, 9]."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        model_params_f32 = model_params.float()
        joint_params = mhr_animatable_gs._compute_joint_parameters(model_params_f32)
        skel_state = mhr_animatable_gs._compute_skeleton_state(joint_params)
        positions = skel_state[..., :3] / 100.0
        quats = skel_state[..., 3:7]
        rot_mats = quaternion_to_matrix(quats)
        rot_6d = rot_mats[..., :2, :].reshape(*quats.shape[:-1], 6)
        per_joint_feat = torch.cat([positions, rot_6d], dim=-1)
    return per_joint_feat.to(model_params.dtype)