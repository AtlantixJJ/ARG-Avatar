from typing import Any, Dict, List, Tuple
import torch
import numpy as np
import roma
from PIL import Image

from xlib.model.sam_3d_body.data.utils.prepare_batch import prepare_batch
from xlib.model.sam_3d_body.utils import recursive_to
from xlib.model.sam_3d_body.models.modules.mhr_utils import rotation_angle_difference, fix_wrist_euler


def _stack_hand_outputs(outputs: List[Dict]) -> Dict:
    """Stack hand decoder outputs from multiple batch items."""
    stacked = {}
    for key in outputs[0].keys():
        if isinstance(outputs[0][key], dict):
            stacked[key] = {}
            for subkey in outputs[0][key].keys():
                tensors = [o[key][subkey] for o in outputs]
                if isinstance(tensors[0], torch.Tensor):
                    stacked[key][subkey] = torch.cat(tensors, dim=0)
                else:
                    stacked[key][subkey] = tensors
        elif isinstance(outputs[0][key], torch.Tensor):
            stacked[key] = torch.cat([o[key] for o in outputs], dim=0)
        else:
            stacked[key] = [o[key] for o in outputs]
    return stacked


def _stack_hand_batches(batches: List[Dict]) -> Dict:
    """Stack hand batch dictionaries from multiple batch items."""
    stacked = {}
    for key in batches[0].keys():
        tensors = [b[key] for b in batches]
        if isinstance(tensors[0], torch.Tensor):
            stacked[key] = torch.cat(tensors, dim=0)
        else:
            stacked[key] = tensors
    return stacked


class HandRefinementMixin:
    """
    Mixin class for hand refinement logic.
    Separated from ARGNetBase to improve code organization.
    """

    def _transform_bbox_proc_to_raw(
        self,
        bbox_proc: np.ndarray,  # [B, 4] xyxy in processed image space
        crop_bboxes: torch.Tensor,  # [B, 1, 4] or [B, 4]
        proc_H: int,
        proc_W: int,
    ) -> np.ndarray:
        """
        Transform bounding box from processed image space to raw image space.

        Args:
            bbox_proc: [B, 4] xyxy bounding box in processed image coordinates
            crop_bboxes: [B, 4] crop region (x_min, y_min, x_max, y_max) in raw image
            proc_H, proc_W: processed image dimensions

        Returns:
            bbox_raw: [B, 4] xyxy bounding box in raw image coordinates
        """
        if crop_bboxes.ndim == 3:
            crop_bboxes = crop_bboxes.squeeze(1)  # [B, 4]

        crop_bboxes_np = crop_bboxes.cpu().numpy()
        B = bbox_proc.shape[0]
        bbox_raw = np.zeros_like(bbox_proc)

        for i in range(B):
            crop_x_min = crop_bboxes_np[i, 0]
            crop_y_min = crop_bboxes_np[i, 1]
            crop_x_max = crop_bboxes_np[i, 2]
            crop_y_max = crop_bboxes_np[i, 3]

            crop_w = crop_x_max - crop_x_min
            crop_h = crop_y_max - crop_y_min

            scale_x = crop_w / proc_W
            scale_y = crop_h / proc_H

            # Transform: proc -> raw
            bbox_raw[i, 0] = bbox_proc[i, 0] * scale_x + crop_x_min  # x1
            bbox_raw[i, 1] = bbox_proc[i, 1] * scale_y + crop_y_min  # y1
            bbox_raw[i, 2] = bbox_proc[i, 2] * scale_x + crop_x_min  # x2
            bbox_raw[i, 3] = bbox_proc[i, 3] * scale_y + crop_y_min  # y2

        return bbox_raw

    @torch.no_grad()
    def _forward_hand_decoders(
        self,
        raw_images: List[Image.Image],
        crop_bboxes: torch.Tensor,
        proc_H: int,
        proc_W: int,
        batch: Dict,
        pose_output: Dict,
        transform_hand: Any,
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Forward hand decoders using raw images for better hand detail.

        Unlike the original _forward_hand_decoders which processes a single image,
        this method processes each image in the batch using raw images.
        """
        B = len(raw_images)
        cam_int = batch["cam_int"].clone()

        # 1. Get hand bboxes in processed image space
        left_xyxy_proc, right_xyxy_proc = self._get_hand_box(pose_output, batch)

        # 2. Transform bboxes from processed to raw image space
        left_xyxy_raw = self._transform_bbox_proc_to_raw(
            left_xyxy_proc, crop_bboxes, proc_H, proc_W
        )
        right_xyxy_raw = self._transform_bbox_proc_to_raw(
            right_xyxy_proc, crop_bboxes, proc_H, proc_W
        )

        # 3. Process each image in the batch
        lhand_outputs, rhand_outputs = [], []
        batch_lhands, batch_rhands = [], []

        for b_idx in range(B):
            # Get raw image as numpy
            raw_img_np = np.asarray(raw_images[b_idx])
            raw_H, raw_W = raw_img_np.shape[:2]

            # Left hand - flip image & box
            flipped_img = raw_img_np[:, ::-1]
            left_box = left_xyxy_raw[b_idx:b_idx+1].copy()
            tmp = left_box.copy()
            left_box[:, 0] = raw_W - tmp[:, 2] - 1
            left_box[:, 2] = raw_W - tmp[:, 0] - 1

            batch_lhand = prepare_batch(
                flipped_img, transform_hand, left_box,
                cam_int=cam_int[b_idx:b_idx+1].clone()
            )
            batch_lhand = recursive_to(batch_lhand, "cuda")
            lhand_output = self.forward_step(batch_lhand, decoder_type="hand")

            # Unflip left hand output
            self._unflip_left_hand_output(lhand_output, batch_lhand, raw_W)

            # Right hand
            right_box = right_xyxy_raw[b_idx:b_idx+1]
            batch_rhand = prepare_batch(
                raw_img_np, transform_hand, right_box,
                cam_int=cam_int[b_idx:b_idx+1].clone()
            )
            batch_rhand = recursive_to(batch_rhand, "cuda")
            rhand_output = self.forward_step(batch_rhand, decoder_type="hand")

            lhand_outputs.append(lhand_output)
            rhand_outputs.append(rhand_output)
            batch_lhands.append(batch_lhand)
            batch_rhands.append(batch_rhand)

        # 4. Stack outputs across batch (use global helper functions)
        lhand_output = _stack_hand_outputs(lhand_outputs)
        rhand_output = _stack_hand_outputs(rhand_outputs)
        batch_lhand = _stack_hand_batches(batch_lhands)
        batch_rhand = _stack_hand_batches(batch_rhands)

        return lhand_output, rhand_output, batch_lhand, batch_rhand

    def _unflip_left_hand_output(
        self,
        lhand_output: Dict,
        batch_lhand: Dict,
        width: int,
    ):
        """Unflip left hand decoder output (modifies in place)."""
        scale_r_hands_mean = self.head_pose.scale_mean[8].item()
        scale_l_hands_mean = self.head_pose.scale_mean[9].item()
        scale_r_hands_std = self.head_pose.scale_comps[8, 8].item()
        scale_l_hands_std = self.head_pose.scale_comps[9, 9].item()

        lhand_output["mhr_hand"]["scale"][:, 9] = (
            (scale_r_hands_mean + scale_r_hands_std * lhand_output["mhr_hand"]["scale"][:, 8])
            - scale_l_hands_mean
        ) / scale_l_hands_std

        lhand_output["mhr_hand"]["joint_global_rots"][:, 78] = (
            lhand_output["mhr_hand"]["joint_global_rots"][:, 42].clone()
        )
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78, [1, 2], :] *= -1

        lhand_output["mhr_hand"]["hand"][:, :54] = lhand_output["mhr_hand"]["hand"][:, 54:]

        batch_lhand["bbox_center"][:, :, 0] = width - batch_lhand["bbox_center"][:, :, 0] - 1

    def _compute_hand_validity_mask(
        self,
        pose_output: Dict,
        lhand_output: Dict,
        rhand_output: Dict,
        batch_lhand: Dict,
        batch_rhand: Dict,
        width: int,
        thresh_wrist_angle: float = 1.4,
    ) -> torch.Tensor:
        """Compute validity mask for hand poses using 4 criteria."""
        ori_local_wrist_rotmat = roma.euler_to_rotmat(
            "XZY",
            pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3)),
        )

        angle_diff_mask = self._check_wrist_angle_criterion(
            pose_output, lhand_output, rhand_output, ori_local_wrist_rotmat, thresh_wrist_angle
        )

        hand_box_size_thresh = 64
        box_size_mask = torch.stack(
            [
                (batch_lhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
                (batch_rhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
            ],
            dim=1,
        )

        hand_kps2d_thresh = 0.5
        kps2d_mask = torch.stack(
            [
                lhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
                rhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
            ],
            dim=1,
        )

        wrist_dist_mask = self._check_wrist_distance_criterion(
            pose_output, lhand_output, rhand_output, batch_lhand, batch_rhand, width
        )

        return angle_diff_mask & box_size_mask & kps2d_mask & wrist_dist_mask

    def _check_wrist_angle_criterion(
        self,
        pose_output: Dict,
        lhand_output: Dict,
        rhand_output: Dict,
        ori_local_wrist_rotmat: torch.Tensor,
        thresh_wrist_angle: float,
    ) -> torch.Tensor:
        """Check wrist angle difference criterion."""
        joint_rotations = pose_output["mhr"]["joint_global_rots"]
        lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]

        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()
        wrist_zero_rot_pose = (
            lowarm_joint_rotations @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )

        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [left_joint_global_rots[:, 78], right_joint_global_rots[:, 42]], dim=1
        )

        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        angle_difference = rotation_angle_difference(ori_local_wrist_rotmat, fused_local_wrist_rotmat)

        return angle_difference < thresh_wrist_angle

    def _check_wrist_distance_criterion(
        self,
        pose_output: Dict,
        lhand_output: Dict,
        rhand_output: Dict,
        batch_lhand: Dict,
        batch_rhand: Dict,
        width: int,
    ) -> torch.Tensor:
        """Check 2D wrist distance criterion."""
        hand_wrist_kps2d_thresh = 0.25
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62

        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][:, [kps_right_wrist_idx]].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][:, [kps_right_wrist_idx]].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1

        body_right_kps_full = pose_output["mhr"]["pred_keypoints_2d"][:, [kps_right_wrist_idx]].clone()
        body_left_kps_full = pose_output["mhr"]["pred_keypoints_2d"][:, [kps_left_wrist_idx]].clone()

        right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(dim=-1) \
            / batch_lhand["bbox_scale"].flatten(0, 1)[:, 0]
        left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(dim=-1) \
            / batch_rhand["bbox_scale"].flatten(0, 1)[:, 0]

        return torch.stack(
            [left_kps_dist < hand_wrist_kps2d_thresh, right_kps_dist < hand_wrist_kps2d_thresh],
            dim=1,
        )

    def _merge_hand_poses(
        self,
        pose_output: Dict,
        lhand_output: Dict,
        rhand_output: Dict,
        hand_valid_mask: torch.Tensor,
        thresh_wrist_angle: float = 1.4,
    ) -> Dict:
        """Merge hand decoder outputs into body pose output."""
        left_hand_pose = lhand_output["mhr_hand"]["hand"][:, :54]
        right_hand_pose = rhand_output["mhr_hand"]["hand"][:, 54:]
        updated_hand_pose = torch.cat([left_hand_pose, right_hand_pose], dim=1)

        updated_scale = pose_output["mhr"]["scale"].clone()
        updated_scale[:, 9] = lhand_output["mhr_hand"]["scale"][:, 9]
        updated_scale[:, 8] = rhand_output["mhr_hand"]["scale"][:, 8]
        updated_scale[:, 18:] = (
            lhand_output["mhr_hand"]["scale"][:, 18:]
            + rhand_output["mhr_hand"]["scale"][:, 18:]
        ) / 2

        updated_shape = pose_output["mhr"]["shape"].clone()
        updated_shape[:, 40:] = (
            lhand_output["mhr_hand"]["shape"][:, 40:]
            + rhand_output["mhr_hand"]["shape"][:, 40:]
        ) / 2

        joint_rotations = self.head_pose.mhr_forward(
            global_trans=pose_output["mhr"]["global_rot"] * 0,
            global_rot=pose_output["mhr"]["global_rot"],
            body_pose_params=pose_output["mhr"]["body_pose"],
            hand_pose_params=updated_hand_pose,
            scale_params=updated_scale,
            shape_params=updated_shape,
            expr_params=pose_output["mhr"]["face"],
            return_joint_rotations=True,
        )[1]

        lowarm_joint_idxs = torch.LongTensor([76, 40]).cuda()
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]

        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).cuda()
        wrist_zero_rot_pose = (
            lowarm_joint_rotations @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )

        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [left_joint_global_rots[:, 78], right_joint_global_rots[:, 42]], dim=1
        )

        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        wrist_xzy = fix_wrist_euler(roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat))

        ori_local_wrist_rotmat = roma.euler_to_rotmat(
            "XZY",
            pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3)),
        )
        angle_difference = rotation_angle_difference(ori_local_wrist_rotmat, fused_local_wrist_rotmat)
        valid_angle = (angle_difference < thresh_wrist_angle) & hand_valid_mask
        valid_angle = valid_angle.unsqueeze(-1)

        body_pose = pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(1, (2, 3))
        updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
        pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]] = updated_body_pose.flatten(1, 2)

        hand_pose = pose_output["mhr"]["hand"].unflatten(1, (2, 54))
        pose_output["mhr"]["hand"] = torch.where(
            valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose
        ).flatten(1, 2)

        hand_scale = torch.stack(
            [pose_output["mhr"]["scale"][:, 9], pose_output["mhr"]["scale"][:, 8]], dim=1
        )
        updated_hand_scale = torch.stack([updated_scale[:, 9], updated_scale[:, 8]], dim=1)
        masked_hand_scale = torch.where(valid_angle.squeeze(-1), updated_hand_scale, hand_scale)
        pose_output["mhr"]["scale"][:, 9] = masked_hand_scale[:, 0]
        pose_output["mhr"]["scale"][:, 8] = masked_hand_scale[:, 1]

        valid_count = valid_angle.squeeze(-1).sum(dim=1, keepdim=True)
        pose_output["mhr"]["scale"][:, 18:] = torch.where(
            valid_count > 0,
            (lhand_output["mhr_hand"]["scale"][:, 18:] * valid_angle.squeeze(-1)[:, [0]]
             + rhand_output["mhr_hand"]["scale"][:, 18:] * valid_angle.squeeze(-1)[:, [1]])
            / (valid_count + 1e-8),
            pose_output["mhr"]["scale"][:, 18:],
        )
        pose_output["mhr"]["shape"][:, 40:] = torch.where(
            valid_count > 0,
            (lhand_output["mhr_hand"]["shape"][:, 40:] * valid_angle.squeeze(-1)[:, [0]]
             + rhand_output["mhr_hand"]["shape"][:, 40:] * valid_angle.squeeze(-1)[:, [1]])
            / (valid_count + 1e-8),
            pose_output["mhr"]["shape"][:, 40:],
        )

        return pose_output

    def _finalize_pose_output(
        self,
        pose_output: Dict,
        height: int,
        width: int,
    ) -> Dict:
        """Re-run MHR forward with merged params and project to 2D."""
        with torch.no_grad():
            verts, j3d, jcoords, mhr_model_params, joint_global_rots = (
                self.head_pose.mhr_forward(
                    global_trans=pose_output["mhr"]["global_rot"] * 0,
                    global_rot=pose_output["mhr"]["global_rot"],
                    body_pose_params=pose_output["mhr"]["body_pose"],
                    hand_pose_params=pose_output["mhr"]["hand"],
                    scale_params=pose_output["mhr"]["scale"],
                    shape_params=pose_output["mhr"]["shape"],
                    expr_params=pose_output["mhr"]["face"],
                    return_keypoints=True,
                    return_joint_coords=True,
                    return_model_params=True,
                    return_joint_rotations=True,
                )
            )
            j3d = j3d[:, :70]
            verts[..., [1, 2]] *= -1
            j3d[..., [1, 2]] *= -1
            jcoords[..., [1, 2]] *= -1

            pose_output["mhr"]["pred_keypoints_3d"] = j3d
            pose_output["mhr"]["pred_vertices"] = verts
            pose_output["mhr"]["pred_joint_coords"] = jcoords
            pose_output["mhr"]["mhr_model_params"] = mhr_model_params

        pred_keypoints_3d_proj = (
            pose_output["mhr"]["pred_keypoints_3d"] + pose_output["mhr"]["pred_cam_t"][:, None, :]
        )
        pred_keypoints_3d_proj[:, :, [0, 1]] *= pose_output["mhr"]["focal_length"][:, None, None]
        pred_keypoints_3d_proj[:, :, [0, 1]] = (
            pred_keypoints_3d_proj[:, :, [0, 1]]
            + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[None, None, :]
            * pred_keypoints_3d_proj[:, :, [2]]
        )
        pred_keypoints_3d_proj[:, :, :2] = (
            pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
        )
        pose_output["mhr"]["pred_keypoints_2d"] = pred_keypoints_3d_proj[:, :, :2]

        return pose_output
