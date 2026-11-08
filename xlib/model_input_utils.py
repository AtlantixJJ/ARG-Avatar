from typing import Dict

import torch


def slice_batch(tensor_dict: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
    return {
        key: value[batch_idx:batch_idx + 1] if isinstance(value, torch.Tensor) else value
        for key, value in tensor_dict.items()
    }


def slice_batch_views(batch: Dict, mask: torch.Tensor) -> Dict:
    """Slice views in a batch based on a boolean mask."""
    if mask.ndim != 1:
        raise ValueError(f"Expected a 1D view mask, got shape {tuple(mask.shape)}")
    new_batch = {}
    mask_list = mask.tolist()
    for key, value in batch.items():
        if key in ("input_pose_dic", "input_view_mask"):
            continue
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == mask.shape[0]:
            new_batch[key] = value[:, mask]
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            new_batch[key] = [
                [item for item, keep in zip(sample, mask_list) if keep]
                for sample in value
            ]
        else:
            new_batch[key] = value
    if "input_pose_dic" in batch:
        new_batch["input_pose_dic"] = {
            key: value[:, mask]
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == mask.shape[0]
            else value
            for key, value in batch["input_pose_dic"].items()
        }
    return new_batch


def create_single_view_batch(batch: Dict, view_idx: int, device: torch.device) -> Dict:
    """Create a single-view model batch from (B, V, ...) format data."""
    B, _, _, H, W = batch["images"].shape
    P = 1

    img = batch["images"][:, view_idx:view_idx + 1]
    mask = batch["masks"][:, view_idx:view_idx + 1]
    cam_int = batch["K_proc"][:, view_idx].to(device)
    raw_cam_int = batch["K_raw"][:, view_idx].to(device)

    crop_bboxes = batch["crop_bboxes"][:, view_idx:view_idx + 1]
    raw_sizes = batch["raw_sizes"][:, view_idx:view_idx + 1]

    keys_to_check = [
        "bbox_center",
        "bbox_scale",
        "ori_img_size",
        "img_size",
        "affine_trans",
        "mask_score",
        "person_valid",
    ]
    assert all(k in batch for k in keys_to_check), f"{keys_to_check} must be in batch"

    bbox_center = batch["bbox_center"][:, view_idx:view_idx + 1]
    bbox_scale = batch["bbox_scale"][:, view_idx:view_idx + 1]
    ori_img_size = batch["ori_img_size"][:, view_idx:view_idx + 1]
    img_size = batch["img_size"][:, view_idx:view_idx + 1]
    affine_trans = batch["affine_trans"][:, view_idx:view_idx + 1]
    mask_score = batch["mask_score"][:, view_idx:view_idx + 1]
    person_valid = batch["person_valid"][:, view_idx:view_idx + 1]

    assert img.shape == (B, P, 3, H, W), f"Invalid img shape: {img.shape}"
    assert mask.shape == (B, P, 1, H, W), f"Invalid mask shape: {mask.shape}"
    assert crop_bboxes.shape == (B, P, 4), f"Invalid crop_bboxes shape: {crop_bboxes.shape}"
    assert raw_sizes.shape == (B, P, 2), f"Invalid raw_sizes shape: {raw_sizes.shape}"
    assert bbox_center.shape == (B, P, 2), f"Invalid bbox_center shape: {bbox_center.shape}"
    assert bbox_scale.shape == (B, P, 2), f"Invalid bbox_scale shape: {bbox_scale.shape}"
    assert ori_img_size.shape == (B, P, 2), f"Invalid ori_img_size shape: {ori_img_size.shape}"
    assert img_size.shape == (B, P, 2), f"Invalid img_size shape: {img_size.shape}"
    assert affine_trans.shape == (B, P, 2, 3), f"Invalid affine_trans shape: {affine_trans.shape}"
    assert mask_score.shape == (B, P), f"Invalid mask_score shape: {mask_score.shape}"
    assert person_valid.shape == (B, P), f"Invalid person_valid shape: {person_valid.shape}"

    single_batch = {
        "img": img.to(device),
        "mask": mask.float().to(device),
        "cam_int": raw_cam_int,
        "crop_cam_int": cam_int,
        "crop_bboxes": crop_bboxes.to(device),
        "raw_sizes": raw_sizes.to(device),
        "mask_score": mask_score.to(device),
        "bbox_center": bbox_center.to(device),
        "bbox_scale": bbox_scale.to(device),
        "ori_img_size": ori_img_size.to(device),
        "img_size": img_size.to(device),
        "affine_trans": affine_trans.to(device),
        "person_valid": person_valid.to(device),
    }

    if "hr_images" in batch:
        single_batch["hr_images"] = batch["hr_images"][:, view_idx:view_idx + 1].to(device)

    return single_batch
