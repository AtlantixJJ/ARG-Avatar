"""
Common utility functions for dataset classes.

This module contains utility functions shared across different dataset implementations,
including statistical computations, feature processing, and camera/geometry utilities.
"""

import torch
import numpy as np
import os
import glob
from typing import Optional, List, Tuple, Any
from PIL import Image

def combine_stds(
    means: torch.Tensor,
    stds: torch.Tensor,
    counts: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Combine standard deviations of multiple groups using the pooled standard deviation formula.

    This function properly combines standard deviations from different groups by accounting
    for both within-group variance and between-group variance.

    Args:
        means: Tensor of shape [N, ...] containing the mean for each group
        stds: Tensor of shape [N, ...] containing the standard deviation for each group
        counts: Optional tensor of shape [N] containing the count for each group.
                If None, assumes equal counts for all groups.

    Returns:
        Combined standard deviation tensor of shape [...] (same as single group)

    Example:
        >>> means = torch.tensor([[1.0, 2.0], [1.5, 2.5]])
        >>> stds = torch.tensor([[0.1, 0.2], [0.15, 0.25]])
        >>> combined_std = combine_stds(means, stds)
    """
    if counts is None:
        # Assume equal weight if counts not provided
        counts = torch.ones(stds.shape[:1]).to(stds)
    else:
        counts = torch.tensor(counts, dtype=torch.float32, device=stds.device)

    dims = [1] * (means.ndim - 1)
    counts = counts.view(-1, *dims)

    # Total number of samples
    N = counts.sum()

    # Calculate weighted mean
    weighted_mean = (means * counts).sum(0) / N

    # Calculate combined variance (within-group + between-group)
    within_group_var = ((stds ** 2) * counts).sum(0) / N
    between_group_var = ((means - weighted_mean)**2 * counts).sum(0) / N

    # Combined standard deviation
    return torch.sqrt(within_group_var + between_group_var)


def dic_to_compact(stats_dic: dict) -> List[torch.Tensor]:
    """
    Convert a statistics dictionary to a compact feature representation.

    This function takes a dictionary containing Gaussian splatting statistics
    (mean, std, max) and converts them into a compact list of tensors suitable
    for normalization purposes.

    Args:
        stats_dic: Dictionary with structure:
            {
                'offset_xyz' or 'xyz': {'mean': ..., 'std': ..., 'max': ...},
                'sh': {'mean': ..., 'std': ..., 'max': ...},
                'opacity': {'mean': ..., 'std': ..., 'max': ...},
                'scale': {'mean': ..., 'std': ..., 'max': ...},
                'rotation': {'mean': ..., 'std': ..., 'max': ...}
            }

    Returns:
        List of 3 tensors [mean_tensor, std_tensor, max_tensor], each of shape [14]:
            - xyz (3) + sh (3) + opacity (1) + scale (3) + rotation (4)

    Example:
        >>> mean_vals, std_vals, max_vals = dic_to_compact(stats_dic)
        >>> # Each has shape [14] representing concatenated Gaussian features
    """
    keys = ['mean', 'std', 'max']

    # Handle different naming conventions for xyz key
    xyz_key = 'xyz' if 'xyz' in stats_dic else 'offset_xyz'

    arrs = []
    for k in keys:
        components = []

        # Process each component and ensure it has proper dimensions
        for component_key, expected_size in [
            (xyz_key, 3),        # xyz coordinates
            ('sh', 3),           # spherical harmonics (first 3 components)
            ('opacity', 1),      # opacity
            ('scale', 3),        # scale
            ('rotation', 4)      # rotation quaternion
        ]:
            tensor = stats_dic[component_key][k]

            # Ensure tensor has at least 1 dimension
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)

            # Handle sh tensor specially - it might have different shapes
            if component_key == 'sh':
                if tensor.numel() >= 3:
                    if tensor.dim() >= 2 and tensor.shape[-2] > 0:
                        tensor = tensor[..., 0, :3]
                    else:
                        tensor = tensor.flatten()[:3]
                else:
                    # If sh has fewer than 3 elements, pad with zeros
                    tensor = torch.nn.functional.pad(tensor.flatten(), (0, max(0, 3 - tensor.numel())))

            # Ensure the tensor has the expected number of elements
            if tensor.numel() < expected_size:
                tensor = torch.nn.functional.pad(tensor.flatten(), (0, expected_size - tensor.numel()))
            elif tensor.numel() > expected_size:
                tensor = tensor.flatten()[:expected_size]

            components.append(tensor.flatten())

        arr = torch.cat(components, dim=0)
        arrs.append(arr)

    return arrs


def glob_images(directory: str) -> List[str]:
    """Return sorted list of .png and .jpg paths in directory."""
    return sorted(
        glob.glob(os.path.join(directory, "*.png"))
        + glob.glob(os.path.join(directory, "*.jpg"))
    )


def binary_mask_from_pil(mask_pil: Image.Image) -> np.ndarray:
    """Convert a PIL mask to a binary mask."""
    mask_np = np.array(mask_pil)
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]
    mask_np = (mask_np > 0).astype(np.uint8) * 255
    return mask_np[..., None]


def bbox_from_binary_mask(mask_np: np.ndarray) -> Tuple[int, int, int, int]:
    """Returns left, top, right, bottom coordinates."""
    mask_2d = mask_np[..., 0] if mask_np.ndim == 3 else mask_np
    h, w = mask_2d.shape

    rows = np.any(mask_2d > 0, axis=1)
    cols = np.any(mask_2d > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, w, h

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)


def apply_affine_to_intrinsics(
    K_raw: np.ndarray, affine_trans: torch.Tensor
) -> torch.Tensor:
    """Apply affine transformation in image transformations to intrinsics matrix."""
    if affine_trans.shape[-2:] == (2, 3):
        affine_3x3 = torch.eye(3, dtype=affine_trans.dtype, device=affine_trans.device)
        affine_3x3[:2, :3] = affine_trans
    elif affine_trans.shape[-2:] == (3, 3):
        affine_3x3 = affine_trans
    else:
        raise ValueError(f"Invalid affine_trans shape: {affine_trans.shape}")

    K_raw_t = torch.from_numpy(K_raw).float().to(affine_3x3)
    K_proc = affine_3x3 @ K_raw_t
    K_proc[2] = torch.tensor([0.0, 0.0, 1.0], dtype=K_proc.dtype, device=K_proc.device)
    return K_proc.float()
