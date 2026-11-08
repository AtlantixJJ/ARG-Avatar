"""Geometry operations on meshes, UV maps, point clouds, etc.
"""
from jaxtyping import Float
from torch import Tensor
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def barycentric_interpolate_vertices(
    vertex_attrs: torch.Tensor,
    pix_to_face: torch.Tensor,
    bary_coords: torch.Tensor,
    tri_faces: torch.Tensor,
) -> torch.Tensor:
    """Interpolate vertex attributes onto the UV grid using barycentric coords.

    For each UV pixel, interpolated_attr = sum(bary_coords * vertex_attrs[tri_faces[face_id]]).

    Args:
        vertex_attrs: [V, D] or [B, V, D] vertex attributes (e.g. positions, rotations).
        pix_to_face: [1, H, W, 1] face index per pixel (-1 if no face).
        bary_coords: [1, H, W, 3] barycentric weights per pixel.
        tri_faces: [F, 3] mesh vertex indices per face.

    Returns:
        [1, H, W, D] interpolated attribute per UV pixel (invalid pixels zeroed).
    """
    if vertex_attrs.dim() == 2:
        vertex_attrs = vertex_attrs.unsqueeze(0)
    batch_size = vertex_attrs.shape[0]
    attr_dim = vertex_attrs.shape[-1]
    if tri_faces.dim() == 2:
        tri_faces = tri_faces.unsqueeze(0)
    if tri_faces.shape[0] == 1 and batch_size > 1:
        tri_faces = tri_faces.expand(batch_size, -1, -1)
    if vertex_attrs.shape[0] == 1 and batch_size > 1:
        vertex_attrs = vertex_attrs.expand(batch_size, -1, -1)
    H, W = pix_to_face.shape[1], pix_to_face.shape[2]
    num_faces = tri_faces.shape[1]
    
    # Ensure all inputs are on the same device
    device = vertex_attrs.device
    pix_to_face = pix_to_face.to(device)
    bary_coords = bary_coords.to(device)
    tri_faces = tri_faces.to(device)

    pix_to_face_clamped = pix_to_face.clamp(min=0, max=max(0, num_faces - 1)).long()
    pix_face_idx = pix_to_face_clamped.squeeze(-1)
    pix_face_idx_flat = pix_face_idx.view(batch_size, -1)
    pix_vert_idx = torch.gather(
        tri_faces,
        dim=1,
        index=pix_face_idx_flat.unsqueeze(-1).expand(-1, -1, 3),
    ).view(batch_size, H, W, 3)
    pix_vert_idx_flat = pix_vert_idx.view(batch_size, -1)
    num_verts = vertex_attrs.shape[1]
    pix_vert_idx_flat = pix_vert_idx_flat.clamp(min=0, max=max(0, num_verts - 1))
    pix_attrs = torch.gather(
        vertex_attrs,
        dim=1,
        index=pix_vert_idx_flat.unsqueeze(-1).expand(-1, -1, attr_dim).long(),
    ).view(batch_size, H, W, 3, attr_dim)
    if bary_coords.shape[0] == 1 and batch_size > 1:
        bary_coords = bary_coords.expand(batch_size, -1, -1, -1)
    interpolated = (bary_coords.unsqueeze(-1) * pix_attrs).sum(dim=-2)
    return interpolated * (pix_to_face >= 0).float()


def scatter_pixel_deform_to_vertices(
    pixel_deform: torch.Tensor,   # [B, L, 3]
    face_ids: torch.Tensor,       # [L]
    bary_coords: torch.Tensor,    # [L, 3]
    faces: torch.Tensor,          # [F, 3]
    num_verts: int,
) -> torch.Tensor:
    """Scatter per-UV-pixel deformation to mesh vertices via barycentric weights.
    
    For each pixel i on face face_ids[i] with bary weights (w0, w1, w2):
      vertex_accum[v_j] += w_j * pixel_deform[i]   for j in {0,1,2}
      weight_accum[v_j] += w_j
    Then normalize: vertex_deform = vertex_accum / weight_accum.
    
    Returns: [B, V, 3] per-vertex deformation.
    """
    B, L, D = pixel_deform.shape
    device = pixel_deform.device
    bary_coords = bary_coords.to(device=device, dtype=pixel_deform.dtype)

    tri_v_idx = faces[face_ids]  # [L, 3] — 3 vertex indices per pixel

    # Accumulators
    vertex_accum = torch.zeros(B, num_verts, D, device=device, dtype=pixel_deform.dtype)
    weight_accum = torch.zeros(B, num_verts, 1, device=device, dtype=pixel_deform.dtype)

    for j in range(3):
        # Index for this barycentric corner: [L] vertex indices
        v_idx = tri_v_idx[:, j]  # [L]
        w_j = bary_coords[:, j]  # [L]

        # Weighted deformation: [B, L, D]
        weighted_deform = pixel_deform * w_j.view(1, L, 1)
        # Weight contribution: [B, L, 1]
        w_contrib = w_j.view(1, L, 1).expand(B, L, 1)

        # Expand index for scatter: [B, L, D] and [B, L, 1]
        idx_expand_d = v_idx.view(1, L, 1).expand(B, L, D)
        idx_expand_1 = v_idx.view(1, L, 1).expand(B, L, 1)

        vertex_accum.scatter_add_(1, idx_expand_d, weighted_deform)
        weight_accum.scatter_add_(1, idx_expand_1, w_contrib)

    # Normalize by total weight per vertex
    vertex_deform = vertex_accum / (weight_accum + 1e-8)
    return vertex_deform


def interpolate_vertex_deform_to_pixels(
    vertex_deform: torch.Tensor,  # [B, V, 3]
    face_ids: torch.Tensor,       # [L]
    bary_coords: torch.Tensor,    # [L, 3]
    faces: torch.Tensor,          # [F, 3]
) -> torch.Tensor:
    """Interpolate per-vertex deformation back to UV pixels via barycentric weights.
    
    For each pixel i: result[i] = sum_j(bary[i,j] * vertex_deform[v_j])
    
    Returns: [B, L, 3] per-pixel base offset.
    """
    bary_coords = bary_coords.to(device=vertex_deform.device, dtype=vertex_deform.dtype)
    tri_v_idx = faces[face_ids]  # [L, 3]
    B = vertex_deform.shape[0]
    L = face_ids.shape[0]

    # Gather the 3 vertex values for each pixel's triangle
    # tri_v_idx[:, j] is [L], need to gather from [B, V, 3]
    result = torch.zeros(B, L, vertex_deform.shape[-1], device=vertex_deform.device, dtype=vertex_deform.dtype)
    for j in range(3):
        v_idx = tri_v_idx[:, j]  # [L]
        v_vals = vertex_deform[:, v_idx, :]  # [B, L, 3]
        result = result + bary_coords[:, j].view(1, L, 1) * v_vals

    return result


def furthest_point_sampling(
        points: torch.Tensor,
        n_samples: int,
        start_set: list = []) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform probability-based furthest point sampling on a set of points.

    Instead of always selecting the furthest point deterministically, this function
    samples based on distance probabilities, providing some randomness while still
    favoring points that are far from already selected points.

    Args:
        points: Tensor of shape [N, 3] containing point coordinates
        n_samples: Number of points to sample

    Returns:
        indices: Array of shape [n_samples] containing indices of sampled points

    Algorithm:
        1. Start with a random point
        2. For each iteration:
           - Compute distance from each point to nearest selected point
           - Convert distances to probabilities (higher distance = higher probability)
           - Sample next point based on these probabilities
        3. Repeat until n_samples points are selected

    Example:
        >>> points = torch.randn(100, 3)  # 100 3D points
        >>> selected_indices = furthest_point_sampling(points, 10)  # Sample 10 points
        >>> sampled_points = points[selected_indices]
    """
    N = points.shape[0]
    if n_samples >= N:
        return np.arange(N)

    # Initialize with a random point
    sampled_indices = [int(np.random.randint(N))] if len(start_set) == 0 else start_set
    distances = torch.full((N,), float('inf'), dtype=points.dtype, device=points.device)

    # Initialize distances from start_set points
    for idx in sampled_indices[:-1]:
        new_distances = torch.norm(points - points[idx], dim=1)
        distances = torch.minimum(distances, new_distances)

    for _ in range(n_samples - len(sampled_indices)):
        # Update distances to the nearest sampled point
        last_point = points[sampled_indices[-1]]
        new_distances = torch.norm(points - last_point, dim=1)
        distances = torch.minimum(distances, new_distances)

        # Convert distances to probabilities (higher distance = higher probability)
        # Set probability of already sampled points to 0
        probs = distances.clone()
        probs[sampled_indices] = 0.0

        # Normalize to create a probability distribution
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # If all distances are 0, use uniform distribution
            probs = torch.ones_like(probs)
            probs[sampled_indices] = 0.0
            probs = probs / probs.sum()

        # Sample next point based on distance probabilities
        next_idx = torch.multinomial(probs, num_samples=1).item()
        sampled_indices.append(int(next_idx))

    sampled_indices = torch.tensor(sampled_indices, device=points.device)
    sampled_points = points[sampled_indices]
    return sampled_indices, sampled_points


def relative_disparity_to_depth(
    relative_disparity: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert relative disparity, where 0 is near and 1 is far, to depth."""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    return 1 / ((1 - relative_disparity) * (disp_near - disp_far) + disp_far + eps)


def depth_to_relative_disparity(
    depth: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert depth to relative disparity, where 0 is near and 1 is far."""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    disp = 1 / (depth + eps)
    return 1 - (disp - disp_far) / (disp_near - disp_far + eps)


def calc_face_normals(vertices: torch.Tensor, faces: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Calculate face normals from vertices and faces.

    Args:
        vertices: Vertex positions [V, 3]
        faces: Face indices [F, 3]
        normalize: Whether to normalize the normals

    Returns:
        Face normals [F, 3]
    """
    full_vertices = vertices[faces]
    v0, v1, v2 = full_vertices.unbind(dim=1)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    if normalize:
        face_normals = F.normalize(face_normals, eps=1e-6, dim=1)
    return face_normals


def calc_vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_normals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Calculate vertex normals by averaging adjacent face normals.

    Args:
        vertices: Vertex positions [V, 3]
        faces: Face indices [F, 3]
        face_normals: Pre-computed face normals [F, 3] (optional)

    Returns:
        Vertex normals [V, 3]
    """
    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)

    F_num = faces.shape[0]
    vertex_normals = torch.zeros(
        (vertices.shape[0], 3, 3), dtype=vertices.dtype, device=vertices.device
    )
    vertex_normals.scatter_add_(
        dim=0,
        index=faces[:, :, None].expand(F_num, 3, 3),
        src=face_normals[:, None, :].expand(F_num, 3, 3),
    )
    vertex_normals = vertex_normals.sum(dim=1)
    return F.normalize(vertex_normals, eps=1e-6, dim=1)
