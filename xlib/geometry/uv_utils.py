import torch
import math
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass

from xlib.geometry.geometry import calc_vertex_normals


###### I/O functions ######


def parse_uv_obj(obj_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse OBJ file and extract UV coords and UV face indices.

    Returns:
        uv_coords: [N_uv, 2] in [0, 1]
        uv_faces: [F, 3] indices into uv_coords
    """
    import numpy as np

    uv_coords_list, uv_faces_list = [], []

    with open(obj_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "vt" and len(parts) >= 3:
                uv_coords_list.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f" and len(parts) >= 4:
                face_uv = []
                for vs in parts[1:4]:
                    if "/" in vs:
                        idx = vs.split("/")
                        uv_idx = int(idx[1]) - 1 if len(idx) > 1 and idx[1] else int(idx[0]) - 1
                        face_uv.append(uv_idx)
                    else:
                        face_uv.append(int(vs) - 1)
                if len(face_uv) == 3:
                    uv_faces_list.append(face_uv)

    uv_coords = (
        np.array(uv_coords_list, dtype=np.float32)
        if uv_coords_list
        else np.zeros((0, 2), dtype=np.float32)
    )
    uv_faces = (
        np.array(uv_faces_list, dtype=np.int64)
        if uv_faces_list
        else np.zeros((0, 3), dtype=np.int64)
    )
    return (
        torch.from_numpy(uv_coords).float(),
        torch.from_numpy(uv_faces).long(),
    )


def load_obj_uv_layout(
    obj_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load OBJ mesh with UVs and return tensors for UV rasterization.

    Supports face elements in forms: `v`, `v/vt`, `v//vn`, `v/vt/vn`.
    Polygons with >3 vertices are fan-triangulated.

    Args:
        obj_path: path to OBJ.
        device: optional output device.

    Returns:
        vertices: [V, 3] float32
        tri_faces: [F, 3] int64, vertex indices into vertices
        uv_coords: [N_uv, 2] float32 in [0, 1]
        uv_faces: [F, 3] int64, UV indices into uv_coords
    """
    vertices: List[List[float]] = []
    uv_coords: List[List[float]] = []
    tri_faces: List[List[int]] = []
    uv_faces: List[List[int]] = []

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue
            if tag == "vt" and len(parts) >= 3:
                uv_coords.append([float(parts[1]), float(parts[2])])
                continue
            if tag != "f" or len(parts) < 4:
                continue

            face_v: List[int] = []
            face_vt: List[int] = []
            for tok in parts[1:]:
                comps = tok.split("/")
                v_idx = int(comps[0]) - 1
                vt_idx = -1
                if len(comps) > 1 and comps[1] != "":
                    vt_idx = int(comps[1]) - 1
                face_v.append(v_idx)
                face_vt.append(vt_idx)

            for i in range(1, len(face_v) - 1):
                tri = [face_v[0], face_v[i], face_v[i + 1]]
                tri_vt = [face_vt[0], face_vt[i], face_vt[i + 1]]
                tri_faces.append(tri)
                uv_faces.append(tri_vt)

    if len(vertices) == 0:
        raise ValueError(f"No vertices found in OBJ: {obj_path}")
    if len(uv_coords) == 0:
        raise ValueError(f"No UV coordinates (vt) found in OBJ: {obj_path}")
    if len(tri_faces) == 0:
        raise ValueError(f"No faces found in OBJ: {obj_path}")

    tri_faces_np = np.asarray(tri_faces, dtype=np.int64)
    uv_faces_np = np.asarray(uv_faces, dtype=np.int64)
    if (uv_faces_np < 0).any():
        raise ValueError(f"OBJ has faces without UV indices; cannot build UV rasterization: {obj_path}")

    out_device = device if device is not None else torch.device("cpu")
    vertices_t = torch.from_numpy(np.asarray(vertices, dtype=np.float32)).to(out_device)
    tri_faces_t = torch.from_numpy(tri_faces_np).to(out_device)
    uv_coords_t = torch.from_numpy(np.asarray(uv_coords, dtype=np.float32)).to(out_device)
    uv_faces_t = torch.from_numpy(uv_faces_np).to(out_device)
    return vertices_t, tri_faces_t, uv_coords_t, uv_faces_t


def load_fbx_uv_layout(
    fbx_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load FBX mesh with UVs via open3d and return tensors for UV rasterization.

    open3d stores per-corner UVs (triangle_uvs has shape [F*3, 2]).
    We deduplicate UVs to build indexed uv_coords / uv_faces.

    Args:
        fbx_path: path to FBX file.
        device: optional output device.

    Returns:
        vertices: [V, 3] float32
        tri_faces: [F, 3] int64, vertex indices
        uv_coords: [N_uv, 2] float32, unique UV coordinates
        uv_faces: [F, 3] int64, indices into uv_coords
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(fbx_path)
    if not mesh.has_triangle_uvs():
        raise ValueError(f"FBX has no UV coordinates: {fbx_path}")

    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh.triangles, dtype=np.int64)
    tri_uvs_np = np.asarray(mesh.triangle_uvs, dtype=np.float32)  # [F*3, 2]

    # Deduplicate UVs: round to avoid floating-point noise
    tri_uvs_flat = tri_uvs_np.reshape(-1, 2)
    uv_coords_unique, inverse_idx = np.unique(
        np.round(tri_uvs_flat, decimals=7), axis=0, return_inverse=True
    )
    uv_faces_np = inverse_idx.reshape(-1, 3).astype(np.int64)

    out_device = device if device is not None else torch.device("cpu")
    return (
        torch.from_numpy(vertices_np).to(out_device),
        torch.from_numpy(faces_np).to(out_device),
        torch.from_numpy(uv_coords_unique.astype(np.float32)).to(out_device),
        torch.from_numpy(uv_faces_np).to(out_device),
    )


##### Creating UV maps ######

def xatlas_uvmap(
    v_pos: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unwrap mesh to UV using xatlas.

    Args:
        v_pos: Vertex positions [V, 3]
        faces: Face indices [F, 3] into v_pos

    Returns:
        uv_coords: [N_uv, 2] UV coordinates in [0, 1]
        uv_faces: [F, 3] indices into uv_coords (for rasterization)
        tri_faces: [F, 3] original vertex indices (for interpolate_attributes).
                   tri_faces[i, j] = vmapping[uv_faces[i, j]] so that
                   interpolate_attributes(vertices, tri_faces) uses correct vertices.
    """
    v_pos_np = v_pos.detach().cpu().numpy().astype(np.float64)
    faces_np = faces.detach().cpu().numpy().astype(np.uint32)
    vmapping, indices, uvs = xatlas.parametrize(v_pos_np, faces_np)
    # indices [F, 3] index into uvs; vmapping maps UV vertex index -> original vertex index
    tri_faces_np = np.asarray(vmapping)[np.asarray(indices)]  # [F, 3]
    uv_coords = torch.from_numpy(uvs.astype(np.float32)).to(device=v_pos.device)
    uv_faces = torch.from_numpy(indices.astype(np.int64)).to(device=v_pos.device)
    tri_faces = torch.from_numpy(tri_faces_np.astype(np.int64)).to(device=v_pos.device)
    return uv_coords, uv_faces, tri_faces


def vertex_uv_dedup(
    uv_coords: torch.Tensor,
    uv_faces: torch.Tensor,
    tri_faces: torch.Tensor,
    num_vertices: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute one UV coordinate per mesh vertex from xatlas output.

    At chart boundaries a vertex can have multiple UVs; we take the first occurrence.

    Args:
        uv_coords: [N_uv, 2] from xatlas_uvmap
        uv_faces: [F, 3] from xatlas_uvmap
        tri_faces: [F, 3] from xatlas_uvmap (original vertex indices)
        num_vertices: V (number of mesh vertices)
        device: target device for output

    Returns:
        [V, 2] UV coordinates in [0, 1] for each mesh vertex
    """
    device = device or uv_coords.device
    uv_coords = uv_coords.to(device)
    uv_faces = uv_faces.to(device)
    tri_faces = tri_faces.to(device)
    vertex_uv = torch.zeros(num_vertices, 2, dtype=uv_coords.dtype, device=device)
    # For each vertex v, set vertex_uv[v] = uv_coords[uv_faces[j, k]] for first (j,k) with tri_faces[j,k]==v
    seen = torch.zeros(num_vertices, dtype=torch.bool, device=device)
    for j in range(tri_faces.shape[0]):
        for k in range(3):
            v = tri_faces[j, k].item()
            if not seen[v]:
                vertex_uv[v] = uv_coords[uv_faces[j, k]]
                seen[v] = True
    # Any unset (e.g. disconnected verts) get (0, 0)
    return vertex_uv


##### UV Coordinates ######


def prepare_uv_coords(uv_coords: torch.Tensor) -> torch.Tensor:
    uv_ndc = uv_coords.clone() * 2.0 - 1.0
    uv_ndc[..., 1] = -uv_ndc[..., 1]
    uv_hom = torch.cat(
        [
            uv_ndc,
            torch.zeros_like(uv_ndc[..., :1]),
            torch.ones_like(uv_ndc[..., :1]),
        ],
        dim=-1,
    )
    return uv_hom.unsqueeze(0) if uv_hom.dim() == 2 else uv_hom


def scatter_feature_uv(features: torch.Tensor, uv_mask: torch.Tensor, uv_size: Optional[int] = None) -> torch.Tensor:
    B, L, C = features.shape
    H = W = uv_size if uv_size is not None else int(math.sqrt(uv_mask.shape[0]))
    if uv_mask is not None and uv_mask.dtype == torch.bool and L != H * W:
        dense = torch.zeros(B, H * W, C, device=features.device, dtype=features.dtype)
        dense[:, uv_mask, :] = features
    else:
        dense = features.reshape(B, H * W, C)
    return dense.permute(0, 2, 1).reshape(B, C, H, W)


def gather_feature_uv(uv_map: torch.Tensor, uv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    features = uv_map.flatten(2).permute(0, 2, 1)
    if uv_mask is not None and uv_mask.dtype == torch.bool:
        return features[:, uv_mask, :]
    return features



##### UV Rasterization ######

class UVRasterizer(nn.Module):
    """
    UV Rasterizer using nvdiffrast for generating position and normal maps.

    Rasterizes the UV layout to get per-pixel barycentric coordinates, then
    interpolates vertex attributes (positions, normals) onto the UV image.
    """

    def __init__(self, uv_size: int = 1024, device: str = "cuda"):
        super().__init__()
        self.uv_size = uv_size
        self.device = device
        self._glctx = None
        self._cached_pix_to_face = None
        self._cached_bary_coords = None
        self._cached_uv_mask = None
        self._cached_uv_mask_idx = None

    @property
    def glctx(self):
        if self._glctx is None:
            self._glctx = dr.RasterizeCudaContext()
            pos = torch.tensor(
                [[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]],
                dtype=torch.float32,
                device=self.device,
            )
            dr.rasterize(
                self._glctx,
                pos,
                torch.tensor([[0, 1, 2]], dtype=torch.int32, device=self.device),
                resolution=[8, 8],
            )
        return self._glctx

    def rasterize_uv(
        self,
        uv_coords: torch.Tensor,
        uv_faces: torch.Tensor,
        cache_results: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uv_hom = prepare_uv_coords(uv_coords)
        rast_out, _ = dr.rasterize(
            self.glctx,
            uv_hom.contiguous(),
            uv_faces.int().contiguous(),
            resolution=[self.uv_size, self.uv_size],
        )

        pix_to_face = rast_out[..., 3:4].int() - 1
        bary_uv = rast_out[..., :2]
        bary_coords = torch.cat(
            [
                1.0 - bary_uv[..., 0:1] - bary_uv[..., 1:2],
                bary_uv[..., 0:1],
                bary_uv[..., 1:2],
            ],
            dim=-1,
        )
        uv_mask = (pix_to_face >= 0).float()

        if cache_results:
            self._cached_pix_to_face = pix_to_face
            self._cached_bary_coords = bary_coords
            self._cached_uv_mask = uv_mask
            self._cached_uv_mask_idx = uv_mask.squeeze(-1).view(-1) > 0

        return rast_out, pix_to_face, bary_coords

    def interpolate_attributes(
        self,
        attributes: torch.Tensor,
        tri_faces: torch.Tensor,
        pix_to_face: Optional[torch.Tensor] = None,
        bary_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Interpolate vertex attributes onto UV raster grid using barycentric coords.

        Args:
            attributes: [B, V, C] or [V, C] vertex attributes (e.g. positions, normals)
            tri_faces: [F, 3] or [B, F, 3] mesh face indices (must match uv_faces indexing)
            pix_to_face, bary_coords: cached from rasterize_uv
        """
        pix_to_face = pix_to_face if pix_to_face is not None else self._cached_pix_to_face
        bary_coords = bary_coords if bary_coords is not None else self._cached_bary_coords

        if attributes.dim() == 2:
            attributes = attributes.unsqueeze(0)
        if tri_faces.dim() == 2:
            tri_faces = tri_faces.unsqueeze(0)

        batch_size, _, attr_dim = attributes.shape
        H, W = pix_to_face.shape[1:3]

        pix_to_face_clamped = pix_to_face.clamp(min=0).long()
        pix_face_idx = pix_to_face_clamped.squeeze(-1)

        if tri_faces.shape[0] == 1 and batch_size > 1:
            tri_faces = tri_faces.expand(batch_size, -1, -1)
        if attributes.shape[0] == 1 and batch_size > 1:
            attributes = attributes.expand(batch_size, -1, -1)

        pix_face_idx_flat = pix_face_idx.view(batch_size, -1)
        pix_vert_idx = torch.gather(
            tri_faces,
            dim=1,
            index=pix_face_idx_flat.unsqueeze(-1).expand(-1, -1, 3),
        ).view(batch_size, H, W, 3)

        pix_vert_idx_flat = pix_vert_idx.view(batch_size, -1)
        pix_attrs = torch.gather(
            attributes,
            dim=1,
            index=pix_vert_idx_flat.unsqueeze(-1).expand(-1, -1, attr_dim),
        ).view(batch_size, H, W, 3, attr_dim)

        interpolated = (bary_coords.unsqueeze(-1) * pix_attrs).sum(dim=-2)
        return interpolated * (pix_to_face >= 0).float()


def compute_position_normal_maps(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uv_rasterizer: UVRasterizer,
    uv_coords: torch.Tensor,
    uv_faces: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute position and normal maps from mesh vertices.

    Based on new_dir Data/neuman.py _compute_uv_maps.

    Args:
        vertices: [V, 3] or [B, V, 3] vertex positions
        faces: [F, 3] face indices
        uv_rasterizer: Initialized UVRasterizer
        uv_coords: [N_uv, 2] UV coordinates in [0, 1]
        uv_faces: [F, 3] UV face indices (must align with faces)

    Returns:
        position_map: [1, H, W, 3] or [B, H, W, 3]
        normal_map: [1, H, W, 3] or [B, H, W, 3]
    """
    with torch.no_grad():
        vertices = vertices.to(device)
        faces = faces.to(device)
        uv_coords = uv_coords.to(device)
        uv_faces = uv_faces.to(device)

        normals = calc_vertex_normals(vertices, faces)

        uv_rasterizer.rasterize_uv(uv_coords, uv_faces)

        position_map = uv_rasterizer.interpolate_attributes(vertices, faces)
        normal_map = uv_rasterizer.interpolate_attributes(normals, faces)
        normal_map = F.normalize(normal_map, dim=-1, eps=1e-6)
        normal_map = normal_map * uv_rasterizer._cached_uv_mask

    return position_map, normal_map


@torch.no_grad()
def rasterize_uv_map(mhr_animatable_gs, uv_size: int, patch_size: int, lod: int, device: Any) -> Optional[Dict[str, Any]]:
    neutral_vertices = mhr_animatable_gs.query_points.to(device)
    mhr_faces = mhr_animatable_gs.faces.to(device)
    uv_fbx_path = f'mesh_exports/mhr_mesh_lod{lod}.fbx'
    fbx_vertices, fbx_tri_faces, uv_coords, uv_faces = load_fbx_uv_layout(uv_fbx_path, device=device)
    
    uv_rasterizer = UVRasterizer(uv_size=uv_size, device=device)
    pos_map, norm_map = compute_position_normal_maps(
        neutral_vertices, mhr_faces, uv_rasterizer, uv_coords, uv_faces, device=device
    )
    pix_to_face = uv_rasterizer._cached_pix_to_face
    bary_coords = uv_rasterizer._cached_bary_coords
    uv_mask_2d = uv_rasterizer._cached_uv_mask
    uv_mask_flat = uv_mask_2d.view(-1).bool().to(device)

    pos_map = pos_map.permute(0, 3, 1, 2)
    norm_map = norm_map.permute(0, 3, 1, 2)
    anchors = pos_map.flatten(2).permute(0, 2, 1)
    anchors = anchors[:, uv_mask_flat, :].unsqueeze(1)

    # Compute patch-center positions for RoPE in UV-pixel units.
    grid_h = grid_w = uv_size // patch_size
    y, x = torch.meshgrid(torch.arange(grid_h, device=device), torch.arange(grid_w, device=device), indexing="ij")
    patch_center = patch_size * 0.5
    positions = torch.stack([x * patch_size + patch_center, y * patch_size + patch_center], dim=-1).view(-1, 2)

    return {
        "position_map": pos_map,
        "normal_map": norm_map,
        "uv_mask": uv_mask_flat,
        "pix_to_face": pix_to_face,
        "bary_coords": bary_coords,
        "tri_faces": mhr_faces,
        "patch_positions": positions,
        "anchor_points": anchors,
    }


##### UV Information Classes #####


@dataclass
class UVComponentInfo:
    component_id: int
    bbox: Tuple[int, int, int, int]     # (r_min, c_min, r_max, c_max)
    scale_factor: float
    scale_anchor: str
    anchor_uv: Tuple[float, float]      # current-scale UV-space anchor in (y, x)
    scaled_size: Tuple[int, int]        # padded to patch_size multiple (H_s, W_s)
    n_patches_h: int
    n_patches_w: int
    pixel_mask: torch.Tensor            # [H_s, W_s] bool
    patch_mask: torch.Tensor            # [n_patches] bool, false for empty local patches
    orig_pixel_mask: torch.Tensor       # [H, W] bool on the original UV raster
    n_valid_pixels: int                 # Number of valid pixels in THIS resampled component
    patch_global_offset: int            # start in token sequence
    pixel_global_offset: int            # start in output [B, L, C]
    patch_positions: torch.Tensor       # [n_patches, 2] global UV-pixel coords for RoPE

    @property
    def n_patches(self) -> int:
        return self.n_patches_h * self.n_patches_w


@dataclass
class UVSampleTopology:
    face_ids: torch.Tensor      # [L]
    bary_coords: torch.Tensor   # [L, 3]
    anchor_points: torch.Tensor # [1, 1, L, 3]
    component_ids: torch.Tensor # [L]
    sample_normals: Optional[torch.Tensor] = None  # [1, 1, L, 3]
