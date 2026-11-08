from __future__ import annotations

"""UV encoding utilities.

Naming convention in this file:
- `_s` denotes tensors or coordinates in a full scale-level UV map.
- `_crop` denotes tensors cropped from a full map and padded so the local canvas
  stays patch-aligned.
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

from typing import Optional, Dict, Tuple, Any, List, Union
import cv2

from xlib.geometry.uv_utils import (
    rasterize_uv_map,
    UVComponentInfo,
    UVSampleTopology,
    scatter_feature_uv,
)
from xlib.geometry.geometry import barycentric_interpolate_vertices, calc_vertex_normals
from xlib.basetype import TensorDict
from xlib.render.gaussian_splatting import gs_to_feature



def export_uv_tokenizer_state_dict(
    model_state: Dict[str, Any],
    *,
    patch_embed_prefix: str,
    uv_decoder_prefix: str,
    uv_token_cond_prefix: Optional[str] = None,
    flattened: bool = False,
) -> Dict[str, Any]:
    """Export nested ``uv_tokenizer`` modules under legacy top-level prefixes."""
    exported: Dict[str, Any] = {}
    prefix = "" if flattened else "uv_tokenizer."
    nested_prefixes = (
        (f"{prefix}patch_embed.", patch_embed_prefix),
        (f"{prefix}uv_decoder.", uv_decoder_prefix),
    )
    if uv_token_cond_prefix is not None:
        nested_prefixes = nested_prefixes + ((f"{prefix}uv_token_cond.", uv_token_cond_prefix),)

    for key, value in model_state.items():
        for old_prefix, new_prefix in nested_prefixes:
            if key.startswith(old_prefix):
                exported[new_prefix + key[len(old_prefix):]] = value
                break
    return exported


def gs_to_avatar_tokens(
    gs_dic: dict,
    uv_encoding: UVEncoding,
    token_cond: UVConditionCollector,
    patch_embed: nn.Module,
    *,
    lbs_feat: Optional[torch.Tensor] = None,
    visibility_ratio: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encode GS features into avatar tokens."""
    feats = token_cond(
        gs_dic,
        lbs_feat=lbs_feat,
        visibility_ratio=visibility_ratio,
    )
    return uv_encoding.encode(feats, patch_embed)


def avatar_tokens_to_gs(
    avatar_tokens: torch.Tensor,
    uv_encoding: UVEncoding,
    decoder: nn.Module,
    gshead: Any,
    gs_tpose_dic_anchor: Optional[dict] = None,
) -> Tuple[dict, Any]:
    """Decode avatar tokens into activated GS parameters."""
    gs_feats, gs_feats_uv = uv_encoding.decode(avatar_tokens, decoder=decoder)
    gs_feats = gs_feats.unsqueeze(1)
    raw_gs_delta = gshead.forward_raw(gs_feats)
    delta_gs = gshead.presum(raw_gs_delta)
    seqs = [delta_gs]
    if gs_tpose_dic_anchor is not None:
        seqs.append(gs_tpose_dic_anchor)
    new_raw_gs = gshead.sum_residuals(seqs)
    return gshead.activate(new_raw_gs), gs_feats_uv


# Order obtained from export_uv_component_masks.py
PHYSICAL_CC_COMPONENT_NAMES = {
    0: 'face_body',
    1: 'right_hand', # left and right are not determined yet
    2: 'left_hand',
    3: 'right_leg',
    4: 'left_leg',
    5: 'right_arm',
    6: 'left_arm',
    7: 'right_foot_upper',
    8: 'left_foot_upper',
    9: 'right_foot_lower',
    10: 'left_foot_lower',
}

CC_COMPONENT_NAMES = {
    0: 'face',
    1: 'body',
    2: 'right_hand',
    3: 'left_hand',
    4: 'right_leg',
    5: 'left_leg',
    6: 'right_arm',
    7: 'left_arm',
    8: 'right_foot_upper',
    9: 'left_foot_upper',
    10: 'right_foot_lower',
    11: 'left_foot_lower',
}


def get_component_mask(level_info: dict, component_idx: int) -> np.ndarray:
    return level_info["mask_labels_int32"] == level_info["sorted_labels"][component_idx]


def split_face_body_mask(comp_mask: np.ndarray, split_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    rows, _ = np.where(comp_mask)
    if rows.size == 0:
        empty_mask = np.zeros_like(comp_mask, dtype=bool)
        return empty_mask, empty_mask

    r_min = int(rows.min())
    r_max = int(rows.max()) + 1
    split_row = r_min + max(1, int((r_max - r_min) * split_ratio))

    face_mask = comp_mask.copy()
    face_mask[split_row:, :] = False

    body_mask = comp_mask.copy()
    body_mask[:split_row, :] = False

    return face_mask, body_mask


def split_logical_component_labels(sorted_labels: np.ndarray, mask_labels_int32: np.ndarray) -> dict:
    if len(sorted_labels) == 0:
        return {
            "sorted_labels": sorted_labels,
            "mask_labels_int32": mask_labels_int32,
        }

    face_body_mask = mask_labels_int32 == 0
    face_mask, body_mask = split_face_body_mask(face_body_mask)

    logical_mask_labels = mask_labels_int32.copy()
    later_component_mask = logical_mask_labels > 0
    logical_mask_labels[later_component_mask] = logical_mask_labels[later_component_mask] + 1
    logical_mask_labels[face_mask] = 0
    logical_mask_labels[body_mask] = 1

    logical_sorted_labels = np.arange(logical_mask_labels.max() + 1)
    return {
        "sorted_labels": logical_sorted_labels,
        "mask_labels_int32": logical_mask_labels,
    }


class UVConditionCollector(nn.Module):
    """Builds the full per-point feature vector for UV token encoding.

    Reads the ``UV_token_cond`` config section and provides:
    * ``total_dim`` — ``gs_base_dim`` plus optional conditioning channels (matches default ``patch_embed`` in channels)
    * ``forward()`` — concatenates configured GS features, optional visibility, and optional conditioning

    Conditioning flags:
        normal           +3   neutral surface normals per UV sample
        lbs              +12  LBS features (requires ``lbs_feat``)
        neutral_anchor   +3   neutral mesh anchor positions per UV sample
        neutral_avatar   +V  neutral T-pose GS features (``gs_to_feature`` of init dict)
                           V = 11 + (sh_degree + 1)^2 * 3

    ``UVEncoding`` is attached after construction via :meth:`set_uv_encoding`.
    """

    def __init__(
        self,
        cfg: dict,
        embed_dim: int,
        patch_size: int,
        sh_degree: int = 0,
    ):
        super().__init__()
        object.__setattr__(self, "uv_encoding", None)
        self.use_normal = bool(cfg.get("normal", False))
        self.use_lbs = bool(cfg.get("lbs", False))
        self.use_neutral_anchor = bool(cfg.get("neutral_anchor", False))
        self.use_neutral_avatar = bool(cfg.get("neutral_avatar", True))

        self.neutral_avatar_dim = 11 + (sh_degree + 1) ** 2 * 3

        self._cond_dim = (
            (3 if self.use_normal else 0)
            + (12 if self.use_lbs else 0)
            + (3 if self.use_neutral_anchor else 0)
            + (self.neutral_avatar_dim if self.use_neutral_avatar else 0)
        )

    def set_uv_encoding(self, uv_encoding: UVEncoding) -> None:
        """Attach the shared ``UVEncoding`` instance (not registered as a child module)."""
        object.__setattr__(self, "uv_encoding", uv_encoding)

    @property
    def total_dim(self) -> int:
        return self._cond_dim

    @property
    def needs_lbs_feat(self) -> bool:
        """True when caller must provide pose-dependent LBS features."""
        return self.use_lbs

    def forward(
        self,
        gs_dic: dict,
        lbs_feat: Optional[torch.Tensor] = None,
        visibility_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``[B, L, C]`` features ready for :meth:`UVEncoding.encode`."""
        feats: List[torch.Tensor] = []

        uv_enc = self.uv_encoding
        if uv_enc is None:
            raise RuntimeError("UVConditionCollector.set_uv_encoding() must be called before forward()")
        B = gs_dic['sh'].shape[0]
        device, dtype = gs_dic['sh'].device, gs_dic['sh'].dtype

        if self.use_normal:
            n = uv_enc._uv_sample_normals.to(device=device, dtype=dtype)
            feats.append(n.expand(B, -1, -1, -1).squeeze(1))
        if self.use_lbs:
            assert lbs_feat is not None, "lbs_feat required when UV_token_cond.lbs=true"
            f = lbs_feat.squeeze(1) if lbs_feat.ndim == 4 else lbs_feat
            feats.append(f.to(device=device, dtype=dtype))
        if self.use_neutral_anchor:
            a = uv_enc._uv_anchor_points.to(device=device, dtype=dtype)
            feats.append(a.expand(B, -1, -1, -1).squeeze(1))
        if self.use_neutral_avatar:
            neutral = gs_to_feature(gs_dic)
            feats.append(neutral)

        if visibility_ratio is not None:
            vis = visibility_ratio.to(device=device, dtype=dtype)
            if vis.ndim == 4 and vis.shape[1] == 1:
                vis = vis.squeeze(1)
            feats.append(vis)

        return torch.cat(feats, dim=-1)


def find_and_sort_components(pix_to_face_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run connectedComponents on the UV mask, sort by descending pixel count.

    Args:
        pix_to_face_np: [H, W] int array, face index per pixel (-1 = background).

    Returns:
        sorted_labels: 1-D array of original component labels sorted by descending count.
        mask_labels_int32: [H, W] int32 label array (background == -1, components 0..N-1).
    """
    binary_mask = (pix_to_face_np >= 0).astype(np.uint8)
    _n_labels, mask_labels_cv = cv2.connectedComponents(binary_mask, connectivity=4)
    mask_labels_int32 = mask_labels_cv.astype(np.int32) - 1  # background → -1
    unique_labels, counts = np.unique(
        mask_labels_int32[mask_labels_int32 >= 0], return_counts=True
    )
    sorted_idx = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_idx]
    return sorted_labels, mask_labels_int32


def sample_component_pixels(
    r_min_s: int,
    c_min_s: int,
    h_crop: int,
    w_crop: int,
    patch_size: int,
    res_s: dict,
    comp_mask_s: np.ndarray,
    device: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop/pad component mask and rasterized topology, then gather valid pixels.
    Args:
        r_min_s: int, top-left row of the current-scale component
        c_min_s: int, top-left column of the current-scale component
        h_crop: int, height of the cropped component
        w_crop: int, width of the cropped component
        patch_size: int, UV patch size
        res_s: dict, rasterized topology at the current scale
        comp_mask_s: np.ndarray, component mask at the current scale
        device: Any, device to use

    Returns:
        pixel_mask: [h_crop, w_crop] bool
        patch_mask: [n_patches] bool
        face_ids:   [K] int32, valid face ids
        bary:       [K, 3] float, valid barycentric coords
    """
    uv_h, uv_w = res_s["pix_to_face"].shape[1:3]

    assert r_min_s >= 0 and c_min_s >= 0, "r_min and c_min must be greater than or equal to 0"
    assert r_min_s < uv_h and c_min_s < uv_w, "r_min and c_min must be within the UV raster bounds"

    r_max = min(r_min_s + h_crop, uv_h)
    c_max = min(c_min_s + w_crop, uv_w)

    pix_to_face_crop = res_s["pix_to_face"][:, r_min_s:r_max, c_min_s:c_max].clone().detach()
    bary_coords_crop = res_s["bary_coords"][:, r_min_s:r_max, c_min_s:c_max].clone().detach()
    comp_mask_crop = torch.from_numpy(comp_mask_s[r_min_s:r_max, c_min_s:c_max]).to(
        device=device, dtype=torch.bool
    )

    pad_h = h_crop - pix_to_face_crop.shape[1]
    pad_w = w_crop - pix_to_face_crop.shape[2]
    if pad_h > 0 or pad_w > 0:
        pix_to_face_crop = F.pad(pix_to_face_crop, (0, 0, 0, pad_w, 0, pad_h), value=-1)
        bary_coords_crop = F.pad(bary_coords_crop, (0, 0, 0, pad_w, 0, pad_h), value=0.0)
        comp_mask_crop = F.pad(comp_mask_crop, (0, pad_w, 0, pad_h), value=False)

    pix_to_face_crop = pix_to_face_crop.to(device)
    bary_coords_crop = bary_coords_crop.to(device)

    pixel_mask = comp_mask_crop.to(device=device) & (pix_to_face_crop.squeeze(0).squeeze(-1) >= 0)
    patch_mask = F.avg_pool2d(
        pixel_mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        kernel_size=patch_size,
        stride=patch_size,
    ).view(-1) > 0
    bary_coords_crop = bary_coords_crop / bary_coords_crop.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    face_ids = pix_to_face_crop.squeeze(0).squeeze(-1)[pixel_mask]
    bary = bary_coords_crop.squeeze(0)[pixel_mask]
    return pixel_mask, patch_mask, face_ids, bary


def build_patch_positions(
    bbox: Tuple[int, int, int, int],
    n_patches_h: int,
    n_patches_w: int,
    patch_size: int,
    device: Any,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Compute [n_patches, 2] global UV patch positions (x, y order) for RoPE.

    Positions are expressed as patch centers in base UV-pixel units so both
    naive and CC mode share the same global UV coordinate convention.
    """
    y_p, x_p = torch.meshgrid(
        torch.arange(n_patches_h, device=device), torch.arange(n_patches_w, device=device), indexing="ij"
    )
    r_min, c_min, _, _ = bbox
    patch_center = patch_size * 0.5 / scale_factor
    global_y = r_min + y_p * patch_size / scale_factor + patch_center
    global_x = c_min + x_p * patch_size / scale_factor + patch_center
    return torch.stack([global_x, global_y], dim=-1).view(-1, 2)


def aggregate_patch_topology(
    face_ids_px: torch.Tensor,   # [n_px], all >= 0
    bary_px: torch.Tensor,        # [n_px, 3]
    patch_ids_px: torch.Tensor,   # [n_px], 0-based, no -1
    n_patches: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average bary coords per patch; scatter-pick one representative face ID.

    Returns (patch_face_ids [n_patches], patch_bary [n_patches, 3]).
    Patches with no pixels get face_id=-1 and bary=0.
    """
    patch_ids_px = patch_ids_px.long()
    ones = torch.ones(len(patch_ids_px), device=device, dtype=torch.float32)
    patch_count = torch.zeros(n_patches, dtype=torch.float32, device=device)
    patch_count.scatter_add_(0, patch_ids_px, ones)

    bary_sum = torch.zeros(n_patches, 3, device=device, dtype=torch.float32)
    bary_sum.scatter_add_(0, patch_ids_px.unsqueeze(-1).expand(-1, 3), bary_px.float())

    valid = patch_count > 0
    patch_bary = torch.zeros(n_patches, 3, device=device, dtype=torch.float32)
    patch_bary[valid] = (bary_sum / patch_count.unsqueeze(-1).clamp(min=1))[valid]

    pixel_idx = torch.arange(len(patch_ids_px), device=device)
    rep_pixel = torch.full((n_patches,), -1, dtype=torch.long, device=device)
    rep_pixel.scatter_(0, patch_ids_px.flip(0), pixel_idx.flip(0))

    patch_face_ids = torch.full((n_patches,), -1, device=device, dtype=torch.long)
    patch_face_ids[valid] = face_ids_px[rep_pixel[valid]].long()

    return patch_face_ids, patch_bary


def decode_component_tokens(
    decoder: nn.Module,
    comp_tokens: torch.Tensor,
    comp: UVComponentInfo,
    use_patch_mask: bool = True,
) -> torch.Tensor:
    """Decode a single component token block using the shared decoder weights.

    Connected-components mode uses per-component token grids, so the local token count
    does not match the full-image decoder's configured `num_patches`.
    """
    return decode_component_tokens_batched(decoder, [comp_tokens], [comp], use_patch_mask=use_patch_mask)[0]


def _densify_component_tokens(
    comp_tokens: torch.Tensor,
    comp: UVComponentInfo,
    use_patch_mask: bool,
) -> torch.Tensor:
    """Scatter a component's (possibly masked) tokens into its dense [B, n_patches, D] grid."""
    B, n_tokens, token_dim = comp_tokens.shape
    patch_mask = comp.patch_mask.to(device=comp_tokens.device)
    n_valid_patches = int(patch_mask.sum().item())
    expected_tokens = n_valid_patches if use_patch_mask else comp.n_patches

    if n_tokens != expected_tokens:
        raise ValueError(
            f"Component token count mismatch: got {n_tokens}, expected {expected_tokens} "
            f"({'masked' if use_patch_mask else 'dense'}) patches."
        )
    if not use_patch_mask:
        return comp_tokens

    full_tokens = torch.zeros(
        B,
        comp.n_patches,
        token_dim,
        device=comp_tokens.device,
        dtype=comp_tokens.dtype,
    )
    full_tokens[:, patch_mask, :] = comp_tokens
    return full_tokens


def decode_component_tokens_batched(
    decoder: nn.Module,
    comp_tokens_list: List[torch.Tensor],
    comps: List[UVComponentInfo],
    use_patch_mask: bool = True,
) -> List[torch.Tensor]:
    """Decode a group of components sharing the same (n_patches_h, n_patches_w) token
    grid in a single batched decoder forward call, instead of one call per component.

    All entries in `comps` must have identical `n_patches_h`/`n_patches_w` (this also
    implies identical `scaled_size`, since that is derived from the patch grid), which
    holds for e.g. the mirrored left/right arm, hand, leg, and the four foot components.
    """
    ref = comps[0]
    for comp in comps[1:]:
        if (comp.n_patches_h, comp.n_patches_w) != (ref.n_patches_h, ref.n_patches_w):
            raise ValueError(
                "decode_component_tokens_batched requires matching token-grid shapes; "
                f"got {(ref.n_patches_h, ref.n_patches_w)} and {(comp.n_patches_h, comp.n_patches_w)}."
            )

    K = len(comps)
    dense_list = [
        _densify_component_tokens(comp_tokens, comp, use_patch_mask)
        for comp_tokens, comp in zip(comp_tokens_list, comps)
    ]
    B, _, token_dim = dense_list[0].shape

    # [K, B, n_patches, D] -> [K*B, D, H_p, W_p], one decoder call for the whole group.
    stacked = torch.stack(dense_list, dim=0)
    x = stacked.permute(0, 1, 3, 2).reshape(K * B, token_dim, ref.n_patches_h, ref.n_patches_w)
    decoded = decoder(x)
    decoded = decoded.view(K, B, *decoded.shape[1:])

    outputs = []
    for i, comp in enumerate(comps):
        d = decoded[i]
        if d.shape[-2:] != comp.scaled_size:
            d = F.interpolate(d, size=comp.scaled_size, mode="bilinear", align_corners=False)
        outputs.append(d)
    return outputs


class UVEncoding(nn.Module):
    def __init__(
        self,
        mode: str = "naive",
        uv_size: int = 512,
        patch_size: int = 8,
        embed_dim: int = 1024,
        out_channels: int = 32,
        in_channels: int = 14,
        component_scale_factors: Optional[Dict[Any, float]] = None,
        use_patch_mask: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.uv_size = uv_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.component_scale_factors = component_scale_factors or {}
        self.use_patch_mask = use_patch_mask

        self.components: List[UVComponentInfo] = []
        self.sample_topology: Optional[UVSampleTopology] = None
        self._setup_complete = False
        self.register_buffer("_neutral_position_map", None)
        self.register_buffer("_neutral_normal_map", None)
        self.register_buffer("_uv_mask", None)
        self.register_buffer("_uv_pix_to_face", None)
        self.register_buffer("_uv_bary_coords", None)
        self.register_buffer("_tri_faces", None)
        self.register_buffer("_mesh_edges", None)
        self.register_buffer("_uv_patch_positions", None)
        self.register_buffer("_uv_token_grid_indices", None)
        self.register_buffer("_uv_anchor_points", None)
        self.register_buffer("_uv_sample_normals", None)
        self.register_buffer("_uv_tri_v_idx", None)
        self._num_verts = 0

    @property
    def is_setup(self): return self._setup_complete
    @property
    def uv_mask(self): return self._uv_mask
    @property
    def anchor_points(self): return self._uv_anchor_points
    @property
    def patch_positions(self): return self._uv_patch_positions
    @property
    def token_grid_indices(self):
        """Map each token to its position in the full T x T UV token grid."""
        if self._uv_token_grid_indices is None:
            T = self.uv_size // self.patch_size
            pos = self.patch_positions
            col = (pos[:, 0] / self.patch_size).long().clamp(0, T - 1)
            row = (pos[:, 1] / self.patch_size).long().clamp(0, T - 1)
            self._uv_token_grid_indices = row * T + col
        return self._uv_token_grid_indices
    @property
    def total_patches(self):
        if self.mode == "naive":
            return (self.uv_size // self.patch_size) ** 2
        if self.use_patch_mask:
            return sum(int(c.patch_mask.sum().item()) for c in self.components)
        return sum(c.n_patches for c in self.components)
    @property
    def total_valid_pixels(self):
        if self.mode == "naive":
            if self._uv_mask is None:
                return self.uv_size * self.uv_size
            return int(self._uv_mask.sum().item())
        else:
            if self.sample_topology is None:
                return 0
            return self.sample_topology.face_ids.shape[0]

    def init_gs_tpose_params(self, scale_init_val: float = -0.5, sh_degree: int = 0):
        """Create canonical t-pose GS ParameterDict and buffers. Call after setup()."""
        L = self.total_valid_pixels
        device = self._uv_anchor_points.device
        dtype = torch.float32
        sh_channels = (sh_degree + 1) ** 2 * 3   # flat raw width

        self.gs_tpose_init_dic = TensorDict(
            offset_xyz=torch.zeros(1, 1, L, 3, device=device, dtype=dtype),
            scale=torch.full((1, 1, L, 1), scale_init_val, device=device, dtype=dtype),
            sh=torch.zeros(1, 1, L, 1, sh_channels, device=device, dtype=dtype),
            rotation=torch.tensor([1, 0, 0, 0], device=device, dtype=dtype).view(1,1,1,4).expand(1,1,L,4).clone(),
            opacity=torch.zeros(1, 1, L, 1, device=device, dtype=dtype),
        )

    def get_gs_tpose_init_dic(self, batch_size: int, device, dtype=None):
        """Materialize batch-expanded GS dict from params."""
        if dtype is None: dtype = torch.float32
        anchor_points = self._uv_anchor_points.to(device=device, dtype=dtype)
        return TensorDict({
            "xyz": anchor_points.expand(batch_size, -1, -1, -1),
            "offset_xyz": self.gs_tpose_init_dic["offset_xyz"].expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype),
            "scale_": self.gs_tpose_init_dic["scale"].expand(batch_size, -1, -1, 3).to(device=device, dtype=dtype),
            "rotation": self.gs_tpose_init_dic["rotation"].expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype),
            "opacity_": self.gs_tpose_init_dic["opacity"].expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype),
            "sh_": self.gs_tpose_init_dic["sh"].expand(batch_size, -1, -1, -1, -1).to(device=device, dtype=dtype),
        })

    def _rasterize_at_scales(self, mhr_animatable_gs, lod: int, device) -> Optional[Dict[float, Dict]]:
        """Rasterize at each unique scale (including 1.0). Returns {scale: result_dict} or None on failure."""
        unique_scales = set(self.component_scale_factors.values()) | {1.0}
        rasterized_levels: Dict[float, Dict] = {}
        for s in unique_scales:
            sz = int(self.uv_size * s)
            res_s = rasterize_uv_map(mhr_animatable_gs, sz, self.patch_size, lod, device)
            if res_s is None:
                return None
            rasterized_levels[s] = res_s
        return rasterized_levels

    def _register_rasterize_buffers(self, res: dict):
        self.register_buffer("_neutral_position_map", res["position_map"])
        self.register_buffer("_neutral_normal_map", res["normal_map"])
        self.register_buffer("_uv_mask", res["uv_mask"])
        self.register_buffer("_uv_pix_to_face", res["pix_to_face"])
        self.register_buffer("_uv_bary_coords", res["bary_coords"])
        self.register_buffer("_tri_faces", res["tri_faces"])

    def _setup_naive_mode(self, mhr_animatable_gs, lod: int, device) -> Optional[dict]:
        """Setup for naive mode: rasterize, register buffers, register patch positions."""
        res = rasterize_uv_map(mhr_animatable_gs, self.uv_size, self.patch_size, lod, device)
        if res is None:
            return None
        self._register_rasterize_buffers(res)
        normal_map = res["normal_map"].flatten(2).permute(0, 2, 1)
        self.register_buffer("_uv_sample_normals", normal_map[:, res["uv_mask"], :].unsqueeze(1))
        self.register_buffer("_uv_anchor_points", res["anchor_points"])
        self.register_buffer("_uv_patch_positions", res["patch_positions"])
        self.register_buffer("_uv_tri_v_idx", res["tri_faces"][res["pix_to_face"].squeeze(0).squeeze(-1)[res["uv_mask"]]])
        self._num_verts = mhr_animatable_gs.query_points.shape[0]

        return res

    def _setup_cc_mode(self, mhr_animatable_gs, lod: int, device) -> Optional[dict]:
        """Setup for connected_components mode: rasterize at all scales, find/sort components,
        build per-component info, and assemble the global sample topology."""
        rasterized_levels = self._rasterize_at_scales(mhr_animatable_gs, lod, device)
        if rasterized_levels is None:
            return None
        res = rasterized_levels[1.0]
        cc_levels = {}
        for scale in rasterized_levels:
            pix_to_face_np = rasterized_levels[scale]["pix_to_face"].cpu().numpy()[0, ..., 0]
            sorted_labels, mask_labels_int32 = find_and_sort_components(pix_to_face_np)
            cc_levels[scale] = split_logical_component_labels(sorted_labels, mask_labels_int32)

        self._register_rasterize_buffers(res)

        neutral_vertices = mhr_animatable_gs.query_points.to(device)
        mhr_faces = mhr_animatable_gs.faces.to(device)

        component_id = 0
        self.components = []
        patch_offset = 0
        pixel_offset = 0
        all_patch_positions: List[torch.Tensor] = []
        all_face_ids: List[torch.Tensor] = []
        all_bary_coords: List[torch.Tensor] = []
        all_comp_ids: List[torch.Tensor] = []
        for component_idx in range(len(cc_levels[1.0]["sorted_labels"])):
            comp_name = CC_COMPONENT_NAMES[component_idx]
            comp_mask_1 = get_component_mask(cc_levels[1.0], component_idx)
            if not comp_mask_1.any():
                continue

            scale = float(self.component_scale_factors.get(comp_name, 1.0))
            res_s = rasterized_levels[scale]
            comp_mask_s = get_component_mask(cc_levels[scale], component_idx)
            if not comp_mask_s.any():
                continue

            rows_s, cols_s = np.where(comp_mask_s)
            r_min_s, r_max_s = int(rows_s.min()), int(rows_s.max()) + 1
            c_min_s, c_max_s = int(cols_s.min()), int(cols_s.max()) + 1

            rows_1, cols_1 = np.where(comp_mask_1)
            r_min, r_max = int(rows_1.min()), int(rows_1.max()) + 1
            c_min, c_max = int(cols_1.min()), int(cols_1.max()) + 1

            h_comp_s, w_comp_s = r_max_s - r_min_s, c_max_s - c_min_s
            h_comp_crop = int(math.ceil(h_comp_s / self.patch_size)) * self.patch_size
            w_comp_crop = int(math.ceil(w_comp_s / self.patch_size)) * self.patch_size
            n_ph, n_pw = h_comp_crop // self.patch_size, w_comp_crop // self.patch_size

            pixel_mask, patch_mask, face_ids_comp, bary_comp = sample_component_pixels(
                r_min_s, c_min_s, h_comp_crop, w_comp_crop, self.patch_size, res_s, comp_mask_s, device
            )

            comp_ids_comp = torch.full_like(face_ids_comp, component_id)
            all_face_ids.append(face_ids_comp)
            all_bary_coords.append(bary_comp)
            all_comp_ids.append(comp_ids_comp)

            patch_pos = build_patch_positions(
                bbox=(r_min, c_min, r_max, c_max),
                n_patches_h=n_ph,
                n_patches_w=n_pw,
                patch_size=self.patch_size,
                device=device,
                scale_factor=scale,
            )
            if self.use_patch_mask:
                all_patch_positions.append(patch_pos[patch_mask])
            else:
                all_patch_positions.append(patch_pos)

            self.components.append(UVComponentInfo(
                component_id=component_id,
                bbox=(r_min, c_min, r_max, c_max),
                scale_factor=scale,
                scale_anchor="topleft",
                anchor_uv=(r_min_s, c_min_s),
                scaled_size=(h_comp_crop, w_comp_crop),
                n_patches_h=n_ph,
                n_patches_w=n_pw,
                pixel_mask=pixel_mask.to(device),
                patch_mask=patch_mask.to(device),
                orig_pixel_mask=torch.from_numpy(
                    comp_mask_1[r_min:r_max, c_min:c_max]
                ).to(device=device, dtype=torch.bool),
                n_valid_pixels=int(pixel_mask.sum()),
                patch_global_offset=patch_offset,
                pixel_global_offset=pixel_offset,
                patch_positions=patch_pos,
            ))
            patch_offset += int(patch_mask.sum()) if self.use_patch_mask else patch_pos.shape[0]
            pixel_offset += int(pixel_mask.sum())
            component_id += 1

        all_patch_positions_t = torch.cat(all_patch_positions, dim=0)
        self.register_buffer("_uv_patch_positions", all_patch_positions_t)

        # Inline build_sample_topology
        face_ids = torch.cat(all_face_ids, dim=0).to(device)
        bary_coords = torch.cat(all_bary_coords, dim=0).to(device)
        comp_ids = torch.cat(all_comp_ids, dim=0).to(device)

        anchors = barycentric_interpolate_vertices(
            neutral_vertices,
            face_ids.view(1, -1, 1, 1),
            bary_coords.view(1, -1, 1, 3),
            mhr_faces,
        ).view(1, 1, -1, 3)

        vertex_normals = calc_vertex_normals(neutral_vertices, mhr_faces)
        normals = barycentric_interpolate_vertices(
            vertex_normals,
            face_ids.view(1, -1, 1, 1),
            bary_coords.view(1, -1, 1, 3),
            mhr_faces,
        ).view(1, 1, -1, 3)
        normals = F.normalize(normals, dim=-1, eps=1e-6)

        self.sample_topology = UVSampleTopology(
            face_ids=face_ids,
            bary_coords=bary_coords,
            anchor_points=anchors,
            component_ids=comp_ids,
            sample_normals=normals,
        )
        self.register_buffer("_uv_anchor_points", self.sample_topology.anchor_points)
        self.register_buffer("_uv_sample_normals", self.sample_topology.sample_normals)
        self.register_buffer("_uv_tri_v_idx", mhr_faces[self.sample_topology.face_ids])
        self._num_verts = neutral_vertices.shape[0]

        # No global flat mask in CC mode; total_valid_pixels uses sample_topology.
        self.register_buffer("_uv_mask", None)

        return res

    @torch.no_grad()
    def setup(self, mhr_animatable_gs, lod: int, device):
        if self.mode == "naive":
            res = self._setup_naive_mode(mhr_animatable_gs, lod, device)
        elif self.mode == "connected_components":
            res = self._setup_cc_mode(mhr_animatable_gs, lod, device)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        if self._tri_faces is not None:
            f = self._tri_faces
            edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)
            edges = torch.sort(edges, dim=1).values
            self.register_buffer("_mesh_edges", torch.unique(edges, dim=0))

        self._setup_complete = True
        return res

    def scatter_to_uv_maps(self, flat_features: torch.Tensor) -> List[torch.Tensor]:
        """Scatter flat [B, L, C] features to UV maps for the current encoding mode."""
        if flat_features.ndim == 4:
            if flat_features.shape[1] != 1:
                raise AssertionError(f"Expected singleton pose dimension in [B, 1, L, C], got {flat_features.shape}.")
            flat_features = flat_features.squeeze(1)

        if self.mode == "naive":
            assert flat_features.ndim == 3, f"Expected flat features to be [B, L, C], got {flat_features.shape}."
            return [scatter_feature_uv(flat_features, self._uv_mask, uv_size=self.uv_size)]

        assert flat_features.ndim == 3, f"Expected flat features to be [B, L, C], got {flat_features.shape}."

        B, _, C = flat_features.shape
        uv_maps = []
        for comp in self.components:
            H_i, W_i = comp.scaled_size
            dense = torch.zeros(B, H_i * W_i, C, device=flat_features.device, dtype=flat_features.dtype)
            comp_feats = flat_features[:, comp.pixel_global_offset : comp.pixel_global_offset + comp.n_valid_pixels]
            dense[:, comp.pixel_mask.view(-1), :] = comp_feats
            uv_maps.append(dense.permute(0, 2, 1).view(B, C, H_i, W_i))
        return uv_maps

    def _encode_naive(self, uv_map: torch.Tensor, patch_embed: nn.Module) -> torch.Tensor:
        """Encode one UV map to tokens.
        Args:
            uv_map: [B, C, H, W], UV map
            patch_embed: patch embedding module
        Returns:
            [B, N_patches, D] tokens
        """
        tokens = patch_embed(uv_map)
        if tokens.ndim == 4:
            return tokens.flatten(2).permute(0, 2, 1)
        if tokens.ndim == 3:
            return tokens
        raise ValueError(
            f"Unsupported patch_embed output rank {tokens.ndim} for {type(patch_embed).__name__}."
        )

    def _encode_connected_components(self, uv_maps: List[torch.Tensor], patch_embed: nn.Module) -> torch.Tensor:
        """Encode UV maps to tokens.
        Args:
            uv_maps: List[B, C, H, W], UV maps
            patch_embed: patch embedding module
        Returns:
            [B, N_patches, D] tokens
        """

        components = self.components if self.mode == "connected_components" else [None]
        all_tokens = []
        for uv_map, comp in zip(uv_maps, components):
            tokens = self._encode_naive(uv_map, patch_embed)
            if comp is not None and self.use_patch_mask:
                patch_mask = comp.patch_mask.to(device=uv_map.device)
                tokens = tokens[:, patch_mask, :]
            all_tokens.append(tokens)
        return torch.cat(all_tokens, dim=1)

    def encode(self, flat_features: torch.Tensor, patch_embed: nn.Module) -> torch.Tensor:
        """Encode flat [B, L, C] features to [B, N_patches, D] tokens."""
        uv_maps = self.scatter_to_uv_maps(flat_features)

        if self.mode == "naive":
            assert len(uv_maps) == 1, f"Expected 1 UV map for naive mode, got {len(uv_maps)}."
            return self._encode_naive(uv_maps[0], patch_embed)
        elif self.mode == "connected_components":
            return self._encode_connected_components(uv_maps, patch_embed)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def scatter_tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """Scatter token features [B, N, C] into the full T x T UV token grid [B, T*T, C]."""
        if tokens.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got {tokens.shape}.")

        B, N, C = tokens.shape
        T = self.uv_size // self.patch_size
        grid = tokens.new_zeros(B, T * T, C)
        grid_indices = self.token_grid_indices.to(device=tokens.device)
        if N != grid_indices.shape[0]:
            raise ValueError(
                f"Token count mismatch: got {N} tokens, expected {grid_indices.shape[0]}."
            )
        grid[:, grid_indices, :] = tokens
        return grid

    def broadcast_token_feat_to_samples(self, token_feat: torch.Tensor) -> torch.Tensor:
        """Broadcast token offsets [B, N_token, C] to valid UV samples [B, L, C]."""
        if token_feat.ndim != 3:
            raise ValueError(f"Expected token_feat [B, N_token, C], got {token_feat.shape}.")

        B, n_tokens, C = token_feat.shape
        if n_tokens != self.total_patches:
            raise ValueError(
                f"Token count mismatch for broadcast: got {n_tokens}, expected {self.total_patches}."
            )

        if self.mode == "naive":
            T = self.uv_size // self.patch_size
            token_grid = self.scatter_tokens_to_grid(token_feat).view(B, T, T, C)
            full_map = token_grid.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)
            flat_map = full_map.reshape(B, self.uv_size * self.uv_size, C)
            if self._uv_mask is not None:
                return flat_map[:, self._uv_mask.view(-1), :]
            return flat_map

        if self.mode == "connected_components":
            all_sample_offsets: List[torch.Tensor] = []
            for comp in self.components:
                comp_patch_count = int(comp.patch_mask.sum().item()) if self.use_patch_mask else comp.n_patches
                comp_tokens = token_feat[
                    :, comp.patch_global_offset : comp.patch_global_offset + comp_patch_count, :
                ]

                if self.use_patch_mask:
                    patch_indices = torch.where(comp.patch_mask.to(token_feat.device))[0]
                    idx = patch_indices[None, :, None].expand(B, -1, C)
                    zeros = torch.zeros(B, comp.n_patches, C, device=token_feat.device, dtype=token_feat.dtype)
                    dense_comp_tokens = zeros.scatter(1, idx, comp_tokens)
                else:
                    dense_comp_tokens = comp_tokens

                yy, xx = torch.where(comp.pixel_mask.to(token_feat.device))
                patch_ids = (yy // self.patch_size) * comp.n_patches_w + (xx // self.patch_size)
                all_sample_offsets.append(dense_comp_tokens[:, patch_ids, :])
            return torch.cat(all_sample_offsets, dim=1)

        raise ValueError(f"Unknown mode: {self.mode}")

    def _decode_connected_components(
        self, tokens: torch.Tensor, decoder: nn.Module
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B = tokens.shape[0]
        out = None  # allocated after first decode to infer actual output channels
        component_uv_maps: List[Optional[torch.Tensor]] = [None] * len(self.components)

        # Components with identical token-grid shapes (e.g. mirrored left/right arm,
        # hand, leg, and the four foot parts) are decoded together in one batched
        # decoder forward call instead of one call per component.
        shape_groups: Dict[Tuple[int, int], List[int]] = {}
        for idx, comp in enumerate(self.components):
            shape_groups.setdefault((comp.n_patches_h, comp.n_patches_w), []).append(idx)

        for group_idxs in shape_groups.values():
            group_comps = [self.components[idx] for idx in group_idxs]
            group_tokens = [
                tokens[:, comp.patch_global_offset : comp.patch_global_offset + (
                    int(comp.patch_mask.sum().item()) if self.use_patch_mask else comp.n_patches
                )]
                for comp in group_comps
            ]
            decoded_group = decode_component_tokens_batched(
                decoder, group_tokens, group_comps, use_patch_mask=self.use_patch_mask
            )

            for idx, comp, decoded in zip(group_idxs, group_comps, decoded_group):
                component_uv_maps[idx] = decoded

                # GATHER valid pixels from the decoded map for this component
                comp_feats = decoded[:, :, comp.pixel_mask].permute(0, 2, 1)
                if out is None:
                    actual_out_channels = comp_feats.shape[2]
                    out = torch.zeros(B, self.total_valid_pixels, actual_out_channels, device=tokens.device, dtype=tokens.dtype)
                out[:, comp.pixel_global_offset : comp.pixel_global_offset + comp.n_valid_pixels] = comp_feats
        return out, component_uv_maps

    def smooth_pixel_deform_via_mesh(
        self,
        pixel_deform: torch.Tensor,   # [B, L, D]
        face_ids: Optional[torch.Tensor] = None,
        bary_coords: Optional[torch.Tensor] = None,
        faces: Optional[torch.Tensor] = None,
        num_verts: Optional[int] = None,
    ) -> torch.Tensor:
        """Fuses scatter_pixel_deform_to_vertices and interpolate_vertex_deform_to_pixels.
        Uses precomputed topology if arguments are not provided.
        
        Returns: [B, L, D] smoothed per-pixel deformation.
        """
        tri_v_idx = faces[face_ids] if faces is not None else None
        vertex_deform = self.scatter_pixel_deform_to_vertices(
            pixel_deform, face_ids=face_ids, bary_coords=bary_coords, tri_v_idx=tri_v_idx, num_verts=num_verts
        )
        return self.interpolate_vertex_deform_to_pixels(
            vertex_deform, face_ids=face_ids, bary_coords=bary_coords, tri_v_idx=tri_v_idx
        )

    def scatter_pixel_deform_to_vertices(
        self,
        pixel_deform: torch.Tensor, # [B, L, D]
        face_ids: Optional[torch.Tensor] = None,
        bary_coords: Optional[torch.Tensor] = None,
        tri_v_idx: Optional[torch.Tensor] = None,
        num_verts: Optional[int] = None,
    ) -> torch.Tensor:
        """Scatter per-pixel deformation to vertices via barycentric weights."""
        B, L, D = pixel_deform.shape
        device = pixel_deform.device

        if face_ids is None and tri_v_idx is None:
            face_ids = self.sample_topology.face_ids if self.mode == "connected_components" else self._uv_pix_to_face.squeeze(0).squeeze(-1)[self._uv_mask]
        
        if tri_v_idx is None:
            tri_v_idx = self._uv_tri_v_idx if face_ids is None else self._tri_faces[face_ids]
            
        if bary_coords is None:
            bary_coords = self.sample_topology.bary_coords if self.mode == "connected_components" else self._uv_bary_coords.squeeze(0)[self._uv_mask]

        if num_verts is None:
            num_verts = self._num_verts

        bary_coords = bary_coords.to(device=device, dtype=pixel_deform.dtype)

        vertex_accum = torch.zeros(B, num_verts, D, device=device, dtype=pixel_deform.dtype)
        weight_accum = torch.zeros(B, num_verts, 1, device=device, dtype=pixel_deform.dtype)

        for j in range(3):
            v_idx = tri_v_idx[:, j]
            w_j = bary_coords[:, j]
            weighted_deform = pixel_deform * w_j.view(1, L, 1)
            w_contrib = w_j.view(1, L, 1).expand(B, L, 1)
            idx_expand_d = v_idx.view(1, L, 1).expand(B, L, D)
            idx_expand_1 = v_idx.view(1, L, 1).expand(B, L, 1)
            vertex_accum.scatter_add_(1, idx_expand_d, weighted_deform)
            weight_accum.scatter_add_(1, idx_expand_1, w_contrib)

        vertex_deform = vertex_accum / (weight_accum + 1e-8)
        return vertex_deform

    def interpolate_vertex_deform_to_pixels(
        self,
        vertex_deform: torch.Tensor, # [B, V, D]
        face_ids: Optional[torch.Tensor] = None,
        bary_coords: Optional[torch.Tensor] = None,
        tri_v_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Interpolate per-vertex deformation to pixels via barycentric weights."""
        B, _, D = vertex_deform.shape
        device = vertex_deform.device

        if face_ids is None and tri_v_idx is None:
            face_ids = self.sample_topology.face_ids if self.mode == "connected_components" else self._uv_pix_to_face.squeeze(0).squeeze(-1)[self._uv_mask]
            
        if tri_v_idx is None:
            tri_v_idx = self._uv_tri_v_idx if face_ids is None else self._tri_faces[face_ids]

        if bary_coords is None:
            bary_coords = self.sample_topology.bary_coords if self.mode == "connected_components" else self._uv_bary_coords.squeeze(0)[self._uv_mask]
        
        bary_coords = bary_coords.to(device=device, dtype=vertex_deform.dtype)
        L = tri_v_idx.shape[0]

        result = torch.zeros(B, L, D, device=device, dtype=vertex_deform.dtype)
        for j in range(3):
            v_idx = tri_v_idx[:, j]
            v_vals = vertex_deform[:, v_idx, :]  # [B, L, D]
            result = result + bary_coords[:, j].view(1, L, 1) * v_vals

        return result

    def interpolate_vertex_deform_to_tokens(
        self,
        vertex_deform: torch.Tensor, # [B, V, 3]
        token_face_ids: torch.Tensor, # [N]
        token_bary_coords: torch.Tensor, # [N, 3]
    ) -> torch.Tensor:
        """Interpolate vertex deformation to token anchor positions using token topology."""
        B, _, D = vertex_deform.shape
        device = vertex_deform.device
        N = token_face_ids.shape[0]
        
        tri_v_idx = self._tri_faces[token_face_ids] # [N, 3]
        bary_coords = token_bary_coords.to(device=device, dtype=vertex_deform.dtype)
        
        result = torch.zeros(B, N, D, device=device, dtype=vertex_deform.dtype)
        for j in range(3):
            v_idx = tri_v_idx[:, j]
            v_vals = vertex_deform[:, v_idx, :] # [B, N, D]
            result = result + bary_coords[:, j].view(1, N, 1) * v_vals
        return result

    def token_feature_to_vertices(
        self,
        token_features: torch.Tensor,  # [B, N, D]
    ) -> torch.Tensor:
        """Lifts token features to vertices directly."""
        sample_features = self.broadcast_token_feat_to_samples(token_features)
        return self.scatter_pixel_deform_to_vertices(sample_features)
    
    def smooth_token_deform_to_pixels(
        self,
        token_features: torch.Tensor, # [B, N, D]
    ) -> torch.Tensor:
        """Broadcast token offsets to UV samples, then apply exact mesh smoothing."""
        vertex_features = self.token_feature_to_vertices(token_features)
        return self.interpolate_vertex_deform_to_pixels(vertex_features)
    
    def _decode_naive(
        self,
        tokens: torch.Tensor,
        decoder: nn.Module,
        uv_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode naive mode tokens to flattened 3DGS features and UV maps.
        Args:
            tokens: [B, N, D]
            decoder: decoder module
            uv_mask: optional valid-pixel mask override
        Returns:
            feats: [B, L, C] valid-pixel features
            uv_maps: [B, C, H, W]
        """
        B, N, D = tokens.shape
        grid = self.uv_size // self.patch_size
        token_map = tokens.permute(0, 2, 1).view(B, D, grid, grid)
        uv_map = decoder(token_map)
        flat = uv_map.flatten(2).permute(0, 2, 1)
        mask = self.uv_mask if uv_mask is None else uv_mask
        feats = flat[:, mask, :] if (mask is not None and mask.dtype == torch.bool) else flat
        return feats, [uv_map]

    def decode(
        self,
        tokens: torch.Tensor,
        decoder: nn.Module,
        uv_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
        """Decode tokens to flattened 3DGS features and UV maps.
        Args:
            tokens: [B, N, D]
            decoder: decoder module
            uv_mask: optional valid-pixel mask override for naive mode
        Returns:
            feats: [B, L, C] valid-pixel features
            uv_maps: [B, C, H, W] for naive mode, List[[B, C, H_i, W_i]] for CC mode
        """
        assert tokens.ndim == 3, f"Tokens must be 3D, got {tokens.shape}"

        if self.mode == "naive":
            return self._decode_naive(tokens, decoder, uv_mask=uv_mask)
        elif self.mode == "connected_components":
            return self._decode_connected_components(tokens, decoder)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def aggregate_patch_points(self, flat_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool flat features [B, L, C] -> (patch_features [B, N, C], patch_mask [B, N]).

        Works for any C-dimensional flat features (xyz coords, visibility, network features, etc.)
        in both naive and connected_components modes.
        """
        B, _, C = flat_features.shape
        if self.mode == "naive":
            uv_map = scatter_feature_uv(flat_features, self.uv_mask, uv_size=self.uv_size)
            uv_mask_map = self.uv_mask.view(1, 1, self.uv_size, self.uv_size).to(uv_map).expand(B, -1, -1, -1)
            pooled = F.avg_pool2d(
                torch.cat([uv_map * uv_mask_map, uv_mask_map], dim=1),
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            patch_mask_avg = pooled[:, C:C + 1, :, :]
            patch_features = pooled[:, :C, :, :] / patch_mask_avg.clamp_min(1e-6)
            patch_features = torch.where(patch_mask_avg > 0, patch_features, torch.zeros_like(patch_features))
            return (
                patch_features.flatten(2).permute(0, 2, 1),
                (patch_mask_avg.flatten(2).permute(0, 2, 1) > 0).squeeze(-1),
            )

        all_values, all_masks = [], []
        for comp in self.components:
            comp_feat = flat_features[:, comp.pixel_global_offset : comp.pixel_global_offset + comp.n_valid_pixels]
            dense = torch.zeros(B, comp.scaled_size[0] * comp.scaled_size[1], C,
                                device=flat_features.device, dtype=flat_features.dtype)
            dense[:, comp.pixel_mask.view(-1), :] = comp_feat
            comp_map = dense.permute(0, 2, 1).view(B, C, *comp.scaled_size)
            mask = comp.pixel_mask.view(1, 1, *comp.scaled_size).to(comp_map).expand(B, -1, -1, -1)
            pooled = F.avg_pool2d(
                torch.cat([comp_map * mask, mask], dim=1),
                kernel_size=self.patch_size, stride=self.patch_size,
            )
            p_mask_avg = pooled[:, C:C + 1, :, :]
            p_values = pooled[:, :C, :, :] / p_mask_avg.clamp_min(1e-6)
            p_values = torch.where(p_mask_avg > 0, p_values, torch.zeros_like(p_values))
            p_values = p_values.flatten(2).permute(0, 2, 1)
            p_mask = comp.patch_mask.view(1, -1).expand(B, -1)
            if self.use_patch_mask:
                p_values = p_values[:, comp.patch_mask, :]
                p_mask = torch.ones(
                    B,
                    int(comp.patch_mask.sum().item()),
                    device=flat_features.device,
                    dtype=torch.bool,
                )
            all_values.append(p_values)
            all_masks.append(p_mask)
        return torch.cat(all_values, dim=1), torch.cat(all_masks, dim=1)
