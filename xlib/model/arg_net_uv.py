"""
Multi-view SAM3D Avatar UV Model.

Uses the neutral-pose mesh: unwraps it to UV with xatlas, interpolates position and normal
maps, and encodes them to avatar tokens. RoPE-3D transformer blocks refine tokens with
per-view pose and image conditioning fused across views; a UV decoder produces UV-space
feature maps and all UV pixels are passed through GSLayer to generate dense 3D Gaussian
attributes (residual on top of the canonical anchor). Single-view use is the N_views=1
case of the same path (this class absorbed the former single-view ARGNetBaseUV).
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from xlib.render.camera import GSCamera
from xlib.model_input_utils import create_single_view_batch, slice_batch, slice_batch_views
from xlib.model.arg_net import ARGNetBase, compute_scale_base
from xlib.model.avatar_conditioner import AvatarConditioner
from xlib.model.multimodal_transformer import Attention, MVAvatarBlock
from xlib.model.gs_layer import GSLayer
from xlib.geometry.uv_encoding import (
    UVEncoding,
    UVConditionCollector,
    gs_to_avatar_tokens,
)
from xlib.model.common import build_uv_decoder, spatial_tokens_to_feature_map
from xlib.model.mhr_animatable_gaussians import animate_gs, extract_lbs_feat
from xlib.model.rope_3d import compute_fac


def _is_cuda_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.lower().startswith("cuda")
    return False


class MultiviewTokenFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = Attention(embedding_dim=embed_dim, num_heads=num_heads)

    def forward(self, base_tokens: torch.Tensor, view_tokens: torch.Tensor) -> torch.Tensor:
        """
        base_tokens: [B, L, C]
        view_tokens: [B, V, L, C]
        """
        B, V, L, C = view_tokens.shape

        # Reshape for cross-attention
        # query: [B*L, 1, C], key/value: [B*L, V, C]
        q = base_tokens.reshape(B * L, 1, C)
        kv = view_tokens.permute(0, 2, 1, 3).reshape(B * L, V, C)

        x = self.attn(q, kv, kv)

        x = x.reshape(B, L, C)

        return x


class ARGNetUV(ARGNetBase):
    """True multi-view UV avatar variant with shared canonical decoding."""

    def __init__(self, cfg: Any, avatar_cfg: Optional[dict] = None):
        super(ARGNetBase, self).__init__(cfg)
        self._initialize_base_avatar(cfg, avatar_cfg or {})
        self._initialize_avatar(avatar_cfg or {})

    def _gs_base_dim(self) -> int:
        return 14

    def _initialize_avatar(self, avatar_cfg: dict) -> None:
        """Build the avatar model from small builders.

        Each builder constructs its own modules and appends its own
        ``trainable_modules`` entries; subclasses override only the builder
        whose content differs. No builder may remove entries added by another.
        """
        self.image_cond_dim = self.backbone.embed_dims
        uv_cfg = avatar_cfg["uv"]
        self.uv_token_size = uv_cfg["uv_size"] // uv_cfg["patch_size"]

        mm_cfg = dict(avatar_cfg["multimodal_avatar_transformer"])
        self.embed_dim = mm_cfg["embed_dim"]
        self._normalize_attention_modes(mm_cfg)

        self.trainable_modules = []
        self._build_uv_core(avatar_cfg)
        self._build_dense_decoder(avatar_cfg)
        self._build_transformer_blocks(avatar_cfg, mm_cfg)

        self._setup_neutral_uv_barycentric('cuda')
        self._init_gs_tpose_init_dic_params()

        self._build_conditioners(avatar_cfg, mm_cfg)
        self._build_pose_dep(avatar_cfg)

    def _build_uv_core(self, avatar_cfg: dict) -> None:
        """UV encoding, canonical/posed token conditioning, and patch embeds."""
        uv_cfg = avatar_cfg["uv"]
        patch_size = uv_cfg["patch_size"]
        uv_size = uv_cfg.get("uv_size", uv_cfg.get("img_size", self.uv_token_size * patch_size))
        gs_layer_cfg = avatar_cfg["gs_layer"]
        sh_degree = gs_layer_cfg.get("sh_degree", 0)
        uv_out_channels = gs_layer_cfg["in_channels"]

        cond_cfg = avatar_cfg.get("UV_token_cond", {})
        canon_cond_cfg = cond_cfg.copy()
        canon_cond_cfg["lbs"] = False
        self.canonical_token_cond = UVConditionCollector(
            cfg=canon_cond_cfg,
            embed_dim=self.embed_dim,
            patch_size=patch_size,
            sh_degree=sh_degree,
        )

        posed_cond_cfg = {
            "normal": False,
            "lbs": True,
            "neutral_anchor": False,
            "neutral_avatar": False,
        }
        self.posed_token_cond = UVConditionCollector(
            cfg=posed_cond_cfg,
            embed_dim=self.embed_dim,
            patch_size=patch_size,
            sh_degree=sh_degree,
        )

        self.uv_encoding = UVEncoding(
            mode=uv_cfg["encoding_mode"],
            uv_size=uv_size,
            patch_size=patch_size,
            embed_dim=self.embed_dim,
            out_channels=uv_out_channels,
            in_channels=self.canonical_token_cond.total_dim,
            component_scale_factors=uv_cfg.get("component_scale_factors", {}),
            use_patch_mask=uv_cfg.get("use_patch_mask", True),
        )
        self.canonical_token_cond.set_uv_encoding(self.uv_encoding)
        self.posed_token_cond.set_uv_encoding(self.uv_encoding)

        self.canonical_patch_embed = nn.Conv2d(
            self.canonical_token_cond.total_dim,
            self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.posed_patch_embed = nn.Conv2d(
            self.posed_token_cond.total_dim,
            self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.trainable_modules.extend([
            "canonical_token_cond",
            "posed_token_cond",
            "canonical_patch_embed",
            "posed_patch_embed",
        ])

    def _build_dense_decoder(self, avatar_cfg: dict) -> None:
        """Dense UV decoder + GS head (PixelDiT overrides to skip the decoder)."""
        uv_cfg = avatar_cfg["uv"]
        patch_size = uv_cfg["patch_size"]
        gs_layer_cfg = avatar_cfg["gs_layer"]
        self.uv_decoder = build_uv_decoder(
            decoder_type=uv_cfg["decoder_type"],
            embed_dim=self.embed_dim,
            out_channels=gs_layer_cfg["in_channels"],
            patch_size=patch_size,
            upsample_factor=uv_cfg.get("decoder_upsample_factor", patch_size),
        )
        self.trainable_modules.append("uv_decoder")

        self.gshead = GSLayer(**gs_layer_cfg)
        self.trainable_modules.append("gshead")

    def _build_transformer_blocks(self, avatar_cfg: dict, mm_cfg: dict) -> None:
        """RoPE-3D avatar blocks and cross-view token fusion."""
        fusion_cfg = avatar_cfg.get("multiview_fusion", {})
        cond_dim = mm_cfg.get("cond_dim", self.backbone.embed_dims)
        num_blocks = mm_cfg["depth"]
        self.rope3d_blocks = nn.ModuleList([
            MVAvatarBlock(
                embed_dim=self.embed_dim,
                cond_dim=cond_dim,
                num_heads=mm_cfg.get("num_heads", 16),
                mlp_ratio=mm_cfg.get("mlp_ratio", 4.0),
                drop_path_rate=mm_cfg.get("drop_path_rate", 0.0),
                dropout=mm_cfg.get("dropout_p", 0.0),
                rope_theta=mm_cfg.get("rope_theta", 10000.0),
                feat_sizes=(self.uv_token_size, self.uv_token_size),
                sa3d_type=self.sa3d_type,
                token_to_image_attn_type=self.token_to_image_attn_type,
            )
            for _ in range(num_blocks)
        ])
        self.multiview_fusion = MultiviewTokenFusion(
            embed_dim=self.embed_dim,
            num_heads=fusion_cfg.get("num_heads", 8),
            mlp_ratio=fusion_cfg.get("mlp_ratio", 4.0),
        )
        self.trainable_modules.extend(["rope3d_blocks", "multiview_fusion"])

    def _build_conditioners(self, avatar_cfg: dict, mm_cfg: dict) -> None:
        """Conditioner for KV tokens (replaces previous hr_backbone setup)."""
        cond_cfg = avatar_cfg["avatar_cond"]
        self.avatar_conditioner = AvatarConditioner(
            cfg=cond_cfg,
            cond_dim=mm_cfg['cond_dim'],
            backbone_dim=self.image_cond_dim,
            mhr_animatable_gs=self.mhr_animatable_gs,
            image_mean=self.image_mean,
            image_std=self.image_std,
            backbone_dtype=self.backbone_dtype,
        )
        self.trainable_modules.append("avatar_conditioner")

    def _build_pose_dep(self, avatar_cfg: dict) -> None:
        """Pose conditioning projection and optional pose-dependent deformation."""
        # pose projection is for conditioning, not only for pose-dependent deformation
        D = self.embed_dim
        self.pose_proj = nn.Sequential(
            nn.Linear(204, D),
            nn.GELU(),
            nn.Linear(D, D),
        )

        # Optional pose-dependent deformation (enabled by presence of pose_dep key in config)
        pose_dep_cfg = avatar_cfg.get("pose_dep", None)
        self.pose_dep_enabled = pose_dep_cfg is not None
        self.deform_shape = False
        if not self.pose_dep_enabled:
            return
        self.pose_dep_cfg = pose_dep_cfg
        self.deform_shape = pose_dep_cfg.get("deform_shape", False)
        mlp_depth = pose_dep_cfg["mlp_depth"]
        mlp_layers = []
        for _ in range(mlp_depth):
            mlp_layers.append(nn.Linear(D, D))
            mlp_layers.append(nn.GELU())
        self.pose_dep_mlp = nn.Sequential(*mlp_layers)
        self._build_posedep_uv_decoder(avatar_cfg)

        self.pixel_pose_offset_max_step = pose_dep_cfg["pixel_offset_max_step"]
        self.token_pose_offset_enabled = "token_offset_max_step" in pose_dep_cfg
        if self.token_pose_offset_enabled:
            self.token_pose_offset_max_step = pose_dep_cfg["token_offset_max_step"]
            self._build_token_pose_offset_heads()
        self.trainable_modules.extend(["pose_proj", "pose_dep_mlp"])

    def _build_posedep_uv_decoder(self, avatar_cfg: dict) -> None:
        """Pose-dependent dense UV decoder (PixelDiT overrides to skip)."""
        uv_cfg = avatar_cfg["uv"]
        patch_size = uv_cfg["patch_size"]
        posedep_out_ch = 10 if self.deform_shape else 3
        self.posedep_uv_decoder = build_uv_decoder(
            decoder_type=uv_cfg["decoder_type"],
            embed_dim=self.embed_dim,
            out_channels=posedep_out_ch,
            patch_size=patch_size,
            upsample_factor=uv_cfg.get("decoder_upsample_factor", patch_size),
        )
        if self.deform_shape:
            with torch.no_grad():
                self.posedep_uv_decoder.conv_out.bias[6] = 1.0  # quaternion w=1
        self.trainable_modules.append("posedep_uv_decoder")

    def _build_token_pose_offset_heads(self) -> None:
        """Single token pose-offset head (TokenCascade overrides with per-block heads)."""
        self.token_pose_offset_proj = self._build_token_pose_offset_proj()
        self.trainable_modules.append("token_pose_offset_proj")

    def _build_token_pose_offset_proj(self) -> nn.Module:
        D = self.embed_dim
        token_proj_mlp_depth = self.pose_dep_cfg["token_proj_mlp_depth"]
        posedep_out_ch = 10 if self.deform_shape else 3
        proj_layers = []
        for _ in range(token_proj_mlp_depth):
            proj_layers.append(nn.Linear(D, D))
            proj_layers.append(nn.GELU())
        proj_layers.append(nn.Linear(D, posedep_out_ch))

        proj = nn.Sequential(*proj_layers)
        final_linear = proj[-1]
        final_linear.weight.data.normal_(mean=0.0, std=0.001)
        final_linear.bias.data.zero_()
        if self.deform_shape:
            final_linear.bias.data[6] = 1.0
        return proj

    def _build_feature_cameras(
        self,
        cameras: List[GSCamera],
        feat_h: int,
        feat_w: int,
    ) -> List[GSCamera]:
        feature_cameras = []
        for cam in cameras:
            K = cam.K.clone()
            K[0] *= float(feat_w) / float(cam.image_width)
            K[1] *= float(feat_h) / float(cam.image_height)
            feature_cameras.append(
                GSCamera.from_extrinsics_intrinsics(
                    cam.world_view_transform.T.to(device=K.device, dtype=K.dtype),
                    K=K,
                    full_K=K,
                    full_image_width=feat_w,
                    full_image_height=feat_h,
                    image_width=feat_w,
                    image_height=feat_h,
                    znear=cam.znear,
                    zfar=cam.zfar,
                )
            )
        return feature_cameras

    def _normalize_attention_modes(self, mm_cfg: dict) -> None:
        self.token_to_image_attn_type = mm_cfg["token_to_image_attn_type"]
        self.sa3d_type = mm_cfg["sa3d_type"]
        allowed_t2i = {"naive", "fac-rope", "projection"}
        allowed_sa3d = {"none", "canonical-rope", "animation-rope"}
        assert self.token_to_image_attn_type in allowed_t2i, \
            f"Invalid token_to_image_attn_type: {self.token_to_image_attn_type}"
        assert self.sa3d_type in allowed_sa3d, f"Invalid sa3d_type: {self.sa3d_type}"

        self.needs_absolute_token_pos = self.token_to_image_attn_type == "naive"
        self.needs_fac = self.token_to_image_attn_type == "fac-rope"
        self.needs_projection_fusion = self.token_to_image_attn_type == "projection"
        self.needs_sa3d_coords = self.sa3d_type != "none"
        self.needs_render = self.needs_fac or self.needs_projection_fusion or self.needs_sa3d_coords

    def _init_gs_tpose_init_dic_params(self):
        """Calculate the scale_base as the initial distance between neutral avatar points."""
        device = self.device
        dtype = torch.float32

        # Compute initial UV-posed state to determine distance-based scale_base if needed
        L = self.uv_encoding.total_valid_pixels
        _sh_degree = self.avatar_cfg["gs_layer"].get("sh_degree", 0)
        _D = (_sh_degree + 1) ** 2
        tpose_gs_dic = {
            "offset_xyz": torch.zeros(1, 1, L, 3, device=device, dtype=dtype),
            "scale": torch.ones(1, 1, L, 1, device=device, dtype=dtype) * 1e-5,
            "rotation": torch.zeros(1, 1, L, 4, device=device, dtype=dtype),
            "opacity": torch.zeros(1, 1, L, 1, device=device, dtype=dtype),
            "sh": torch.zeros(1, 1, L, _D, 3, device=device, dtype=dtype),
        }

        gs_posed, _ = animate_gs(
            self.mhr_animatable_gs,
            tpose_gs_dic,
            dict(
                shape_params=torch.zeros(1, 45, device=device),
                model_params=torch.zeros(1, 1, 204, device=device),
                expr_params=torch.zeros(1, 1, 72, device=device)),
            to_colmap=True
        )

        scale_base = compute_scale_base(gs_posed.xyz[0, 0]).to(device=device, dtype=dtype)
        self.gshead.set_scale_base(scale_base)

        sh_degree = self.avatar_cfg["gs_layer"].get("sh_degree", 0)
        self.uv_encoding.init_gs_tpose_params(sh_degree=sh_degree)

    def _predict_token_deform(
        self,
        posed_avatar_token: torch.Tensor,
    ) -> torch.Tensor:
        """Predict token-level pose-dependent deformation offsets.

        Returns raw token offsets of shape [B, N, 3] without broadcasting to UV pixels.
        Only callable when ``token_pose_offset_enabled=True``.

        Intended for use in cascade loops that need a fast per-block deformation estimate
        for re-rendering token GS before the next cross-attention block, while full
        pixel-level decoding is deferred to the final pass.

        Args:
            posed_avatar_token: [B, N, C] = pose_dep_mlp(avatar_tokens + pose_embed[:, None]),
                pre-computed by the caller.

        Returns:
            token_offsets: [B, N, 3] pose-dependent XYZ offsets at token resolution.
        """
        token_offsets = self.token_pose_offset_proj(posed_avatar_token)  # [B, N, 3]
        if self.token_pose_offset_max_step is not None:
            token_offsets = torch.tanh(token_offsets) * float(self.token_pose_offset_max_step)
        return token_offsets

    def _decode_dense_outputs(
        self,
        avatar_tokens: torch.Tensor,
        gs_tpose_dic_anchor: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Decode dense canonical GS as pixel residual on top of anchor."""
        pixel_feats, _ = self.uv_encoding.decode(avatar_tokens, decoder=self.uv_decoder)
        pixel_gs_raw = self.gshead.forward_raw(pixel_feats.unsqueeze(1))
        pixel_gs_delta = self.gshead.presum(pixel_gs_raw)
        final_gs_raw = self.gshead.sum_residuals([gs_tpose_dic_anchor, pixel_gs_delta])
        final_gs = self.gshead.activate(final_gs_raw)
        final_gs["xyz"] = gs_tpose_dic_anchor["xyz"]
        return {
            "output_canonical_gs_dic": final_gs,
            "pixel_gs_residual": pixel_gs_raw,
        }

    def _setup_neutral_uv_barycentric(self, device='cuda') -> None:
        """Unwrap neutral-pose MHR mesh and cache UV data. Delegates to UVEncoding."""
        lod = self.mhr_animatable_gs.lod
        self.uv_encoding.setup(self.mhr_animatable_gs, lod=lod, device=device)
        self.mhr_animatable_gs.set_uv_barycentric_buffers(
            self.uv_encoding.sample_topology
        )

    def to(self, device) -> "ARGNetUV":
        super(ARGNetBase, self).to(device)
        self.mhr_animatable_gs = self.mhr_animatable_gs.to(device)
        # Defer UV/barycentric setup until we're on CUDA (nvdiffrast is CUDA-only)
        if (
            _is_cuda_device(device)
            and getattr(self, "uv_encoding", None) is not None
            and not self.uv_encoding.is_setup
        ):
            self._setup_neutral_uv_barycentric(device)
        return self

    def compute_target_deform_dic(
        self,
        posed_avatar_token: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute pose-dependent deform_dic for a target pose.

        Only callable when pose_dep_enabled=True (pose_dep key present in config).

        Args:
            posed_avatar_token: [B, L, C] = pose_dep_mlp(avatar_tokens + pose_embed[:, None]),
                pre-computed by the caller (batch expansion included if needed).
        """
        raw_pixel, _ = self.uv_encoding.decode(posed_avatar_token, decoder=self.posedep_uv_decoder)
        # [B, L, 3] or [B, L, 10]

        pixel_offset_pose = raw_pixel[..., :3] * self.pixel_pose_offset_max_step

        if self.token_pose_offset_enabled:
            token_raw = self._predict_token_deform(posed_avatar_token)  # [B, N, 3|10]
            token_on_pixels = self.uv_encoding.broadcast_token_feat_to_samples(token_raw)  # [B, L, 3|10]
            offset_pose = pixel_offset_pose + token_on_pixels[..., :3]
        else:
            offset_pose = pixel_offset_pose

        deform_dic = {"offset_pose": offset_pose, "pixel_offset_pose": pixel_offset_pose}

        if self.deform_shape:
            if self.token_pose_offset_enabled:
                deform_dic["offset_pose_scale_"] = raw_pixel[..., 3:6] + token_on_pixels[..., 3:6]
                deform_dic["offset_pose_rotation"] = raw_pixel[..., 6:10] + token_on_pixels[..., 6:10]
            else:
                deform_dic["offset_pose_scale_"] = raw_pixel[..., 3:6]
                deform_dic["offset_pose_rotation"] = raw_pixel[..., 6:10]

        return deform_dic

    def _prepare_all_pose_outputs(self, batch: Dict, input_pose_dic: Dict) -> List[Dict]:
        total_views = batch["images"].shape[1]
        all_pose_outputs = []

        def _view_tensor(value, v_idx):
            if not isinstance(value, torch.Tensor):
                return value
            if value.ndim >= 2 and value.shape[1] == total_views:
                return value[:, v_idx]
            return value

        for v_idx in range(total_views):
            shape_params = _view_tensor(input_pose_dic["shape"], v_idx)
            model_params = _view_tensor(input_pose_dic["mhr_model_params"], v_idx)
            expr_params = _view_tensor(input_pose_dic["face"], v_idx)
            if model_params.ndim == 2:
                model_params = model_params.unsqueeze(1)
            if expr_params.ndim == 2:
                expr_params = expr_params.unsqueeze(1)

            pose_dic = {
                "shape_params": shape_params,
                "model_params": model_params,
                "expr_params": expr_params,
            }
            pred_camera = {
                "pred_cam_t": _view_tensor(input_pose_dic["pred_cam_t"], v_idx),
            }
            if "cam_rot" in input_pose_dic:
                pred_camera["cam_rot"] = _view_tensor(input_pose_dic["cam_rot"], v_idx)
            if "cam_trans" in input_pose_dic:
                pred_camera["cam_trans"] = _view_tensor(input_pose_dic["cam_trans"], v_idx)
            all_pose_outputs.append({
                "mhr": {
                    key: _view_tensor(value, v_idx)
                    for key, value in input_pose_dic.items()
                    if isinstance(value, torch.Tensor)
                },
                "pose_dic": pose_dic,
                "pred_camera": pred_camera,
                "pred_cam_t": pred_camera["pred_cam_t"],
            })
        return all_pose_outputs

    @torch.no_grad()
    def _prepare_two_phase_pose_lbs(
        self,
        batch: Dict,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute all-view target poses and LBS cache for two-phase inference."""
        device = batch["images"].device
        input_pose_dic = self.get_input_pose_dic(batch)
        factorized_pose_dic = self._ensure_factorized_input_pose_dic(dict(input_pose_dic))
        all_pose_outputs = self._prepare_all_pose_outputs(batch, factorized_pose_dic)
        all_pose_dics = [view_output["pose_dic"] for view_output in all_pose_outputs]

        # Only precompute LBS for input views; sup-view LBS is computed lazily per-view
        # in render_in_raw_space to avoid OOM when the batch contains many sup views.
        ivm = batch.get("input_view_mask")
        if ivm is not None:
            if isinstance(ivm, torch.Tensor) and ivm.ndim == 2:
                ivm = ivm[0]
            input_view_indices = torch.where(ivm.bool())[0].tolist()
        else:
            input_view_indices = [0]
        input_pose_dics = [all_pose_dics[i] for i in input_view_indices]
        all_lbs_cache = self.mhr_animatable_gs.precompute_lbs_cache(input_pose_dics)

        B, V = batch["images"].shape[:2]
        all_model_params: List[torch.Tensor] = []
        all_shape_params: List[torch.Tensor] = []
        all_expr_params: List[torch.Tensor] = []
        for v_idx in range(V):
            for b_idx in range(B):
                pose_v = slice_batch(all_pose_outputs[v_idx]["pose_dic"], b_idx)
                all_model_params.append(pose_v["model_params"].reshape(1, -1))
                all_shape_params.append(pose_v["shape_params"].reshape(1, -1))
                all_expr_params.append(pose_v["expr_params"].reshape(1, -1))

        poses = {
            "model_params": torch.cat(all_model_params, dim=0).to(device),
            "shape_params": torch.cat(all_shape_params, dim=0).to(device),
            "expr_params": torch.cat(all_expr_params, dim=0).to(device),
        }
        return poses, all_lbs_cache

    def two_phase_model_forward(
        self,
        batch: Dict,
        output_on_cpu: bool = False,
        debug_save_folder: str = "",
    ) -> Dict:
        """Two-phase forward pass for memory-efficient cache generation.

        Args:
            output_on_cpu: If True, move ``output_canonical_gs_dic`` and
                ``offset_pose`` in the last pass to CPU before returning so
                callers can free GPU activations (del result + empty_cache)
                and let render loops move tensors back to device on demand.
        """
        device = self.device
        V = batch["images"].shape[1]

        input_view_mask = batch.get("input_view_mask")
        if input_view_mask is not None:
            if isinstance(input_view_mask, torch.Tensor) and input_view_mask.ndim == 2:
                input_view_mask = input_view_mask[0]
            input_view_mask = input_view_mask.bool()
            input_view_indices = torch.where(input_view_mask)[0].tolist()
        else:
            input_view_mask = torch.zeros(V, dtype=torch.bool, device=device)
            input_view_mask[0] = True
            input_view_indices = [0]
        sup_view_indices = [v_idx for v_idx in range(V) if v_idx not in set(input_view_indices)]

        all_pose_all, all_lbs_cache = self._prepare_two_phase_pose_lbs(batch)

        if len(input_view_indices) == 1:
            input_batch = create_single_view_batch(batch, input_view_indices[0], device)
        else:
            input_batch = slice_batch_views(batch, input_view_mask)
        input_batch["input_view_indices"] = input_view_indices
        input_batch["pose_all"] = {
            key: value[input_view_indices]
            for key, value in all_pose_all.items()
        }
        input_batch["lbs_cache_all"] = all_lbs_cache

        result = self(batch=input_batch, debug_save_folder=debug_save_folder)
        result["pose_all"] = all_pose_all

        if not sup_view_indices:
            return result
        if not self.pose_dep_enabled:
            return result

        pass_output = result["passes"][-1]
        avatar_tokens = pass_output.get("avatar_tokens")
        if avatar_tokens is None:
            print("[two-phase] Warning: avatar_tokens not in model output; falling back to single-phase deform.")
            return result

        input_offset_pose = pass_output.get("offset_pose")
        if output_on_cpu:
            input_offset_pose = input_offset_pose.cpu()
        gs_tpose_dic = pass_output.get("input_canonical_gs_dic")
        sup_offsets = []
        for v_idx in sup_view_indices:
            pose_params = all_pose_all["model_params"][v_idx:v_idx + 1]
            pose_embed_v = self.pose_proj(pose_params).unsqueeze(1)
            posed_token = self.pose_dep_mlp(avatar_tokens + pose_embed_v)
            deform_v = self.compute_target_deform_dic(posed_token)
            offset_pose = deform_v["offset_pose"]
            if output_on_cpu:
                offset_pose = offset_pose.cpu()
            sup_offsets.append(offset_pose)

        all_offsets = []
        if input_offset_pose is not None:
            all_offsets.append(input_offset_pose)
        all_offsets.extend(offset for offset in sup_offsets if offset is not None)
        if all_offsets:
            result["passes"][-1]["offset_pose"] = torch.cat(all_offsets, dim=0)

        return result

    def _prepare_gs_tpose(self, batch_size: int, gs_tpose_dic: Optional[Dict] = None) -> Dict:
        """Create model input-ready GS dict from placeholders or unactivate provided dict."""
        device = self.device
        if gs_tpose_dic is None:
            gs_tpose_dic = self.uv_encoding.get_gs_tpose_init_dic(batch_size=batch_size, device=device)
            self.gshead.activate(gs_tpose_dic)
        else:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                self.gshead.unactivate(gs_tpose_dic)
                if gs_tpose_dic["offset_xyz"].shape[2] == self.mhr_animatable_gs.num_verts:
                    anchor_points = self.mhr_animatable_gs.query_points[None, None].to(gs_tpose_dic["sh"])
                else:
                    anchor_points = self.uv_encoding.anchor_points
                    anchor_points = anchor_points.to(gs_tpose_dic["sh"])
                gs_tpose_dic["xyz"] = anchor_points
        return gs_tpose_dic

    def _compute_canonical_patch_coords(self, gs_tpose_dic: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz_points = (gs_tpose_dic["xyz"] + gs_tpose_dic["offset_xyz"]).squeeze(1)
        patch_coords, _ = self.uv_encoding.aggregate_patch_points(xyz_points)
        return patch_coords

    def _sample_projected_features(
        self,
        spatial_cond_tokens: torch.Tensor,
        spatial_cond_coords: Optional[torch.Tensor],
        patch_coords: torch.Tensor,
    ) -> torch.Tensor:
        feat_map = spatial_tokens_to_feature_map(spatial_cond_tokens, spatial_cond_coords)
        B, C, H, W = feat_map.shape
        grid_x = 2.0 * patch_coords[..., 0] / max(W - 1, 1) - 1.0
        grid_y = 2.0 * patch_coords[..., 1] / max(H - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            feat_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2).contiguous()

    def _get_pose_all(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Resolve full-pose data from batch."""
        pose_all = batch.get("pose_all")
        assert pose_all is not None, "batch['pose_all'] must be provided"
        return pose_all

    def _build_all_view_cameras(
        self,
        batch: Dict,
        mhr: Dict[str, torch.Tensor],
        image_height: int,
        image_width: int,
    ) -> List[List[GSCamera]]:
        B, V = batch["img"].shape[:2]
        cam_int = batch.get("crop_cam_int", batch["cam_int"])
        raw_cam_int = batch.get("cam_int")
        raw_sizes = batch.get("raw_sizes")
        all_cameras: List[List[GSCamera]] = []

        for v in range(V):
            # cam_int may be [B, V, 3, 3] (MV path) or [B, 3, 3] (single-view path).
            K_v = cam_int[:, v] if cam_int.ndim == 4 else cam_int
            rK_v = raw_cam_int[:, v] if (raw_cam_int is not None and raw_cam_int.ndim == 4) else raw_cam_int

            cameras_v: List[GSCamera] = []
            for i in range(B):
                world2cam = torch.eye(4, device=cam_int.device, dtype=cam_int.dtype)
                world2cam[:3, :3] = mhr["cam_rot"][i, v]
                cam_t = mhr["pred_cam_t"][i, v]
                if "cam_trans" in mhr:
                    cam_t = cam_t + mhr["cam_trans"][i, v]
                world2cam[:3, 3] = cam_t

                full_h, full_w = None, None
                if raw_sizes is not None:
                    full_h = int(raw_sizes[i, v, 0].item())
                    full_w = int(raw_sizes[i, v, 1].item())

                cameras_v.append(
                    GSCamera.from_extrinsics_intrinsics(
                        world2cam,
                        K=K_v[i],
                        full_K=rK_v[i] if rK_v is not None else None,
                        full_image_width=full_w,
                        full_image_height=full_h,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )
            all_cameras.append(cameras_v)

        return all_cameras

    def _build_all_view_feature_cameras(
        self,
        cameras_per_view: List[List[GSCamera]],
        feat_h: int,
        feat_w: int,
    ) -> List[List[GSCamera]]:
        return [
            self._build_feature_cameras(cameras_v, feat_h=feat_h, feat_w=feat_w)
            for cameras_v in cameras_per_view
        ]

    def _compute_all_view_fac(
        self,
        canonical_gs: Dict[str, torch.Tensor],
        lbs_cache,
        cameras_per_view: List[List[GSCamera]],
        feat_h: int,
        feat_w: int,
        coord_h: int,
        coord_w: int,
        depth_scale: float = 1.0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        patch_fac_list = []
        patch_posed_xyz_list = []
        render_result_list = []

        with torch.autocast("cuda", enabled=False):
            for v_idx in range(len(cameras_per_view)):
                gs_posed_v, _ = self.mhr_animatable_gs.animate_from_cache_barycentric(
                    lbs_cache,
                    canonical_gs,
                    self.mhr_animatable_gs._flat_face_ids,
                    self.mhr_animatable_gs._flat_bary_coords,
                    pose_indices=[v_idx],
                    to_colmap=True,
                )
                patch_fac, patch_posed_xyz, render_result = compute_fac(
                    posed_gs=gs_posed_v,
                    cameras=cameras_per_view[v_idx],
                    feat_h=feat_h,
                    feat_w=feat_w,
                    coord_h=coord_h,
                    coord_w=coord_w,
                    depth_scale=depth_scale,
                    uv_encoding=self.uv_encoding,
                )
                render_result["cameras"] = cameras_per_view[v_idx]
                patch_fac_list.append(patch_fac)
                patch_posed_xyz_list.append(patch_posed_xyz)
                render_result_list.append(render_result)

        return patch_fac_list, patch_posed_xyz_list, render_result_list

    def _extract_pred_camera_dict(
        self,
        mhr: Dict[str, torch.Tensor],
        view_index: int,
    ) -> Dict[str, torch.Tensor]:
        pred_camera = {
            "pred_cam_t": mhr["pred_cam_t"][:, view_index],
        }
        if "cam_rot" in mhr:
            pred_camera["cam_rot"] = mhr["cam_rot"][:, view_index]
        if "cam_trans" in mhr:
            pred_camera["cam_trans"] = mhr["cam_trans"][:, view_index]
        return pred_camera

    def get_ray_condition(self, batch):
        if batch["cam_int"].ndim == 4:
            B, V, _, H, W = batch["img"].shape
            meshgrid_xy = (
                torch.stack(
                    torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"),
                    dim=2,
                ).permute(1, 0, 2)[None, None, :, :, :].repeat(B, V, 1, 1, 1).to(batch["img"].device)
            )
            meshgrid_xy = meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
            meshgrid_xy = (
                meshgrid_xy
                - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
                / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
            )
            meshgrid_xy = meshgrid_xy - batch["cam_int"][:, :, None, None, [0, 1], [2, 2]]
            meshgrid_xy = meshgrid_xy / batch["cam_int"][:, :, None, None, [0, 1], [0, 1]]
            return meshgrid_xy.permute(0, 1, 4, 2, 3).to(batch["img"].dtype)
        return super().get_ray_condition(batch)

    def _forward_pose_estimation_multi_view(
        self,
        batch: Dict,
        thresh_wrist_angle: float = 1.4,
    ) -> Dict:
        """Forward pose estimation branch for multi-view batch."""
        V = batch["images"].shape[1]
        device = batch["images"].device
        input_pose_dic = batch.get("input_pose_dic", None)
        have_full_pose = (
            input_pose_dic is not None
            and "has_pose" in input_pose_dic
            and bool(input_pose_dic["has_pose"].all())
        )
        if have_full_pose:
            mhr = {k: v for k, v in input_pose_dic.items()}
        else:
            per_view_mhr = []
            for v_idx in range(V):
                if "K_proc" in batch:
                    single_batch = create_single_view_batch(batch, v_idx, device)
                else:
                    if V != 1:
                        raise KeyError("K_proc")
                    single_batch = batch
                self._initialize_batch(single_batch)
                pose_out_v = self.forward_pose_branch_full(
                    single_batch,
                    get_intermediate_features=False,
                    enable_hand_refinement=True,
                    thresh_wrist_angle=thresh_wrist_angle,
                )
                per_view_mhr.append(pose_out_v["mhr"])

            mhr = {}
            for key, value in per_view_mhr[0].items():
                if isinstance(value, torch.Tensor):
                    mhr[key] = torch.stack([pose_v[key] for pose_v in per_view_mhr], dim=1)
                else:
                    mhr[key] = value
            self._initialize_batch(batch)

        self._ensure_factorized_input_pose_dic(mhr)
        return {"mhr": mhr}

    def _initialize_batch(self, batch: Dict) -> None:
        if "img" not in batch and "images" in batch:
            batch["img"] = batch["images"]
        if "images" not in batch and "img" in batch:
            batch["images"] = batch["img"]
        if "mask" not in batch and "masks" in batch:
            batch["mask"] = batch["masks"]
        if "masks" not in batch and "mask" in batch:
            batch["masks"] = batch["mask"]
        if "person_valid" not in batch:
            batch["person_valid"] = torch.ones(
                batch["img"].shape[:2],
                device=batch["img"].device,
                dtype=torch.bool,
            )
        if "cam_int" not in batch and "K_raw" in batch:
            batch["cam_int"] = batch["K_raw"]
        if "crop_cam_int" not in batch and "K_proc" in batch:
            batch["crop_cam_int"] = batch["K_proc"]
        super()._initialize_batch(batch)

    def _get_full_lbs_cache(self, batch, B, include_uv_cache=False):
        """Resolve full precomputed LBS cache from batch, or rebuild from pose_all.

        If rebuilding, assumes pose_all["model_params"] has shape [B*V_total, dim] ordered as
        (v0,b0), (v0,b1), ..., (v0,bB-1), (v1,b0), ... from prepare_pose_lbs.
        Returns lbs_cache with N=V_total so pose_indices=[v_idx] is always valid.
        """
        lbs_cache_all = batch.get("lbs_cache_all")
        if lbs_cache_all is not None:
            return lbs_cache_all

        pose_all = self._get_pose_all(batch)
        model_params = pose_all["model_params"]
        shape_params = pose_all["shape_params"]
        expr_params = pose_all["expr_params"]
        BV = model_params.shape[0]
        if BV % B != 0:
            return None
        V_total = BV // B
        pose_dics = []
        for v_idx in range(V_total):
            s, e = v_idx * B, (v_idx + 1) * B
            pose_dics.append({
                "shape_params": shape_params[s:e],
                "model_params": model_params[s:e].unsqueeze(1),
                "expr_params": expr_params[s:e].unsqueeze(1),
            })
        return self.mhr_animatable_gs.precompute_lbs_cache(pose_dics, include_uv_cache=include_uv_cache)

    def _run_avatar_conditioner(self, image_embeddings, batch: Dict, pose_output: Dict):
        """Conditioner call site inside _forward_prologue; subclasses override to
        pass extra inputs (e.g. projection-fetch HR feature pyramid)."""
        return self.avatar_conditioner(image_embeddings, batch, pose_output)

    def _forward_pose_output(self, batch: Dict, thresh_wrist_angle: float = 1.4) -> Dict:
        """Pose-output resolution inside _forward_prologue; PixelDiT overrides to
        reuse precomputed batch['pose_all'] poses."""
        return self._forward_pose_estimation_multi_view(
            batch,
            thresh_wrist_angle=thresh_wrist_angle,
        )

    def _forward_prologue(
        self,
        batch: Dict,
        gs_tpose_dic: Optional[dict] = None,
        thresh_wrist_angle: float = 1.4,
    ) -> Dict[str, Any]:
        """Shared forward prologue for all MV-UV variants.

        Runs batch init, backbone, pose estimation, LBS-cache resolution/slicing,
        avatar conditioning, and anchor GS preparation — in exactly this order —
        and returns the locals as a context dict. Subclass forwards must not
        repeat any of these steps.
        """
        self._initialize_batch(batch)
        B, V, _, H, W = batch["images"].shape

        with torch.no_grad(), torch.autocast("cuda", dtype=self.backbone_dtype):
            backbone_output = self.forward_backbone(batch)
        image_embeddings = backbone_output["image_embeddings"] # (BV, C, H, W)

        # 2. Full pose estimation with hand refinement (or use pre-computed poses)
        pose_output = self._forward_pose_output(batch, thresh_wrist_angle=thresh_wrist_angle)
        mhr = pose_output["mhr"]

        full_lbs_cache = self._get_full_lbs_cache(batch, B, include_uv_cache=True)
        input_view_indices = batch.get('input_view_indices', list(range(V)))
        n_poses = full_lbs_cache["rigid_quat"].shape[1]
        lbs_cache = {
            k: v[:, input_view_indices] if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[0] == B and v.shape[1] == n_poses else v
            for k, v in full_lbs_cache.items()
        }

        cond_out = self._run_avatar_conditioner(image_embeddings, batch, pose_output)
        C = cond_out.spatial_cond_tokens.shape[-1]
        spatial_cond_tokens = cond_out.spatial_cond_tokens.view(B, V, -1, C)
        spatial_cond_coords = cond_out.spatial_cond_coords
        feat_h, feat_w = image_embeddings.shape[-2:]

        gs_tpose_dic = self._prepare_gs_tpose(batch_size=B, gs_tpose_dic=gs_tpose_dic)

        return {
            "B": B, "V": V, "H": H, "W": W,
            "backbone_output": backbone_output,
            "image_embeddings": image_embeddings,
            "pose_output": pose_output,
            "mhr": mhr,
            "full_lbs_cache": full_lbs_cache,
            "input_view_indices": input_view_indices,
            "n_poses": n_poses,
            "lbs_cache": lbs_cache,
            "spatial_cond_tokens": spatial_cond_tokens,
            "spatial_cond_coords": spatial_cond_coords,
            "feat_h": feat_h, "feat_w": feat_w,
            "gs_tpose_dic": gs_tpose_dic,
        }

    def forward(
        self,
        batch: Dict,
        gs_tpose_dic: Optional[dict] = None,
        input_pose_dic: Optional[Dict] = None,
        thresh_wrist_angle: float = 1.4,
        debug_save_folder: str = "",
    ) -> Dict:
        ctx = self._forward_prologue(batch, gs_tpose_dic, thresh_wrist_angle)
        B, V, H, W = ctx["B"], ctx["V"], ctx["H"], ctx["W"]
        backbone_output = ctx["backbone_output"]
        mhr = ctx["mhr"]
        full_lbs_cache = ctx["full_lbs_cache"]
        input_view_indices = ctx["input_view_indices"]
        lbs_cache = ctx["lbs_cache"]
        spatial_cond_tokens = ctx["spatial_cond_tokens"]
        spatial_cond_coords = ctx["spatial_cond_coords"]
        feat_h, feat_w = ctx["feat_h"], ctx["feat_w"]
        gs_tpose_dic = ctx["gs_tpose_dic"]

        # --- posed token additive ---
        pose_all = self._get_pose_all(batch)
        n_poses_embed = pose_all["model_params"].shape[0] // B
        pose_embed_all = self.pose_proj(pose_all["model_params"]).view(B, n_poses_embed, -1)

        # When pose_dep is enabled, extract LBS conditioning for ALL target poses so
        # the pose_dep block can reuse posed_token_additive_all without re-extracting.
        _n_posed = n_poses_embed if self.pose_dep_enabled else V
        _lbs_for_posed = full_lbs_cache if self.pose_dep_enabled else lbs_cache
        lbs_feat_all = extract_lbs_feat(
            self.mhr_animatable_gs,
            _lbs_for_posed,
            self.mhr_animatable_gs._flat_face_ids,
            self.mhr_animatable_gs._flat_bary_coords,
            pose_indices=list(range(_n_posed)),
        )
        _posed_enc = self.uv_encoding.encode(
            lbs_feat_all.view(B * _n_posed, *lbs_feat_all.shape[2:]),
            self.posed_patch_embed,
        )
        # reshape needed: _posed_enc is non-contiguous (encode returns flatten+permute output)
        # and its channel dim differs from C (spatial_cond backbone dim vs UV embed_dim)
        _posed_enc = _posed_enc.reshape(B, _n_posed, -1, _posed_enc.shape[-1])

        if self.pose_dep_enabled:
            # Bake pose_embed into all-view additive, then slice for main transformer.
            posed_token_additive_all = _posed_enc + pose_embed_all.unsqueeze(2)
            posed_token_additive = posed_token_additive_all[:, input_view_indices]
        else:
            posed_token_additive_all = None
            posed_token_additive = _posed_enc + pose_embed_all[:, input_view_indices].unsqueeze(2)

        # --- coord computation ---
        cameras_all = None
        patch_fac = None        # List[Tensor [B,L,3]], length V  — feature-aligned coords
        patch_posed_xyz = None  # List[Tensor [B,L,3]], length V  — raw posed xyz

        if self.needs_render:
            with torch.no_grad():
                cameras_all = self._build_all_view_cameras(batch, mhr, H, W)
                if self.needs_fac or self.needs_sa3d_coords:
                    patch_fac, patch_posed_xyz, render_result_list = self._compute_all_view_fac(
                        canonical_gs=gs_tpose_dic,
                        lbs_cache=lbs_cache,
                        cameras_per_view=cameras_all,
                        feat_h=feat_h,
                        feat_w=feat_w,
                        coord_h=H,
                        coord_w=W,
                    )

                fac_stacked = torch.stack(patch_fac, dim=1) if patch_fac is not None else None
                posed_stacked = torch.stack(patch_posed_xyz, dim=1) if patch_posed_xyz is not None else None

        canonical_patch_coords = self._compute_canonical_patch_coords(gs_tpose_dic)
        canonical_patch_coords_mv = canonical_patch_coords[:, None, :, :].expand(-1, V, -1, -1)

        avatar_coords_3d_cross = fac_stacked if self.needs_fac else None
        avatar_coords_3d_self = None
        if self.sa3d_type == "canonical-rope":
            avatar_coords_3d_self = canonical_patch_coords_mv
        elif self.sa3d_type == "animation-rope":
            avatar_coords_3d_self = posed_stacked

        avatar_tokens = gs_to_avatar_tokens(
            gs_tpose_dic,
            uv_encoding=self.uv_encoding,
            token_cond=self.canonical_token_cond,
            patch_embed=self.canonical_patch_embed,
            lbs_feat=None,
        )

        for block in self.rope3d_blocks:
            avatar_tokens = block(
                avatar_tokens=avatar_tokens,
                posed_token_additive=posed_token_additive,
                spatial_cond_tokens=spatial_cond_tokens,
                token_coords=avatar_coords_3d_cross,
                spatial_cond_coords=spatial_cond_coords,
                patch_uv_coords=self.uv_encoding.patch_positions,
                multiview_fusion=self.multiview_fusion,
                sa3d_coords=avatar_coords_3d_self,
            )

        # 3. Decode to GS params
        dense_out = self._decode_dense_outputs(
            avatar_tokens,
            gs_tpose_dic_anchor=gs_tpose_dic,
        )
        new_gs_tpose = dense_out["output_canonical_gs_dic"]

        pred_cameras = [self._extract_pred_camera_dict(mhr, v_idx) for v_idx in range(V)]

        # Run pose-dep inside DDP forward for proper gradient sync
        deform_dic = {}
        if self.pose_dep_enabled:
            _, L, D = avatar_tokens.shape
            # posed_token_additive_all covers all n_poses_embed views (computed above).
            tokens_all = avatar_tokens.unsqueeze(1).expand(-1, n_poses_embed, -1, -1).reshape(B * n_poses_embed, L, D)
            posed_additive_bv = posed_token_additive_all.reshape(B * n_poses_embed, L, -1)
            posed_tokens = self.pose_dep_mlp(tokens_all + posed_additive_bv)
            deform_dic = self.compute_target_deform_dic(posed_tokens)

            if self.deform_shape and "offset_pose_scale_" in deform_dic:
                # exp converts log-space offset to a multiplicative scale factor
                new_gs_tpose["offset_pose_scale_"] = torch.tanh(deform_dic["offset_pose_scale_"]) + 1
                # safe-normalize: zero-padded checkpoints produce near-zero quaternion → skip
                new_gs_tpose["offset_pose_rotation"] = F.normalize(deform_dic["offset_pose_rotation"], dim=-1)

        pass_output = {
            "input_canonical_gs_dic": gs_tpose_dic,
            "output_canonical_gs_dic": new_gs_tpose,
            "pred_camera": pred_cameras,
            "mhr": mhr,
            "avatar_tokens": avatar_tokens,
            "lbs_cache": lbs_cache,
            "pixel_gs_residual": dense_out["pixel_gs_residual"],
        }
        pass_output.update(deform_dic)

        return {
            "passes": [pass_output],
            "render_input": [],
            "backbone_output": backbone_output,
            "lbs_cache": full_lbs_cache,
        }
