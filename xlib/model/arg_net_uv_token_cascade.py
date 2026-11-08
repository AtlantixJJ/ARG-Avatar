from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from xlib.render.camera import GSCamera
from xlib.model.mhr_animatable_gaussians import extract_lbs_feat
from xlib.model.arg_net_uv import ARGNetUV
import os
import numpy as np
from PIL import Image
import torch.nn as nn
from diffusers.utils.export_utils import export_to_video
from xlib.render.gaussian_splatting import GaussianParameters
from xlib.render.gaussian_splatting_3d import index_gs
from xlib.render.visualizer import centerize_g, render_rotate
from xlib.model.gs_layer import GSLayer
from xlib.basetype import TensorDict
from xlib.model.rope_3d import debug_rope_projection, compute_fac
from xlib.geometry.uv_encoding import gs_to_avatar_tokens


def animate_token_from_cache(
    animatable_gs,
    lbs_cache: Dict[str, torch.Tensor],
    token_gs_dic: Dict[str, torch.Tensor],
    gs_tpose_dic_anchor: Dict[str, torch.Tensor],
    uv_encoding: Any,
    pose_indices: List[int] = None,
    to_colmap: bool = True,
    offset_pose: Optional[torch.Tensor] = None,
) -> Tuple[GaussianParameters, torch.Tensor]:
    """Animate token GS via UV-sample broadcast, UV-cache animation, and patch pooling."""
    uv_anchor_points = gs_tpose_dic_anchor["xyz"].to(
        device=token_gs_dic["offset_xyz"].device,
        dtype=token_gs_dic["offset_xyz"].dtype,
    )
    B = token_gs_dic["offset_xyz"].shape[0]
    D_sh = token_gs_dic["sh"].shape[-2]  # (sh_degree+1)^2

    bcast_fields = [
        token_gs_dic["offset_xyz"].squeeze(1),
        token_gs_dic["scale"].squeeze(1),
        token_gs_dic["rotation"].squeeze(1),
        token_gs_dic["opacity"].squeeze(1),
        token_gs_dic["sh"].squeeze(1).flatten(-2),  # [..., D, 3] -> [..., D*3]
    ]
    bcast_sizes = [field.shape[-1] for field in bcast_fields]
    bcast_out = uv_encoding.broadcast_token_feat_to_samples(
        torch.cat(bcast_fields, dim=-1)
    ).split(bcast_sizes, dim=-1)

    uv_sample_gs = TensorDict.from_dict({
        "xyz": uv_anchor_points.expand(B, -1, -1, -1),
        "offset_xyz": bcast_out[0].unsqueeze(1),
        "scale": bcast_out[1].unsqueeze(1),
        "rotation": bcast_out[2].unsqueeze(1),
        "opacity": bcast_out[3].unsqueeze(1),
        "sh": bcast_out[4].unflatten(-1, (D_sh, 3)).unsqueeze(1),  # [..., D*3] -> [B, 1, L, D, 3]
    })

    if offset_pose is not None and offset_pose.shape[2] != uv_sample_gs["offset_xyz"].shape[2]:
        # Broadcast token-level pose offset to UV-sample-level
        offset_pose = uv_encoding.broadcast_token_feat_to_samples(offset_pose.squeeze(1)).unsqueeze(1)

    uv_sample_gs_posed, posed_mesh_verts = animatable_gs.animate_from_cache_barycentric(
        lbs_cache,
        uv_sample_gs,
        animatable_gs._flat_face_ids,
        animatable_gs._flat_bary_coords,
        pose_indices=pose_indices,
        to_colmap=to_colmap,
        offset_pose=offset_pose,
    )

    _, N, L, _ = uv_sample_gs_posed.xyz.shape
    agg_inputs = [
        uv_sample_gs_posed.xyz,
        uv_sample_gs_posed.scale,
        uv_sample_gs_posed.rotation,
        uv_sample_gs_posed.opacity,
        uv_sample_gs_posed.sh.flatten(-2),  # [B, N, L, D, 3] -> [B, N, L, D*3]
    ]
    agg_sizes = [feat.shape[-1] for feat in agg_inputs]
    agg_feat = torch.cat(agg_inputs, dim=-1).reshape(B * N, L, -1)
    agg_patch_feat, _ = uv_encoding.aggregate_patch_points(agg_feat)
    agg_split = agg_patch_feat.view(B, N, agg_patch_feat.shape[1], agg_patch_feat.shape[2]).split(agg_sizes, dim=-1)

    token_gs_posed = GaussianParameters(
        xyz=agg_split[0],
        scale=agg_split[1],
        rotation=F.normalize(agg_split[2], dim=-1),
        opacity=agg_split[3],
        sh=agg_split[4].unflatten(-1, (D_sh, 3)),  # [B, N, P, D*3] -> [B, N, P, D, 3]
        active_sh_degree=uv_sample_gs_posed.active_sh_degree,
    )
    token_gs_posed.rotation = token_gs_posed.rotation * token_gs_posed.rotation[..., :1].sign()
    return token_gs_posed, posed_mesh_verts



class ARGNetUVTokenCascade(ARGNetUV):
    """Multi-view token-cascade UV avatar.

    Combines ARGNetUV's cross-view fusion with a token-level cascade:
    after every block we predict token-level offsets, animate them per-view,
    and recompute per-view feature-aligned coordinates (fac) for the next
    block's cross-attention — using cheap token GS instead of the dense
    pixel GS.
    """

    def _initialize_avatar(self, avatar_cfg: dict) -> None:
        # Cascade knobs are parsed before the parent builders run so that
        # builder overrides (e.g. _build_token_pose_offset_heads) can use them.
        self.return_intermediate_results = bool(avatar_cfg.get("return_intermediate_results", False))
        self.shared_token_head = bool(avatar_cfg["shared_token_head"])
        if avatar_cfg.get("use_token_gs", True) is not True:
            raise ValueError("ARGNetUVTokenCascade requires use_token_gs=True.")
        self.use_token_gs = True
        # Differentiable-FAC: intermediate FAC supervised by the final loss.
        # Optional block; absent means the historical detached-FAC behavior.
        diff_fac_cfg = avatar_cfg.get("differentiable_fac") or {}
        self.diff_fac_enabled = bool(diff_fac_cfg.get("enabled", False))
        self.diff_fac_z_grad = bool(diff_fac_cfg.get("z_grad", False))
        self.diff_fac_depth_grad = bool(diff_fac_cfg.get("depth_grad", False))
        self.diff_fac_z_clamp_min = float(diff_fac_cfg.get("z_clamp_min", 0.05))
        self.diff_fac_grad_clip = float(diff_fac_cfg.get("fac_grad_clip", 1.0))
        # Alpha ramp measured in training forward calls (grad-accum included).
        self.diff_fac_warmup_forwards = int(diff_fac_cfg.get("fac_grad_warmup_forwards", 0))
        self._diff_fac_forward_count = 0
        self.freeze_unsupervised_token_heads = bool(
            avatar_cfg.get("freeze_unsupervised_token_heads", False))
        super()._initialize_avatar(avatar_cfg)
        self._build_token_heads(avatar_cfg)

    def setup_avatar_training(self):
        super().setup_avatar_training()
        if self.freeze_unsupervised_token_heads:
            # Without an intermediate token render loss (and without diff-FAC
            # keeping them in the backward graph), the token-head sh/rotation
            # rows never receive gradients; DDP reduction then hangs on them
            # because token_canonical_gs_dic keeps them in the output graph.
            for head in self.token_gsheads:
                for key in ("sh", "rotation"):
                    if key in head.out_layers:
                        for p in head.out_layers[key].parameters():
                            p.requires_grad = False

    def _build_token_pose_offset_heads(self) -> None:
        """Per-block token pose-offset heads instead of the parent's single proj."""
        num_token_pose_offset_projs = 1 if self.shared_token_head else len(self.rope3d_blocks)
        self.token_pose_offset_projs = nn.ModuleList([
            self._build_token_pose_offset_proj() for _ in range(num_token_pose_offset_projs)
        ])
        self.trainable_modules.append("token_pose_offset_projs")

    def _build_token_heads(self, avatar_cfg: dict) -> None:
        """Per-block token GS heads."""
        num_blocks = len(self.rope3d_blocks)
        num_token_heads = 1 if self.shared_token_head else num_blocks
        gs_layer_cfg = dict(avatar_cfg["gs_layer"])
        gs_layer_cfg["in_channels"] = self.embed_dim
        if gs_layer_cfg.get("n_layers", 0) < 1:
            gs_layer_cfg["n_layers"] = 1
        self.token_gsheads = nn.ModuleList([
            GSLayer(**gs_layer_cfg) for _ in range(num_token_heads)
        ])
        for head in self.token_gsheads:
            head.set_scale_base(self.gshead.scale_base.clone() * self.uv_encoding.patch_size * 0.8)
        self.trainable_modules.append("token_gsheads")

    @torch.no_grad()
    def _write_debug_mesh_gs_video(
        self,
        mesh_gs_posed: GaussianParameters,
        folder: str,
        prefix: str,
        resolution: int = 512,
    ) -> str:
        video_dir = folder
        video_path = os.path.join(video_dir, f"{prefix}_rotate.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        gs_view = mesh_gs_posed
        if gs_view.xyz.dim() == 4:
            gs_view = index_gs(gs_view, 0, 0)
        elif gs_view.xyz.dim() == 3:
            gs_view = GaussianParameters(
                xyz=gs_view.xyz[0],
                scale=gs_view.scale[0],
                rotation=gs_view.rotation[0],
                opacity=gs_view.opacity[0],
                sh=gs_view.sh[0] if gs_view.sh is not None else None,
                offset_xyz=gs_view.offset_xyz[0] if gs_view.get("offset_xyz") is not None else None,
                active_sh_degree=gs_view.active_sh_degree,
            )
        gs_view = centerize_g(gs_view)
        with torch.autocast("cuda", enabled=False):
            gs_view = gs_view.to(device=self.device, dtype=torch.float32)
            frames, _ = render_rotate(
                gs_view,
                bg_color=torch.ones(3, device=gs_view.xyz.device, dtype=torch.float32),
                fov=1.0,
                radius=1.8,
                resolution=resolution,
            )
        frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        export_to_video([Image.fromarray(frame) for frame in frames_np], video_path, fps=15)
        print(f"[DEBUG] token GS rotation video saved to {video_path}")
        return video_path

    def _predict_token_gs_for_block(self, avatar_tokens, block_idx):
        head = self.token_gsheads[0 if self.shared_token_head else block_idx]
        with torch.autocast("cuda", enabled=False):
            token_gs_raw_linear = head.forward_raw(avatar_tokens.unsqueeze(1))
        token_gs_raw = head.presum(token_gs_raw_linear)
        token_gs = head.activate(TensorDict.from_dict(dict(token_gs_raw)))
        return token_gs, token_gs_raw

    def _build_canonical_gs_from_token_gs(
        self,
        token_gs: TensorDict,
        gs_tpose_dic: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Broadcast token GS attributes to UV sample space for canonical rendering."""
        bcast_fields = [
            token_gs["offset_xyz"].squeeze(1),        # [B, N, 3]
            token_gs["scale"].squeeze(1),              # [B, N, S]
            token_gs["rotation"].squeeze(1),           # [B, N, 4]
            token_gs["opacity"].squeeze(1),            # [B, N, 1]
            token_gs["sh"].squeeze(1).squeeze(-2),     # [B, N, 3]
        ]
        bcast_sizes = [f.shape[-1] for f in bcast_fields]
        bcast_out = self.uv_encoding.broadcast_token_feat_to_samples(
            torch.cat(bcast_fields, dim=-1)
        ).split(bcast_sizes, dim=-1)
        return {
            "xyz":        gs_tpose_dic["xyz"],
            "offset_xyz": bcast_out[0].unsqueeze(1),
            "scale":      bcast_out[1].unsqueeze(1),
            "rotation":   bcast_out[2].unsqueeze(1),
            "opacity":    bcast_out[3].unsqueeze(1),
            "sh":         bcast_out[4].unsqueeze(1).unsqueeze(-2),
        }

    def _predict_token_deform(self, posed_avatar_token: torch.Tensor) -> torch.Tensor:
        # Use the last proj (final cascade block) for non-block-specific calls
        proj = self.token_pose_offset_projs[-1]
        token_offsets = proj(posed_avatar_token)
        if self.token_pose_offset_max_step is not None:
            token_offsets = torch.tanh(token_offsets) * float(self.token_pose_offset_max_step)
        return token_offsets

    def _predict_token_deform_for_block(
        self,
        posed_avatar_token: torch.Tensor,
        block_idx: int,
    ) -> torch.Tensor:
        token_pose_offset_proj = self.token_pose_offset_projs[0 if self.shared_token_head else block_idx]
        token_offsets = token_pose_offset_proj(posed_avatar_token)
        if self.token_pose_offset_max_step is not None:
            token_offsets = torch.tanh(token_offsets) * float(self.token_pose_offset_max_step)
        return token_offsets

    def _decode_dense_outputs(
        self,
        avatar_tokens: torch.Tensor,
        gs_tpose_dic_anchor: Dict[str, torch.Tensor],
        token_offset: torch.Tensor,
    ) -> Dict[str, Any]:
        """Decode dense canonical GS and add mesh-smoothed token offsets to `offset_xyz`."""
        pixel_feats, _ = self.uv_encoding.decode(avatar_tokens, decoder=self.uv_decoder)
        pixel_gs_raw = self.gshead.forward_raw(pixel_feats.unsqueeze(1))
        pixel_gs_delta = self.gshead.presum(pixel_gs_raw)

        # 1. Base dense decode (residual on top of canonical anchor)
        final_gs_raw = self.gshead.sum_residuals([gs_tpose_dic_anchor, pixel_gs_delta])
        final_gs = self.gshead.activate(final_gs_raw)
        final_gs["xyz"] = gs_tpose_dic_anchor["xyz"]
        pixel_offset_xyz_residual = final_gs["offset_xyz"].clone()

        # 2. Geometry: smooth token offsets to dense pixels via mesh
        vertex_offset = self.uv_encoding.token_feature_to_vertices(token_offset)
        vertex_offset_pixel = self.uv_encoding.interpolate_vertex_deform_to_pixels(vertex_offset)
        final_gs["offset_xyz"] = pixel_offset_xyz_residual + vertex_offset_pixel

        return {
            "output_canonical_gs_dic": final_gs,
            "pixel_offset": pixel_offset_xyz_residual,
            "token_offset": token_offset,
            "vertex_offset": vertex_offset,
        }

    def compute_deform_dic(
        self,
        posed_avatar_token: torch.Tensor,
        token_pose_offset: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Decode pose-dependent deformation from precomputed posed tokens.

        Args:
            posed_avatar_token: [B_poses, N_uv, D] — output of pose_dep_mlp for all target poses.
            token_pose_offset:  [B_poses, N_uv, 3|10] or [1, N_uv, 3|10] — last block's token
                pose offset(s) aligned with `posed_avatar_token` (or None).
        """
        raw_pixel, _ = self.uv_encoding.decode(posed_avatar_token, decoder=self.posedep_uv_decoder)
        # [B, L, 3] or [B, L, 10]

        pixel_offset_pose = raw_pixel[..., :3] * self.pixel_pose_offset_max_step

        if self.token_pose_offset_enabled and token_pose_offset is not None:
            vertex_pose_offsets = self.uv_encoding.token_feature_to_vertices(token_pose_offset[..., :3])
            token_pose_offset_pixels = self.uv_encoding.interpolate_vertex_deform_to_pixels(vertex_pose_offsets)
            offset_pose = pixel_offset_pose + token_pose_offset_pixels

            deform_dic = {
                "offset_pose": offset_pose,
                "pixel_offset_pose": pixel_offset_pose,
                "token_offset_pose": token_pose_offset,
                "vertex_offset_pose": vertex_pose_offsets,
            }
            if self.deform_shape:
                token_on_pixels = self.uv_encoding.broadcast_token_feat_to_samples(token_pose_offset)
                deform_dic["offset_pose_scale_"] = raw_pixel[..., 3:6] + token_on_pixels[..., 3:6]
                deform_dic["offset_pose_rotation"] = raw_pixel[..., 6:10] + token_on_pixels[..., 6:10]
            return deform_dic

        deform_dic = {"offset_pose": pixel_offset_pose}
        if self.deform_shape:
            deform_dic["offset_pose_scale_"] = raw_pixel[..., 3:6]
            deform_dic["offset_pose_rotation"] = raw_pixel[..., 6:10]
        return deform_dic

    def _diff_fac_kwargs(self) -> Dict[str, Any]:
        """compute_fac gradient-routing kwargs for the per-block FAC recompute."""
        return {
            "differentiable": self.diff_fac_enabled,
            "z_grad": self.diff_fac_z_grad,
            "depth_grad": self.diff_fac_depth_grad,
            "z_clamp_min": self.diff_fac_z_clamp_min,
        }

    def _apply_diff_fac_grad_policy(self, patch_fac_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Soft alpha phase-in and gradient clamp on differentiable FAC coords.

        Forward values are exactly unchanged for any alpha; only the gradient
        magnitude ramps 0->1 over `fac_grad_warmup_forwards` training forwards.
        """
        if not (self.diff_fac_enabled and self.training and torch.is_grad_enabled()):
            return patch_fac_list
        if self.diff_fac_warmup_forwards > 0:
            alpha = min(1.0, self._diff_fac_forward_count / float(self.diff_fac_warmup_forwards))
        else:
            alpha = 1.0
        clip = self.diff_fac_grad_clip
        out = []
        for fac in patch_fac_list:
            if fac.requires_grad:
                if clip > 0:
                    fac.register_hook(lambda g: g.nan_to_num(0.0).clamp(-clip, clip))
                if alpha < 1.0:
                    fac = fac.detach() + alpha * (fac - fac.detach())
            out.append(fac)
        return out

    def _compute_token_fac_all_views(
        self,
        token_gs: TensorDict,
        token_pose_offset: Optional[torch.Tensor],
        lbs_cache,
        cameras_per_view: List[List[GSCamera]],
        feat_h: int,
        feat_w: int,
        coord_h: int,
        coord_w: int,
        gs_tpose_dic: Optional[Dict[str, torch.Tensor]] = None,
        depth_scale: float = 1.0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, Any]]]:
        """Animate token GS per-view and compute per-view feature-aligned coordinates."""
        patch_fac_list = []
        patch_posed_xyz_list = []
        render_result_list = []
        with torch.autocast("cuda", enabled=False):
            for v_idx in range(len(cameras_per_view)):
                pose_offset_v = token_pose_offset[:, v_idx] if token_pose_offset is not None else None
                token_gs_posed, _ = animate_token_from_cache(
                    self.mhr_animatable_gs, lbs_cache, token_gs, gs_tpose_dic,
                    self.uv_encoding, pose_indices=[v_idx], to_colmap=True,
                    offset_pose=pose_offset_v[..., :3].unsqueeze(1) if pose_offset_v is not None else None,
                )
                patch_fac, patch_posed_xyz, render_result = compute_fac(
                    posed_gs=token_gs_posed, cameras=cameras_per_view[v_idx],
                    feat_h=feat_h, feat_w=feat_w, coord_h=coord_h, coord_w=coord_w,
                    depth_scale=depth_scale, uv_encoding=None,
                    **self._diff_fac_kwargs(),
                )
                render_result["cameras"] = cameras_per_view[v_idx]
                render_result["mesh_gs_posed"] = token_gs_posed
                patch_fac_list.append(patch_fac)
                patch_posed_xyz_list.append(patch_posed_xyz)
                render_result_list.append(render_result)
        return patch_fac_list, patch_posed_xyz_list, render_result_list

    def _assemble_pass_output(
        self,
        dense_out: Dict[str, Any],
        deform_dic: Dict[str, Any],
        *,
        gs_tpose_dic: Dict[str, torch.Tensor],
        pred_cameras: List[Dict[str, torch.Tensor]],
        mhr: Dict[str, torch.Tensor],
        lbs_cache: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Assemble one entry of result["passes"] (shared by cascade variants)."""
        pass_output = {
            "input_canonical_gs_dic": gs_tpose_dic,
            "pred_camera": pred_cameras,
            "mhr": mhr,
            "lbs_cache": lbs_cache,
        }
        pass_output.update(dense_out)
        pass_output.update(deform_dic)

        if self.deform_shape and "offset_pose_scale_" in deform_dic:
            new_gs_tpose = pass_output["output_canonical_gs_dic"]
            # exp converts log-space offset to a multiplicative scale factor
            new_gs_tpose["offset_pose_scale_"] = torch.tanh(deform_dic["offset_pose_scale_"]) + 1
            # safe-normalize: zero-padded checkpoints produce near-zero quaternion → skip
            new_gs_tpose["offset_pose_rotation"] = F.normalize(deform_dic["offset_pose_rotation"], dim=-1)

        return pass_output

    def _collect_intermediate_passes(self) -> bool:
        """Whether _cascade_encode records per-block intermediate passes.
        PixelDiT overrides to False: its decode path ignores pass_outputs."""
        return self.return_intermediate_results

    def _cascade_encode(
        self,
        batch: Dict,
        ctx: Dict[str, Any],
        debug_save_folder: str = "",
    ) -> Dict[str, Any]:
        """Posed-token conditioning, initial FAC, and the cascade block loop.

        ``ctx`` is the dict returned by ``_forward_prologue``. The body is shared
        verbatim between TokenCascade and ProjectionFetch; it returns the loop
        state their decode epilogues consume.
        """
        collect_intermediate = self._collect_intermediate_passes()
        if self.diff_fac_enabled and self.training and torch.is_grad_enabled():
            self._diff_fac_forward_count += 1
        B, V, H, W = ctx["B"], ctx["V"], ctx["H"], ctx["W"]
        mhr = ctx["mhr"]
        full_lbs_cache = ctx["full_lbs_cache"]
        input_view_indices = ctx["input_view_indices"]
        n_poses = ctx["n_poses"]
        lbs_cache = ctx["lbs_cache"]
        spatial_cond_tokens = ctx["spatial_cond_tokens"]
        spatial_cond_coords = ctx["spatial_cond_coords"]
        feat_h, feat_w = ctx["feat_h"], ctx["feat_w"]
        gs_tpose_dic = ctx["gs_tpose_dic"]

        avatar_tokens = gs_to_avatar_tokens(
            gs_tpose_dic,
            uv_encoding=self.uv_encoding,
            token_cond=self.canonical_token_cond,
            patch_embed=self.canonical_patch_embed,
            lbs_feat=None,
        )
        _, L, D = avatar_tokens.shape

        compute_all_views = self.pose_dep_enabled
        _lbs_for_posed = full_lbs_cache if compute_all_views else lbs_cache
        _n_posed = n_poses if compute_all_views else V
        posed_token_additive_list = []
        for v_idx in range(_n_posed):
            lbs_feat_v = extract_lbs_feat(
                self.mhr_animatable_gs,
                _lbs_for_posed,
                self.mhr_animatable_gs._flat_face_ids,
                self.mhr_animatable_gs._flat_bary_coords,
                pose_indices=[v_idx],
            )[:, 0]
            posed_token_additive_list.append(
                gs_to_avatar_tokens(
                    gs_tpose_dic,
                    uv_encoding=self.uv_encoding,
                    token_cond=self.posed_token_cond,
                    patch_embed=self.posed_patch_embed,
                    lbs_feat=lbs_feat_v,
                )
            )
        if compute_all_views:
            posed_token_additive_all = torch.stack(posed_token_additive_list, dim=1)
            posed_token_additive = posed_token_additive_all[:, input_view_indices]
        else:
            posed_token_additive_all = None
            posed_token_additive = torch.stack(posed_token_additive_list, dim=1)

        cameras_all = self._build_all_view_cameras(batch, mhr, H, W)
        #feature_cameras_all = self._build_all_view_feature_cameras(
        #    cameras_all,
        #    feat_h=feat_h * 2, # anti-aliasing
        #    feat_w=feat_w * 2, # anti-aliasing
        #)
        feature_cameras_all = cameras_all
        patch_positions = self.uv_encoding.patch_positions
        with torch.no_grad():
            patch_fac, patch_posed_xyz, render_result_list = self._compute_all_view_fac(
                canonical_gs=gs_tpose_dic,
                lbs_cache=lbs_cache,
                cameras_per_view=cameras_all,
                feat_h=feat_h,
                feat_w=feat_w,
                coord_h=H,
                coord_w=W,
                depth_scale=3.0,
            )

        DEBUG_ROPE_PROJECTION = len(debug_save_folder) > 0
        if DEBUG_ROPE_PROJECTION:
            for v_idx in range(V):
                debug_rope_projection(
                    batch={"img": batch["images"][:, v_idx]},
                    render_result=render_result_list[v_idx],
                    fac_patch=patch_fac[v_idx],
                    posed_xyz=patch_posed_xyz[v_idx],
                    folder=debug_save_folder,
                    prefix=f"pass0_v{v_idx}",
                    posed_gs=render_result_list[v_idx].get("mesh_gs_posed"),
                )

        pred_cameras = [self._extract_pred_camera_dict(mhr, v_idx) for v_idx in range(V)]

        def _build_pass_output(dense_out, deform_dic={}):
            return self._assemble_pass_output(
                dense_out,
                deform_dic,
                gs_tpose_dic=gs_tpose_dic,
                pred_cameras=pred_cameras,
                mhr=mhr,
                lbs_cache=lbs_cache,
            )

        pass_outputs = []
        token_offset = None
        token_gs = None
        token_gs_pair = None
        intermediate_token_offsets = []
        token_pose_offset = None
        pose_embed_all = None
        last_posed_avatar_token = None
        if self.pose_dep_enabled:
            pose_all = self._get_pose_all(batch)
            n_poses_embed = pose_all["model_params"].shape[0] // B
            pose_embed_all = self.pose_proj(pose_all["model_params"]).view(B, n_poses_embed, -1)
            # Bake pose_embed into posed_token_additive for all views, then re-slice.
            posed_token_additive_all = posed_token_additive_all + pose_embed_all.unsqueeze(2)
            posed_token_additive = posed_token_additive_all[:, input_view_indices]
            if self.token_pose_offset_enabled:
                posedep_out_ch = 10 if self.deform_shape else 3
                token_pose_offset = avatar_tokens.new_zeros(B, V, L, posedep_out_ch)

        for i, block in enumerate(self.rope3d_blocks):
            avatar_tokens = block(
                avatar_tokens=avatar_tokens,
                posed_token_additive=posed_token_additive,
                spatial_cond_tokens=spatial_cond_tokens,
                token_coords=patch_fac,
                spatial_cond_coords=spatial_cond_coords,
                patch_uv_coords=patch_positions,
                multiview_fusion=self.multiview_fusion,
                sa3d_coords=patch_posed_xyz,
            )

            token_gs_pair = self._predict_token_gs_for_block(avatar_tokens, i)
            token_gs, token_gs_raw = token_gs_pair
            token_offset = token_gs["offset_xyz"].squeeze(1)  # [B, N, 3] for _decode_dense_outputs

            if i < len(self.rope3d_blocks) - 1:
                intermediate_token_offsets.append(token_offset)

            if self.pose_dep_enabled and self.token_pose_offset_enabled:
                # avatar_tokens is (B, L, D) -> (B*V, L, D)
                # token_pose_offset is (B, V, L, 3)
                tokens_bv = avatar_tokens.unsqueeze(1).expand(-1, V, -1, -1).view(B * V, avatar_tokens.shape[1], avatar_tokens.shape[2])
                posed_additive_bv = posed_token_additive.view(B * V, avatar_tokens.shape[1], avatar_tokens.shape[2])
                last_posed_avatar_token = self.pose_dep_mlp(tokens_bv + posed_additive_bv)
                posedep_out_ch = 10 if self.deform_shape else 3
                token_pose_offset = self._predict_token_deform_for_block(
                    last_posed_avatar_token, block_idx=i).view(B, V, L, posedep_out_ch)

            if collect_intermediate:
                pass_out_i = _build_pass_output({"output_canonical_gs_dic": dict(token_gs_pair[0])}, {})
                pass_out_i["token_canonical_gs_dic"] = token_gs_pair[0]
                pass_out_i["token_gs_delta"] = token_gs_pair[1]
                pass_out_i["is_token_gs"] = True
                pass_outputs.append(pass_out_i)

            if i < len(self.rope3d_blocks) - 1:
                patch_fac, patch_posed_xyz, render_result_list = self._compute_token_fac_all_views(
                    token_gs=token_gs,
                    token_pose_offset=token_pose_offset,
                    lbs_cache=lbs_cache,
                    cameras_per_view=feature_cameras_all,
                    feat_h=feat_h, feat_w=feat_w, coord_h=H, coord_w=W,
                    gs_tpose_dic=gs_tpose_dic, depth_scale=3.0,
                )
                patch_fac = self._apply_diff_fac_grad_policy(patch_fac)

                if DEBUG_ROPE_PROJECTION:
                    debug_token_can_gs = None
                    if token_gs["sh"].shape[-2] == 1:
                        debug_token_can_gs = self._build_canonical_gs_from_token_gs(token_gs, gs_tpose_dic)
                    for v_idx in range(V):
                        if "mesh_gs_posed" in render_result_list[v_idx]:
                            self._write_debug_mesh_gs_video(
                                render_result_list[v_idx]["mesh_gs_posed"],
                                folder=debug_save_folder,
                                prefix=f"pass{i+1}_v{v_idx}",
                            )
                        debug_rope_projection(
                            batch={"img": batch["images"][:, v_idx]},
                            render_result=render_result_list[v_idx],
                            fac_patch=patch_fac[v_idx],
                            posed_xyz=patch_posed_xyz[v_idx],
                            folder=debug_save_folder,
                            prefix=f"pass{i+1}_v{v_idx}",
                            posed_gs=render_result_list[v_idx].get("mesh_gs_posed"),
                            token_canonical_gs_dic=debug_token_can_gs if v_idx == 0 else None,
                        )

        return {
            "avatar_tokens": avatar_tokens,
            "L": L, "D": D,
            "token_offset": token_offset,
            "token_gs": token_gs,
            "token_gs_pair": token_gs_pair,
            "token_pose_offset": token_pose_offset,
            "last_posed_avatar_token": last_posed_avatar_token,
            "pose_embed_all": pose_embed_all,
            "posed_token_additive_all": posed_token_additive_all,
            "intermediate_token_offsets": intermediate_token_offsets,
            "pass_outputs": pass_outputs,
            "pred_cameras": pred_cameras,
            "feature_cameras_all": feature_cameras_all,
            "build_pass_output": _build_pass_output,
        }

    def _gather_all_pose_deform_inputs(
        self,
        ctx: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """Collect posed tokens + token pose offsets for input views and any extra
        target poses (shared TokenCascade/ProjectionFetch pose-dep epilogue).

        Returns (all_posed_avatar_token, all_token_pose_offset, n_extra) where
        n_extra is None when no extra-pose branch was taken.
        """
        B, V = ctx["B"], ctx["V"]
        L, D = state["L"], state["D"]
        avatar_tokens = state["avatar_tokens"]
        pose_embed_all = state["pose_embed_all"]
        posed_token_additive_all = state["posed_token_additive_all"]
        last_posed_avatar_token = state["last_posed_avatar_token"]
        token_pose_offset = state["token_pose_offset"]
        input_view_indices = ctx["input_view_indices"]

        posedep_out_ch = 10 if self.deform_shape else 3
        all_token_pose_offset = token_pose_offset.view(B * V, -1, posedep_out_ch)
        index_start = max(input_view_indices) + 1
        if index_start <= pose_embed_all.shape[1]:
            n_extra = pose_embed_all.shape[1] - index_start
            B_extra = B * n_extra
            extra_additive = posed_token_additive_all[:, index_start:].reshape(B_extra, L, D)
            extra_posed = self.pose_dep_mlp(avatar_tokens.expand(B_extra, -1, -1) + extra_additive)
            all_posed_avatar_token = torch.cat([last_posed_avatar_token, extra_posed], dim=0)
            extra_token_pose_offset = self._predict_token_deform_for_block(extra_posed, len(self.rope3d_blocks) - 1)
            all_token_pose_offset = torch.cat([all_token_pose_offset, extra_token_pose_offset], dim=0)
        else:
            all_posed_avatar_token = last_posed_avatar_token
            n_extra = None
        return all_posed_avatar_token, all_token_pose_offset, n_extra

    def _finalize_passes(
        self,
        result_pass: Dict[str, Any],
        state: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict:
        """Attach cascade extras to the final pass and build the forward result."""
        if state["token_gs_pair"] is not None:
            result_pass["token_canonical_gs_dic"] = state["token_gs_pair"][0]
            result_pass["token_gs_delta"] = state["token_gs_pair"][1]
        result_pass["intermediate_token_offsets"] = state["intermediate_token_offsets"]
        result_pass["avatar_tokens"] = state["avatar_tokens"]

        pass_outputs = state["pass_outputs"]
        if self.return_intermediate_results and pass_outputs:
            pass_outputs[-1] = result_pass
        else:
            pass_outputs = [result_pass]

        return {
            "passes": pass_outputs,
            "render_input": [],
            "backbone_output": ctx["backbone_output"],
            "lbs_cache": ctx["full_lbs_cache"],
        }

    def forward(
        self,
        batch: Dict,
        gs_tpose_dic: Optional[dict] = None,
        debug_save_folder: str = "",
    ) -> Dict:
        ctx = self._forward_prologue(batch, gs_tpose_dic)
        state = self._cascade_encode(batch, ctx, debug_save_folder=debug_save_folder)

        dense_out = self._decode_dense_outputs(
            state["avatar_tokens"], ctx["gs_tpose_dic"], state["token_offset"],
        )

        deform_dic = {}
        if self.pose_dep_enabled:
            # Input-view tokens already computed; compute any extra target poses.
            all_posed_avatar_token, all_token_pose_offset, _ = \
                self._gather_all_pose_deform_inputs(ctx, state)
            # generate deformation for input and target images
            deform_dic = self.compute_deform_dic(all_posed_avatar_token, all_token_pose_offset)

        final_pass = state["build_pass_output"](
            dense_out=dense_out,
            deform_dic=deform_dic,
        )
        return self._finalize_passes(final_pass, state, ctx)
