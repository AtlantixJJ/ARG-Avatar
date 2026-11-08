from typing import Dict, Optional, NamedTuple

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from xlib.model.mhr_utils import mhr_model_params_to_joint_pos_rot6d
from xlib.runtime_assets import DINO_CONFIG_REVISION, DINO_REPO, ROOT


logger = logging.getLogger(__name__)


class ConditionerOutput(NamedTuple):
    spatial_cond_tokens: Optional[torch.Tensor]       # [B, N_spatial, C] or None
    spatial_cond_coords: Optional[torch.Tensor]       # [N_spatial, 2] original-image-space (x, y) coords, or None
    nonspatial_cond_tokens: Optional[torch.Tensor]  # [B, N_nonspatial, C] or None
    spatial_feat_hw: Optional[tuple[int, int]]       # active spatial feature-map size backing image tokens


def _build_spatial_coords(
    H: int,
    W: int,
    device: torch.device,
    coord_h: Optional[int] = None,
    coord_w: Optional[int] = None,
) -> torch.Tensor:
    """Build [H*W, 2] grid of pixel-center coords in a target image space."""
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    if coord_h is None:
        coord_h = H
    if coord_w is None:
        coord_w = W
    x = (x + 0.5) * (float(coord_w) / float(W))
    y = (y + 0.5) * (float(coord_h) / float(H))
    return torch.stack([x.flatten(), y.flatten()], dim=-1)  # [H*W, 2]


class AvatarConditioner(nn.Module):
    """Manages all conditioning sources for the avatar cross-attention KV side.

    Also owns the optional HR image backbone (ConvNeXt) that was previously
    in ARGNetBaseUV._setup_uv_hrnet_backbones.
    """
    def __init__(
        self,
        cfg: dict,
        cond_dim: int,
        mhr_animatable_gs,
        image_mean: torch.Tensor,
        image_std: torch.Tensor,
        backbone_dtype: torch.dtype,
        backbone_dim: int = None,
    ):
        super().__init__()
        self.sources = cfg.get("sources", ["sam3d_backbone_image"])
        logger.info("AvatarConditioner: sources=%s", self.sources)
        self.cond_dim = cond_dim
        self.mhr_animatable_gs = mhr_animatable_gs  # ref, not owned
        self.need_last_feature = "sam3d_backbone_feature" in self.sources
        self.trainable_module_names = []
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        self.backbone_dtype = backbone_dtype

        # MHR projection
        if "mhr_raw" in self.sources:
            mhr_cfg = cfg.get("mhr_raw", {})
            self.mhr_embed_mode = mhr_cfg.get("embed_mode", "flat")
            if self.mhr_embed_mode == "flat":
                self.mhr_proj = nn.Sequential(
                    nn.Linear(204, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim)
                )
                self.trainable_module_names.append("mhr_proj")
            elif self.mhr_embed_mode == "joint_6d_expanded":
                self.joint_proj = nn.Sequential(
                    nn.Linear(9, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim)
                )
                self.trainable_module_names.append("joint_proj")
            else:
                raise ValueError(f"Unsupported mhr_raw embed_mode: {self.mhr_embed_mode}")

        # HR image backbone (moved from ARGNetBaseUV._setup_uv_hrnet_backbones)
        self._hr_backbone_trainable = False
        self._hr_image_input_size = cfg.get("uv_image_input_size", 1024)
        conv_extractor = cfg.get("uv_image_extractor", None)
        if "hr_image" in self.sources:
            if conv_extractor is None:
                raise ValueError("hr_image source requires uv_image_extractor in config")
            logger.info("AvatarConditioner: hr_image source enabled, loading %s", conv_extractor)
            from transformers import AutoConfig
            extractor_cfg = AutoConfig.from_pretrained(
                conv_extractor, cache_dir=ROOT / "pretrained_models",
                revision=DINO_CONFIG_REVISION if conv_extractor == DINO_REPO else None,
                local_files_only=True,
            )
            model_type = extractor_cfg.model_type
            if model_type == "dinov3_vit":
                from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
                hr_backbone = DINOv3ViTModel.from_pretrained(
                    conv_extractor, cache_dir="pretrained_models"
                )
                self._hr_vit_num_extra_tokens = 1 + hr_backbone.config.num_register_tokens
                self._hr_vit_patch_size = hr_backbone.config.patch_size
                self.hr_vit_backbone = hr_backbone
                self.hr_image_stages = None
                # mask_token is a pretraining-only param; freeze it so DDP doesn't expect its gradient
                hr_backbone.embeddings.mask_token.requires_grad = False
                out_dim = hr_backbone.config.hidden_size
                logger.info(
                    "AvatarConditioner: loaded DINOv3ViTModel (hidden_size=%d, patch_size=%d, registers=%d)",
                    out_dim, self._hr_vit_patch_size, hr_backbone.config.num_register_tokens,
                )
            else:
                from transformers.models.dinov3_convnext.modeling_dinov3_convnext import DINOv3ConvNextModel
                hr_backbone = DINOv3ConvNextModel(extractor_cfg)
                self.hr_vit_backbone = None
                self.hr_image_stages = hr_backbone.model.stages[:-1]
                out_dim = hr_backbone.config.hidden_sizes[-2]
            trainable = cfg.get("uv_image_extractor_trainable", False)
            if not trainable:
                backbone_to_freeze = self.hr_vit_backbone if self.hr_vit_backbone is not None else self.hr_image_stages
                backbone_to_freeze.eval()
                for p in backbone_to_freeze.parameters():
                    p.requires_grad = False
            self._hr_backbone_trainable = trainable
            self.project_conv = nn.Conv2d(out_dim, cond_dim, kernel_size=1)
            self.trainable_module_names.append("project_conv")
            # Project backbone features to cond_dim if they differ in channel count
            self.backbone_proj = None
            if backbone_dim is not None and backbone_dim != cond_dim:
                self.backbone_proj = nn.Conv2d(backbone_dim, cond_dim, kernel_size=1)
                self.trainable_module_names.append("backbone_proj")
            if trainable:
                if self.hr_vit_backbone is not None:
                    self.hr_vit_backbone.gradient_checkpointing_enable()
                    self.trainable_module_names.append("hr_vit_backbone")
                else:
                    self.trainable_module_names.append("hr_image_stages")
        elif conv_extractor is not None:
            logger.warning(
                "AvatarConditioner: uv_image_extractor specified but hr_image not in sources %s - skipping HR backbone",
                self.sources,
            )

    def _data_preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize images (crop_width=False only)."""
        if inputs.max() > 1 and self.image_mean.max() <= 1.0:
            inputs = inputs / 255.0
        elif inputs.max() <= 1.0 and self.image_mean.max() > 1:
            inputs = inputs * 255.0
        return (inputs - self.image_mean) / self.image_std

    def enable_trainable_modules(self) -> None:
        """Enable only the conditioner submodules selected by the active sources."""
        self.eval()
        for submodule_name in self.trainable_module_names:
            submodule = getattr(self, submodule_name)
            if isinstance(submodule, nn.Parameter):
                submodule.requires_grad = True
            else:
                submodule.train()
                for param in submodule.parameters():
                    param.requires_grad = True
        # mask_token is only used during masked pretraining, never in forward; keep it frozen
        if self.hr_vit_backbone is not None:
            self.hr_vit_backbone.embeddings.mask_token.requires_grad = False

    def _forward_hr_backbone(self, batch: Dict) -> Optional[torch.Tensor]:
        """Run HR backbone (DINOv3 ViT or ConvNext) on hr_img from batch."""
        hr_img = batch.get("hr_images", batch.get("hr_img"))
        if hr_img is None:
            return None
        if hr_img.ndim == 5:
            B, N, C, H, W = hr_img.shape
            hr_img = hr_img.reshape(B * N, C, H, W)

        x = self._data_preprocess(hr_img)
        target = self._hr_image_input_size
        if x.shape[-1] != target:
            x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)

        ctx = torch.autocast("cuda", dtype=self.backbone_dtype)
        if self.hr_vit_backbone is not None:
            # DINOv3 ViT path: outputs [B, 1+R+N_patches, C]; reshape patches to spatial map
            def _run_vit(img):
                out = self.hr_vit_backbone(img)
                tokens = out.last_hidden_state  # [B, 1+R+N, C]
                patch_tokens = tokens[:, self._hr_vit_num_extra_tokens:]  # [B, N, C]
                h = w = img.shape[-1] // self._hr_vit_patch_size
                feat = patch_tokens.permute(0, 2, 1).reshape(patch_tokens.shape[0], -1, h, w)
                return self.project_conv(feat)

            if self._hr_backbone_trainable:
                with ctx:
                    return _run_vit(x)
            else:
                with torch.no_grad(), ctx:
                    return _run_vit(x)
        else:
            # ConvNext path
            def _run_convnext(img):
                for stage in self.hr_image_stages:
                    img = stage(img)
                return self.project_conv(img)

            if self._hr_backbone_trainable:
                with ctx:
                    return _run_convnext(x)
            else:
                with torch.no_grad(), ctx:
                    return _run_convnext(x)

    def forward_hr_feature_pyramid(self, batch: Dict) -> list:
        """Expose all intermediate ConvNeXt stage feature maps of the HR backbone.

        Reuses the exact preprocessing of `_forward_hr_backbone`: resolve
        `hr_images`/`hr_img`, flatten [B, V, C, H, W] -> [B*V, C, H, W],
        `_data_preprocess`, resize to `self._hr_image_input_size`, and the same
        autocast(`self.backbone_dtype`) + trainable/frozen no_grad pattern.
        ConvNeXt only: stage 0 of `self.hr_image_stages` owns the patchify stem
        (4x4 stride-4 conv inside its `downsample_layers`), so it consumes the
        raw preprocessed image directly — identical to the live
        `_forward_hr_backbone` stage loop.

        Returns:
            list of dicts, one per stage in `self.hr_image_stages`, each with:
              - "feature_map": Tensor [B*V, C_s, H_s, W_s], the raw stage output
                (no `project_conv`); carries gradients when
                `self._hr_backbone_trainable` is True.
              - "stride": int, `self._hr_image_input_size // W_s`
                (4/8/16 for dinov3-convnext-tiny `stages[:-1]`).
              - "coord_h", "coord_w": int, the preprocessed HR input frame
                (`self._hr_image_input_size` for both); callers scale projected
                xy by (W_s / coord_w).
        """
        if self.hr_vit_backbone is not None or self.hr_image_stages is None:
            raise ValueError(
                "forward_hr_feature_pyramid requires the ConvNeXt HR backbone "
                "(hr_image_stages); the DINOv3 ViT path or a missing HR backbone "
                "has no intermediate ConvNet stages for projection-fetch."
            )
        hr_img = batch.get("hr_images", batch.get("hr_img"))
        if hr_img is None:
            raise ValueError(
                "forward_hr_feature_pyramid requires batch['hr_images'] or batch['hr_img']"
            )
        if hr_img.ndim == 5:
            B, N, C, H, W = hr_img.shape
            hr_img = hr_img.reshape(B * N, C, H, W)

        x = self._data_preprocess(hr_img)
        target = self._hr_image_input_size
        if x.shape[-1] != target:
            x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)

        ctx = torch.autocast("cuda", dtype=self.backbone_dtype)

        def _run_stages(img):
            stage_maps = []
            for stage in self.hr_image_stages:
                img = stage(img)
                stage_maps.append(img)
            return stage_maps

        if self._hr_backbone_trainable:
            with ctx:
                stage_maps = _run_stages(x)
        else:
            with torch.no_grad(), ctx:
                stage_maps = _run_stages(x)

        pyramid = []
        for feat in stage_maps:
            pyramid.append(
                {
                    "feature_map": feat,
                    "stride": target // int(feat.shape[-1]),
                    "coord_h": target,
                    "coord_w": target,
                }
            )
        return pyramid

    @staticmethod
    def _get_coord_image_size(batch: Dict) -> tuple[int, int]:
        """Use the model input image resolution as the shared RoPE image space."""
        img = batch.get("img")
        if isinstance(img, torch.Tensor):
            return int(img.shape[-2]), int(img.shape[-1])

        images = batch.get("images")
        if isinstance(images, torch.Tensor):
            return int(images.shape[-2]), int(images.shape[-1])

        raise ValueError("Expected batch['img'] or batch['images'] to infer original image size")

    def forward(
        self,
        image_embeddings: torch.Tensor,
        batch: Dict,
        pose_output: Dict,
        rendered_embeddings: Optional[torch.Tensor] = None,
        hr_feature_pyramid: Optional[list] = None,
    ) -> ConditionerOutput:
        """Build conditioning tokens split into spatial (with 2D coords) and nonspatial.

        Both types are concatenated along the sequence dimension to form the
        cross-attention KV source.  Spatial tokens carry 2D grid coordinates for
        optional RoPE positional encoding; nonspatial tokens (e.g. MHR pose
        embeddings) receive zero coordinates when RoPE is active, making the
        rotary encoding an identity for those keys.

        When `hr_feature_pyramid` (output of `forward_hr_feature_pyramid`) is
        provided, the HR ConvNeXt backbone is NOT run again: the final pyramid
        stage is the exact tensor `_forward_hr_backbone` would feed into
        `project_conv`, so only the projection is applied. This lets
        projection-fetch models share one backbone pass (and one set of backward
        activations) between conditioning and evidence fetching.
        """
        spatial_parts: list[torch.Tensor] = []
        spatial_coord_parts: list[torch.Tensor] = []
        nonspatial_parts: list[torch.Tensor] = []
        spatial_map = image_embeddings
        coord_h, coord_w = self._get_coord_image_size(batch)

        if "hr_image" in self.sources:
            if hr_feature_pyramid is not None:
                assert self.hr_vit_backbone is None, (
                    "hr_feature_pyramid reuse is only valid for the ConvNeXt HR backbone"
                )
                with torch.autocast("cuda", dtype=self.backbone_dtype):
                    conv_feats = self.project_conv(hr_feature_pyramid[-1]["feature_map"])
            else:
                conv_feats = self._forward_hr_backbone(batch)
            if conv_feats is not None:
                avatar_hr = F.interpolate(
                    image_embeddings,
                    size=conv_feats.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if self.backbone_proj is not None:
                    avatar_hr = self.backbone_proj(avatar_hr)
                spatial_map = avatar_hr + conv_feats

        for source in self.sources:
            if source == "sam3d_backbone_image":
                B, C, H, W = spatial_map.shape
                spatial_parts.append(spatial_map.flatten(2).permute(0, 2, 1))
                spatial_coord_parts.append(
                    _build_spatial_coords(H, W, spatial_map.device, coord_h=coord_h, coord_w=coord_w)
                )
            elif source == "hr_image":
                continue
            elif source == "mhr_raw":
                mhr_output = pose_output.get("mhr")
                if mhr_output is not None:
                    model_params = mhr_output["mhr_model_params"].detach()
                    if self.mhr_embed_mode == "flat":
                        nonspatial_parts.append(self.mhr_proj(model_params).unsqueeze(1))
                    elif self.mhr_embed_mode == "joint_6d_expanded":
                        per_joint_feat = mhr_model_params_to_joint_pos_rot6d(
                            model_params=model_params,
                            mhr_animatable_gs=self.mhr_animatable_gs,
                        )
                        nonspatial_parts.append(self.joint_proj(per_joint_feat))
            else:
                raise ValueError(f"Unsupported avatar conditioner source: {source}")

        if rendered_embeddings is not None:
            B_r, C_r, H_r, W_r = rendered_embeddings.shape
            spatial_parts.append(rendered_embeddings.flatten(2).permute(0, 2, 1))
            spatial_coord_parts.append(
                _build_spatial_coords(H_r, W_r, rendered_embeddings.device, coord_h=coord_h, coord_w=coord_w)
            )

        if not spatial_parts and not nonspatial_parts:
            raise ValueError("AvatarConditioner produced no conditioning tokens")

        spatial_cond_tokens = torch.cat(spatial_parts, dim=1) if spatial_parts else None
        spatial_cond_coords = torch.cat(spatial_coord_parts, dim=0) if spatial_coord_parts else None
        nonspatial_cond_tokens = torch.cat(nonspatial_parts, dim=1) if nonspatial_parts else None
        spatial_feat_hw = tuple(int(v) for v in spatial_map.shape[-2:]) if spatial_parts else None

        return ConditionerOutput(
            spatial_cond_tokens=spatial_cond_tokens,
            spatial_cond_coords=spatial_cond_coords,
            nonspatial_cond_tokens=nonspatial_cond_tokens,
            spatial_feat_hw=spatial_feat_hw,
        )
