from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import torch
import torch.nn as nn

from xlib.model.sam_3d_body.models.meta_arch.sam3db_body_base import SAM3DBodyBase
from xlib.model.sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from torchvision.transforms import ToTensor
from xlib.model.multimodal_transformer import MultiModalTransformer
from xlib.basetype import TensorDict
from xlib.model.gs_layer import GSLayer
from xlib.model.common import MLP as PTMLP
from xlib.model.mhr_animatable_gaussians import MHRAnimatableGaussians, animate_gs
from xlib.render.gaussian_splatting import splat_gaussians
from xlib.render.camera import GSCamera
from xlib.model_input_utils import create_single_view_batch
from xlib.train.train_helper import factorize_root_rotation
from xlib.model.model_utils import ModelUtils
from xlib.model.hand_refinement import HandRefinementMixin
from simple_knn._C import distCUDA2


def compute_scale_base(pts: torch.Tensor) -> torch.Tensor:
    """Compute per-point scale_base = sqrt(closest_neighbor_squared_distance).

    Uses simple_knn.distCUDA2 on self.mhr_animatable_gs.query_points [N, 3] to get average distance.
    """
    pts = pts.cuda().float().contiguous()
    dist2 = torch.clamp_min(distCUDA2(pts), 0.0000001)  # [N]
    scale_base = torch.sqrt(dist2)  # [N]
    # Two times average distance
    return scale_base.mean(dim=-1, keepdim=True)


class ARGNetBase(SAM3DBodyBase, HandRefinementMixin, ModelUtils):
    """
    A unified 3D Avatar reconstruction model integrating SAM-3D-Body backbone.
    
    This model reconstructs high-fidelity 3D human avatars from multi-view images
    by combining a frozen SAM-3D-Body pose estimator with a learnable avatar branch.
    
    Key Features:
    - Shared Backbone: Uses SAM-3D-Body ViT features for both pose and avatar.
    - F4D Avatar Encoder: Learns point-based avatar representation from canonical points.
    - Multi-Modal Transformer: Fuses avatar tokens with image and pose information.
    - Hand Refinement: Optionally performs high-resolution hand pose refinement using raw images.
    - Joint Animation: Skins the canonical Gaussians using the estimated body/hand pose.

    Args:
        cfg: The SAM 3D Body config (YACS-style uppercase keys).
        avatar_cfg: The avatar config (OmegaConf-style lowercase keys).
    """

    def __init__(self, cfg, avatar_cfg: dict):
        super().__init__(cfg)
        self.trainable_modules = []
        self._initialize_base_avatar(cfg, avatar_cfg)
        self._initialize_avatar(avatar_cfg)

    def _initialize_base_avatar(self, cfg, avatar_cfg: dict):
        """
        Initialize common avatar components shared by all variants.
        
        Sets up:
        - MHRAnimatableGaussians: The skinning engine for GS animation.
        - Hand crop transforms: Preprocessing for hand refinement.
        """
        self.cfg = cfg
        self.avatar_cfg = avatar_cfg

        # Initialize MHR animatable Gaussians
        mhr_model_path = cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH
        
        mhr_lod = 1
        if avatar_cfg is not None and "mhr" in avatar_cfg and "mhr_lod" in avatar_cfg["mhr"]:
            mhr_lod = avatar_cfg["mhr"]["mhr_lod"]
        elif hasattr(cfg.MODEL.MHR_HEAD, "MHR_LOD"):
            mhr_lod = cfg.MODEL.MHR_HEAD.MHR_LOD

        ignore_animation_rotation = avatar_cfg.get("mhr", {}).get("ignore_animation_rotation", False)
        # If isotropic gaussians are used, rotation during animation is not meaningful
        if avatar_cfg.get("gs_layer", {}).get("isotropic_gaussians", False):
            ignore_animation_rotation = True

        self.mhr_animatable_gs = MHRAnimatableGaussians(
            lod=mhr_lod,
            mhr_model_path=mhr_model_path,
            device=self.device,
            ignore_animation_rotation=ignore_animation_rotation,
        )

        # Hand crop transform (mirrors SAM3DBodyEstimator)
        self.transform_hand = Compose(
            [
                GetBBoxCenterScale(padding=0.9),
                TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

    def load_pretrained_weights(self, avatar_ckpt_path: Optional[str] = None):
        """
        Load SAM3DBody pretrained weights from config's CHECKPOINT_PATH.
        This handles partial loading (backbone) while ignoring avatar specific keys if they are missing.
        """
        sam3d_ckpt_path = self.cfg['CHECKPOINT_PATH'] if 'CHECKPOINT_PATH' in self.cfg else None
        if sam3d_ckpt_path and os.path.exists(sam3d_ckpt_path):
            logging.info(f"Loading SAM3DBody pretrained weights from {sam3d_ckpt_path}")
            ckpt = torch.load(sam3d_ckpt_path, map_location=self.device, weights_only=False)
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            # Load with strict=False to allow missing avatar-specific keys
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded SAM3DBody weights. Missing keys (avatar modules): {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            logging.info("No SAM3DBody pretrained weights specified or path does not exist. Starting from scratch or relying on other checkpoints.")

        if avatar_ckpt_path and os.path.exists(avatar_ckpt_path):
            logging.info(f"Loading Avatar pretrained weights from {avatar_ckpt_path}")
            ckpt = torch.load(avatar_ckpt_path, map_location=self.device, weights_only=False)
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
            
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded Avatar weights from {avatar_ckpt_path}. Missing: {len(missing)}, Unexpected: {len(unexpected)}") 

    def _setup_image_extractor(self, avatar_cfg: dict) -> int:
        """Use the SAM3DBody backbone directly for avatar image features."""
        self.avatar_image_dim = self.backbone.embed_dims
        return self.avatar_image_dim

    def _initialize_avatar(self, avatar_cfg: dict):
        """
        Initialize avatar-specific learnable components.
        
        This legacy base initializer sets up the old encoder, transformer,
        and Gaussian prediction head (GSLayer). Subclasses should override
        this method to customize the avatar architecture.
        """

        raise RuntimeError("ARGNetBase is not part of the inference-only release; use ARGNetUV variants.")

        # Image Extractor Configuration
        avatar_image_dim = self._setup_image_extractor(avatar_cfg)

        # MultiModalTransformer: avatar tokens cross-attend to pose + image tokens
        mm_cfg = avatar_cfg.get("multimodal_transformer", {})
        self._use_rope_3d = mm_cfg.get("use_rope_3d", False)
        self._rope_depth_scale = mm_cfg.get("rope_depth_scale", 1.0)

        decoder_depth = self.cfg.MODEL.DECODER.DEPTH

        self.multimodal_transformer = MultiModalTransformer(
            depth=decoder_depth,
            avatar_dim=avatar_token_dim,
            cond_dim=avatar_image_dim,
            num_heads=mm_cfg.get("num_heads", 8),
            mlp_ratio=mm_cfg.get("mlp_ratio", 4.0),
            drop_path_rate=mm_cfg.get("drop_path_rate", 0.1),
            use_image=True,
            use_rendered=False,
            use_rope_3d=False,
            rope_theta=mm_cfg.get("rope_theta", 10000.0),
        )

        # Update trainable modules list
        self.trainable_modules = [
            'avatar_encoder',
            'avatar_pe',
            'multimodal_transformer',
        ]

        # Initialize GS head
        self.trainable_modules.extend(self._initialize_gs_head(avatar_cfg))

        # Initialize model-owned placeholder parameters
        self._init_gs_tpose_init_dic_params()

    def _init_gs_tpose_init_dic_params(self):
        """Initialize canonical Gaussian placeholder parameters."""
        # Use unactivated values consistent with GSLayer.unactivate
        # scale: 5e-3 -> log(5e-3), isotropic one-dim
        # opacity: fixed, inverse_sigmoid(0.5) = 0
        # sh: 0.5 (RGB) -> inverse_sigmoid(0.5) = 0
        # rotation: fixed [1,0,0,0]
        # offset_xyz: [0,0,0]

        device = self.device
        dtype = torch.float32 # Use float32 for parameters

        # Get shapes from mhr_animatable_gs
        # mhr_animatable_gs.offset_xyz is [1, 1, N, 3] or similar
        ref_offset = self.mhr_animatable_gs.offset_xyz # [1, 1, N, 3]
        N = ref_offset.shape[2]

        # Compute and set distance-based scale_base.
        scale_base = compute_scale_base(self.mhr_animatable_gs.query_points)
        self.gshead.set_scale_base(scale_base)
        # sigmoid(0) = 0.5, so initial scale = 0.5 * scale_base
        scale_init_val = -0.5

        _sh_degree = self.avatar_cfg["gs_layer"].get("sh_degree", 0)
        _D = (_sh_degree + 1) ** 2

        self.gs_tpose_init_dic = nn.ParameterDict({
            "offset_xyz": nn.Parameter(torch.zeros(1, 1, N, 3, device=device, dtype=dtype)),
            # Isotropic raw scale init: one dim, broadcast downstream as needed.
            "scale": nn.Parameter(torch.full((1, 1, N, 1), scale_init_val, device=device, dtype=dtype)),
            "sh": nn.Parameter(torch.zeros(1, 1, N, _D, 3, device=device, dtype=dtype)),
        })

        # Fixed (non-trainable) initialization values.
        self.register_buffer(
            "gs_tpose_init_rotation",
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 1, 4).expand(1, 1, N, 4).clone(),
            persistent=False,
        )
        self.register_buffer(
            "gs_tpose_init_opacity",
            torch.zeros(1, 1, N, 1, device=device, dtype=dtype),
            persistent=False,
        )

    def get_gs_tpose_init_dic(self, batch_size: int, device: torch.device, dtype: Optional[torch.dtype] = None):
        """Materialize batch-expanded GS dict from parameters."""
        if dtype is None:
            dtype = torch.float32

        # Expand each parameter to [B, 1, N, ...]
        # Note: we use names that decode_gaussians expects in gs_tpose_dic
        # specifically scale_, opacity_, sh_ for raw values

        return TensorDict({
            "xyz": self.mhr_animatable_gs.query_points[None, None].expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype),
            "offset_xyz": self.gs_tpose_init_dic["offset_xyz"].expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype),
            "scale_": self.gs_tpose_init_dic["scale"].expand(batch_size, -1, -1, 3).to(device=device, dtype=dtype),
            "rotation": self.gs_tpose_init_rotation.expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype),
            "opacity_": self.gs_tpose_init_opacity.expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype),
            "sh_": self.gs_tpose_init_dic["sh"].expand(batch_size, -1, -1, -1, -1).to(device=device, dtype=dtype),
        })

    def _prepare_gs_tpose(self, batch_size: int, gs_tpose_dic: Optional[Dict] = None) -> Dict:
        """Prepare gs_tpose_dic: initialize from placeholders or unactivate provided dict."""
        device = self.device
        if gs_tpose_dic is None:
            gs_tpose_init = self.get_gs_tpose_init_dic(batch_size=batch_size, device=device)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                raw_for_act = {
                    "scale_": gs_tpose_init["scale_"],
                    "rotation": gs_tpose_init["rotation"],
                    "opacity_": gs_tpose_init["opacity_"],
                    "sh_": gs_tpose_init["sh_"],
                    "offset_xyz": gs_tpose_init["offset_xyz"],
                }
                activated = self.gshead.activate(raw_for_act)
                gs_tpose_init.update({
                    "scale": activated["scale"],
                    "opacity": activated["opacity"],
                    "sh": activated["sh"],
                    "rotation": activated["rotation"],
                })
                anchor_points = self.mhr_animatable_gs.query_points[None, None].to(gs_tpose_init["sh"])
                gs_tpose_init["xyz"] = anchor_points
            return gs_tpose_init

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            raw_gs = self.gshead.unactivate(gs_tpose_dic)
            gs_tpose_dic["scale_"] = raw_gs["scale_"]
            gs_tpose_dic["opacity_"] = raw_gs["opacity_"]
            gs_tpose_dic["sh_"] = raw_gs["sh_"]
            anchor_points = self.mhr_animatable_gs.query_points[None, None].to(gs_tpose_dic["sh"])
            gs_tpose_dic["xyz"] = anchor_points
        return gs_tpose_dic

    def _initialize_gs_head(self, avatar_cfg: dict):
        """Initialize Gaussian head."""
        # GS prediction operates on decoded tokens at dec_channels[-1] dimension
        decoded_dim = self.avatar_encoder.dec_channels[-1]

        gs_layer_cfg = avatar_cfg["gs_layer"]
        gs_feat_dim = gs_layer_cfg["gs_feat_dim"]
        gs_mlp_channels = gs_layer_cfg["gs_mlp_channels"]
        isotropic_gaussians = gs_layer_cfg["isotropic_gaussians"]
        fix_rotation = gs_layer_cfg["fix_rotation"]
        if isotropic_gaussians:
            fix_rotation = True

        self.gsfeat_mlp = PTMLP(
            in_channels=decoded_dim,
            hidden_channels=gs_mlp_channels,
            out_channels=gs_feat_dim,
        )
        self.gshead = GSLayer(
            in_channels=gs_feat_dim,
            sh_degree=gs_layer_cfg.get("sh_degree", 0),
            n_layers=gs_layer_cfg["n_layers"],
            clip_scaling=gs_layer_cfg["clip_scaling"],
            init_scaling=gs_layer_cfg["init_scaling"],
            init_density=gs_layer_cfg["init_density"],
            xyz_offset=True,
            xyz_offset_max_step=1.0,
            fix_opacity=gs_layer_cfg["fix_opacity"],
            fix_rotation=fix_rotation,
            isotropic_gaussians=isotropic_gaussians,
            scale_base_factor=gs_layer_cfg.get("scale_base_factor", 1.0),
        )

        return ['gsfeat_mlp', 'gshead']

    def setup_avatar_training(self):
        """Setup model for training with frozen pose parameters."""
        # Call parent to freeze all and enable trainable modules
        # set all to eval
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

        # set avatar to training only
        for module_name in self.trainable_modules:
            module = getattr(self, module_name)
            if isinstance(module, nn.Parameter):
                # Handle nn.Parameter directly
                module.requires_grad = True
            else:
                # Handle nn.Module
                module.train()
                for param in module.parameters():
                    param.requires_grad = True

        if hasattr(self, "gshead") and self.gshead is not None and self.gshead.fix_rotation:
            for param in self.gshead.out_layers["rotation"].parameters():
                param.requires_grad = False

        # mask_token is a pretraining-only param; keep it frozen regardless of trainable_modules
        if (hasattr(self, 'avatar_conditioner')
                and hasattr(self.avatar_conditioner, 'hr_vit_backbone')
                and self.avatar_conditioner.hr_vit_backbone is not None
                and hasattr(self.avatar_conditioner.hr_vit_backbone, 'embeddings')
                and hasattr(self.avatar_conditioner.hr_vit_backbone.embeddings, 'mask_token')):
            self.avatar_conditioner.hr_vit_backbone.embeddings.mask_token.requires_grad = False

    def _build_cameras(self, batch, pred_cam_t, H, W, cam_rot=None, cam_trans=None):
        """Build camera list from batch intrinsics."""
        B = batch["img"].shape[0]
        cam_int = batch.get("crop_cam_int", batch["cam_int"])
        raw_cam_int = batch.get("cam_int")
        raw_sizes = batch.get("raw_sizes")  # [B, 1, 2] or [B, 2]

        cameras = []
        for i in range(B):
            if cam_int.ndim == 4:
                K = cam_int[i, 0]
            elif cam_int.ndim == 3:
                K = cam_int[i]
            else:
                K = cam_int

            rK = None
            full_image_width, full_image_height = None, None
            if raw_cam_int is not None:
                if raw_cam_int.ndim == 4:
                    rK = raw_cam_int[i, 0]
                elif raw_cam_int.ndim == 3:
                    rK = raw_cam_int[i]
                else:
                    rK = raw_cam_int

                if raw_sizes is not None:
                    if raw_sizes.ndim == 3:
                        full_image_height = int(raw_sizes[i, 0, 0].item())
                        full_image_width = int(raw_sizes[i, 0, 1].item())
                    else:
                        full_image_height = int(raw_sizes[i, 0].item())
                        full_image_width = int(raw_sizes[i, 1].item())

            world2cam = torch.eye(4, device=K.device, dtype=K.dtype)
            if cam_rot is not None:
                world2cam[:3, :3] = cam_rot[i]
            if pred_cam_t is not None:
                cam_t = pred_cam_t[i]
                if cam_trans is not None:
                    cam_t = cam_t + cam_trans[i]
                world2cam[:3, 3] = cam_t

            cameras.append(
                GSCamera.from_extrinsics_intrinsics(
                    world2cam, K=K,
                    full_K=rK if raw_cam_int is not None else None,
                    full_image_width=full_image_width,
                    full_image_height=full_image_height,
                    image_width=W, image_height=H,
                )
            )
        return cameras

    def _ensure_factorized_input_pose_dic(self, input_pose_dic: Dict) -> Dict:
        """Factorize root rotation for views that aren't already factorized.

        Handles both multi-view [B, V, D] and single-view [B, D] mhr_model_params.
        Uses per-view is_factorized flags to skip already-factorized views.
        """
        mhr_model_params = input_pose_dic["mhr_model_params"]
        squeezed = mhr_model_params.ndim == 2
        if squeezed:
            mhr_model_params = mhr_model_params.unsqueeze(1)
            if "cam_rot" in input_pose_dic:
                input_pose_dic["cam_rot"] = input_pose_dic["cam_rot"].unsqueeze(1)
            if "cam_trans" in input_pose_dic:
                input_pose_dic["cam_trans"] = input_pose_dic["cam_trans"].unsqueeze(1)
            if "is_factorized" in input_pose_dic:
                input_pose_dic["is_factorized"] = input_pose_dic["is_factorized"].unsqueeze(1)

        V = mhr_model_params.shape[1]
        is_factorized = input_pose_dic.get("is_factorized", None)  # [B, V]

        has_cam = "cam_rot" in input_pose_dic and "cam_trans" in input_pose_dic
        cam_rots = list(input_pose_dic["cam_rot"].unbind(1)) if has_cam else [None] * V
        cam_transs = list(input_pose_dic["cam_trans"].unbind(1)) if has_cam else [None] * V
        zero_params = list(mhr_model_params.unbind(1))  # V x [B, D]

        for v_idx in range(V):
            # Skip views already factorized (all items in batch)
            if is_factorized is not None and is_factorized[:, v_idx].all():
                continue

            zero_mp, cam_rot, cam_trans = factorize_root_rotation(
                self.mhr_animatable_gs,
                mhr_model_params[:, v_idx],
            )
            zero_params[v_idx] = zero_mp
            cam_rots[v_idx] = cam_rot
            cam_transs[v_idx] = cam_trans

        input_pose_dic["mhr_model_params"] = torch.stack(zero_params, dim=1)
        input_pose_dic["cam_rot"] = torch.stack(cam_rots, dim=1)
        input_pose_dic["cam_trans"] = torch.stack(cam_transs, dim=1)

        if squeezed:
            input_pose_dic["mhr_model_params"] = input_pose_dic["mhr_model_params"].squeeze(1)
            input_pose_dic["cam_rot"] = input_pose_dic["cam_rot"].squeeze(1)
            input_pose_dic["cam_trans"] = input_pose_dic["cam_trans"].squeeze(1)

        return input_pose_dic

    def _forward_pose_estimation(
        self,
        batch: Dict,
        image_embeddings: torch.Tensor = None,
        ray_cond: torch.Tensor = None,
        thresh_wrist_angle: float = 1.4,
    ) -> Dict:
        """Forward pose estimation branch."""
        input_pose_dic = batch.get("input_pose_dic", None)
        if input_pose_dic is not None and input_pose_dic["has_pose"].all():
            pose_output = {"mhr": input_pose_dic}
        else: # for simplicity, when a batch is not fully pre-computed, we just compute them in full
            # initialize sam3dbody pose prediction
            self._max_num_person = 1
            self._batch_size = batch["img"].shape[0]
            self._person_valid = torch.tensor([True] * self._batch_size, device=batch["img"].device)

            # require raw_img, crop_bboxes, ori_img_size, bbox_center, bbox_scale,
            pose_output = self.forward_pose_branch_full(
                batch,
                get_intermediate_features=False,
                enable_hand_refinement=True,
                thresh_wrist_angle=thresh_wrist_angle,
                image_embeddings=image_embeddings,
                ray_cond=ray_cond,
            )
        self._ensure_factorized_input_pose_dic(pose_output["mhr"])
        return pose_output

    @torch.no_grad()
    def render_animated_figure(self, gs_tpose_dic, pose_output, batch, lbs_cache=None):
        """
        Render animated Gaussians from T-pose using estimated pose.
        This is a common method shared across all ARGNetBase variants.

        Args:
            gs_tpose_dic: T-pose Gaussian parameters dict
            pose_output: Pose estimation output dict with keys:
                - 'shape' or 'shape_params': [B, 45] shape parameters
                - 'mhr_model_params' or 'model_params': [B, 204] or [B, 1, 204] pose parameters
                - 'face' or 'expr_params': [B, 72] or [B, 1, 72] expression parameters
                - 'pred_cam_t': [B, 3] camera translation (optional)
            batch: Input batch with 'img', 'cam_int', 'fov_x', 'fov_y'
            lbs_cache: Optional pre-computed LBS cache.

        Returns:
            Tuple of (render_result, gs_posed) where:
                - render_result: Dict with 'image' [B, 3, H, W] and 'mask'
                - gs_posed: Animated GaussianParameters
        """
        # Extract pose parameters - handle both formats
        shape_params = pose_output['shape'] if 'shape' in pose_output else pose_output['shape_params']
        model_params = pose_output['mhr_model_params'] if 'mhr_model_params' in pose_output else pose_output['model_params']
        expr_params = pose_output['face'] if 'face' in pose_output else pose_output['expr_params']

        # Ensure model_params and expr_params have view dimension [B, V, ...]
        if model_params.ndim == 2:
            model_params = model_params.unsqueeze(1)
        if expr_params is not None and expr_params.ndim == 2:
            expr_params = expr_params.unsqueeze(1)

        temp_gs = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in gs_tpose_dic.items()}
        if "offset_xyz" not in temp_gs and "xyz" in temp_gs:
            anchor_points = self.mhr_animatable_gs.query_points[None, None].to(temp_gs["xyz"])
            temp_gs["offset_xyz"] = temp_gs["xyz"] - anchor_points

        all_indices = list(range(model_params.shape[1]))
        
        if lbs_cache is None:
            gs_posed, posed_mesh_verts = animate_gs(
                self.mhr_animatable_gs,
                temp_gs,
                dict(
                    shape_params=shape_params,
                    model_params=model_params,
                    expr_params=expr_params,
                ),
                pose_indices=all_indices,
                to_colmap=True
            )
        else:
            L = temp_gs["sh"].shape[2]
            if L == self.mhr_animatable_gs.num_verts:
                gs_posed, posed_mesh_verts = self.mhr_animatable_gs.animate_from_cache(
                    lbs_cache, temp_gs, pose_indices=all_indices, to_colmap=True
                )
            else:
                gs_posed, posed_mesh_verts = self.mhr_animatable_gs.animate_from_cache_barycentric(
                    lbs_cache, temp_gs,
                    self.mhr_animatable_gs._flat_face_ids,
                    self.mhr_animatable_gs._flat_bary_coords,
                    pose_indices=all_indices, to_colmap=True,
                )

        gs_posed_ret = gs_posed.clone()

        # Squeeze view dimension for rendering if V=1
        if gs_posed.xyz.ndim == 4 and gs_posed.xyz.shape[1] == 1:
            for key in gs_posed.tensor_keys():
                val = gs_posed[key] if key in gs_posed else None
                if val is None or not isinstance(val, torch.Tensor):
                    continue
                if val.ndim >= 2 and val.shape[1] == 1:
                    gs_posed[key] = val.squeeze(1)

        # Build cameras
        pred_cam_t = pose_output['pred_cam_t'] if 'pred_cam_t' in pose_output else None
        cam_rot = pose_output['cam_rot'] if 'cam_rot' in pose_output else None
        cam_trans = pose_output['cam_trans'] if 'cam_trans' in pose_output else None

        H, W = batch['img'].shape[-2], batch['img'].shape[-1]
        cameras = self._build_cameras(
            batch, pred_cam_t, H, W,
            cam_rot=cam_rot, cam_trans=cam_trans,
        )

        render_result = splat_gaussians(
            gs_posed, cameras,
            bg_color=torch.ones(3).to(shape_params)
        )
        render_result["cameras"] = cameras
        return render_result, gs_posed_ret, posed_mesh_verts

    def get_input_pose_dic(self, batch: Dict):
        """Run pose estimation on all views and return predicted pose cache."""
        V = batch["images"].shape[1]
        assert 'input_pose_dic' in batch, "input_pose_dic must be in batch"
        input_pose_dic = batch["input_pose_dic"]

        for v_idx in range(V):
            if not input_pose_dic['has_pose'][:, v_idx].all():
                single_batch = create_single_view_batch(batch, v_idx, batch["images"].device)
                self._initialize_batch(single_batch)
                pose_out = self.forward_pose_branch_full(
                    single_batch,
                    get_intermediate_features=False,
                    enable_hand_refinement=True)

                for k, v in pose_out['mhr'].items():
                    if isinstance(v, torch.Tensor) and k in input_pose_dic:
                        input_pose_dic[k][:, v_idx] = v

        return input_pose_dic

    def forward_pose_branch_full(
        self,
        batch: Dict,
        get_intermediate_features: bool = False,
        enable_hand_refinement: bool = True,
        thresh_wrist_angle: float = 1.4,
        image_embeddings: torch.Tensor = None,
        ray_cond: torch.Tensor = None,
    ) -> Dict:
        """
        Full pose estimation including body and hand refinement.

        Flow:
        1. Call parent forward_pose_branch for body pose
        2. Optionally run hand refinement on raw images
        3. Validate and merge hand poses
        4. Finalize pose output
        """
        # 1. Body pose estimation
        batch_size, num_person = batch["img"].shape[:2]
        body_batch_idx = list(range(batch_size * num_person))

        pose_output = self.forward_pose_branch(
            batch,
            get_intermediate_features=get_intermediate_features,
            body_batch_idx=body_batch_idx,
            hand_batch_idx=[],
            image_embeddings=image_embeddings,
            ray_cond=ray_cond,
        )

        intermediate_features = pose_output.pop("intermediate_features", None)
        last_feature = pose_output.pop("last_feature", None)
        proc_H, proc_W = batch["img"].shape[-2:]

        # 2. Hand refinement (optional)
        if enable_hand_refinement and 'raw_img' in batch:
            raw_images = batch['raw_img']
            crop_bboxes = batch['crop_bboxes']

            lhand_output, rhand_output, batch_lhand, batch_rhand = self._forward_hand_decoders(
                raw_images=raw_images,
                crop_bboxes=crop_bboxes,
                proc_H=proc_H,
                proc_W=proc_W,
                batch=batch,
                pose_output=pose_output,
                transform_hand=self.transform_hand,
            )

            # 3. Validate hand poses
            hand_valid_mask = self._compute_hand_validity_mask(
                pose_output,
                lhand_output,
                rhand_output,
                batch_lhand,
                batch_rhand,
                proc_W,
                thresh_wrist_angle,
            )

            # 4. Merge hand poses into body pose
            pose_output = self._merge_hand_poses(
                pose_output,
                lhand_output,
                rhand_output,
                hand_valid_mask,
                thresh_wrist_angle,
            )
            pose_output["hand_valid_mask"] = hand_valid_mask

            # Re-compute mhr_model_params after hand merge changed component params
            pose_output = self._finalize_pose_output(pose_output, proc_H, proc_W)

        # Re-attach intermediate features
        if intermediate_features is not None:
            pose_output["intermediate_features"] = intermediate_features
        if last_feature is not None:
            pose_output["last_feature"] = last_feature

        return pose_output

    def decode_gaussians(
        self,
        refined_avatar_tokens: torch.Tensor,
        point,
        metadata,
        gs_tpose_dic: dict,
        encoder: Optional[Any] = None,
    ) -> tuple:
        """
        Decode refined avatar tokens into Gaussian parameters.

        This method handles:
        1. Avatar encoder decode (upsample tokens)
        2. GS feature MLP
        3. GS head forward (predict deltas)
        4. Sum residuals with input gs_tpose_dic
        5. Activate to get final Gaussians

        Args:
            refined_avatar_tokens: Output from forward_avatar_branch [B, N, C]
            point: Point cloud data from avatar_encoder.encode
            metadata: Metadata from avatar_encoder.encode
            gs_tpose_dic: Input T-pose Gaussian dict with keys:
                - offset_xyz, scale_, rotation, opacity_, sh_
            encoder: Optional encoder to use for decoding. If None, uses self.avatar_encoder.

        Returns:
            Tuple of (new_gs_tpose, new_raw_gs_tpose) where:
                - new_gs_tpose: Activated Gaussian dict with xyz, scale, rotation, opacity, sh
                - new_raw_gs_tpose: Raw (pre-activation) Gaussian dict
        """
        # Decode avatar tokens (upsample)
        if encoder is None:
            encoder = self.avatar_encoder
        upsampled_tokens = encoder.decode(
            refined_avatar_tokens, point, metadata
        )

        # Generate Gaussian parameters
        gs_feat = self.gsfeat_mlp(upsampled_tokens).unsqueeze(1)
        raw_gs = self.gshead.forward_raw(gs_feat)
        delta_gs = self.gshead.presum(raw_gs)

        # Build raw input dict for residual sum
        raw_gs_tpose_dic = {
            "offset_xyz": gs_tpose_dic["offset_xyz"],
            "scale_": gs_tpose_dic["scale_"],
            "rotation": gs_tpose_dic["rotation"],
            "opacity_": gs_tpose_dic["opacity_"],
            "sh_": gs_tpose_dic["sh_"],
        }

        # Sum residuals and activate
        new_raw_gs_tpose = self.gshead.sum_residuals([raw_gs_tpose_dic, delta_gs])
        new_gs_tpose = self.gshead.activate(new_raw_gs_tpose)

        # Add xyz with anchor offset
        anchor_points = self.mhr_animatable_gs.query_points[None, None].to(new_gs_tpose["offset_xyz"])
        new_gs_tpose["offset_xyz"] = new_raw_gs_tpose["offset_xyz"]
        new_gs_tpose["xyz"] = anchor_points

        return new_gs_tpose, new_raw_gs_tpose

    def forward_avatar_branch(
        self,
        gs_tpose_dic: dict,
        spatial_cond_tokens: Optional[torch.Tensor] = None,
        spatial_cond_coords: Optional[torch.Tensor] = None,
        nonspatial_cond_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[dict, dict, torch.Tensor]:
        """
        Forward avatar branch using intermediate features from pose branch.
        """
        avatar_tokens, point, metadata = self.avatar_encoder.encode(gs_tpose_dic)

        batch_size = gs_tpose_dic['offset_xyz'].shape[0]
        avatar_tokens = avatar_tokens.expand(batch_size, -1, -1)
        avatar_tokens = avatar_tokens + self.avatar_pe

        refined_avatar_tokens = self.multimodal_transformer(
            avatar_tokens=avatar_tokens,
            spatial_cond_tokens=spatial_cond_tokens,
        )

        new_gs_tpose, new_raw_gs_tpose = self.decode_gaussians(
            refined_avatar_tokens, point, metadata, gs_tpose_dic
        )

        return new_gs_tpose, new_raw_gs_tpose, refined_avatar_tokens

    def forward(
        self,
        batch: Dict,
        gs_tpose_dic: Optional[dict] = None,
        input_pose_dic: Optional[Dict] = None,
        thresh_wrist_angle: float = 1.4,
        debug_save_folder: str = "",
    ):
        """
        Main forward pass for ARGNetBase.
        
        The execution flow is:
        1. Feature Extraction: Runs the ViT backbone (and optional avatar backbone).
        2. Pose Estimation: Runs SAM-3D-Body to get body pose, followed by 
           optional high-res hand refinement.
        3. Avatar Branch: Encodes canonical GS tokens, refines them via 
           MultiModalTransformer with image features, and decodes GS deltas.
        4. Animation & Rendering: Skins the canonical Gaussians to the estimated
           pose using MHR linear blend skinning (LBS).

        Args:
            batch: Data dictionary from DataLoader/Collate.
            gs_tpose_dic: Optional input T-pose GS (used during iterative fitting).
            input_pose_dic: Optional pre-computed pose (skips pose estimation).
            thresh_wrist_angle: Threshold for validating hand refinement results.

        Returns:
            Dict containing 'passes' list, where each pass has:
                - output_gs_pose_base: Final animated Gaussians.
                - output_canonical_gs_dic: Refined T-pose Gaussians.
                - mhr: Estimated pose parameters.
        """
        self._initialize_batch(batch)
        # 1. Forward backbone (shared between pose and avatar)
        with torch.autocast('cuda', dtype=self.backbone_dtype):
            backbone_output = self.forward_backbone(batch)

        image_embeddings = backbone_output["image_embeddings"]
        ray_cond = backbone_output["ray_cond"]

        # 2. Full pose estimation with hand refinement (or use pre-computed poses)
        pose_output = self._forward_pose_estimation(
            batch,
            image_embeddings=image_embeddings,
            ray_cond=ray_cond,
            thresh_wrist_angle=thresh_wrist_angle,
        )

        # 3. Prepare gs_tpose_dic
        B = batch["images"].shape[0] if "images" in batch else batch["img"].shape[0]
        gs_tpose_dic = self._prepare_gs_tpose(batch_size=B, gs_tpose_dic=gs_tpose_dic)

        # 4. Forward avatar branch: image embeddings are spatial conditioning
        spatial_cond_tokens = image_embeddings.flatten(2).permute(0, 2, 1)
        new_gs_tpose, new_raw_gs_tpose, avatar_tokens = self.forward_avatar_branch(
            gs_tpose_dic=gs_tpose_dic,
            spatial_cond_tokens=spatial_cond_tokens,
        )

        # Prepare pose dict for animation
        predicted_pose_dic = {
            "shape_params": pose_output["mhr"]["shape"],
            "model_params": pose_output["mhr"]["mhr_model_params"].unsqueeze(1),
            "expr_params": pose_output["mhr"]["face"].unsqueeze(1),
        }
        pred_camera = {
            "pred_cam_t": pose_output["mhr"]["pred_cam_t"],
            "cam_rot": pose_output["mhr"]["cam_rot"],
            "cam_trans": pose_output["mhr"]["cam_trans"],
        }
        
        # Animate Gaussians
        # Compute lbs_cache once
        lbs_cache = self.mhr_animatable_gs.precompute_lbs_cache(
            [predicted_pose_dic]
        )
        
        all_indices = list(range(predicted_pose_dic["model_params"].shape[1]))

        with torch.amp.autocast(device_type='cuda', enabled=False):
            gs_posed, _ = self.mhr_animatable_gs.animate_from_cache(
                lbs_cache, new_gs_tpose, pose_indices=all_indices
            )
            input_gs_posed, _ = self.mhr_animatable_gs.animate_from_cache(
                lbs_cache, gs_tpose_dic, pose_indices=all_indices
            )

        pass_output = {
            'input_canonical_gs_dic': gs_tpose_dic, # The input canonical avatar 
            'input_gs_pose': input_gs_posed, # The input canonical avatar, animated to the input pose
            'output_gs_pose_base': gs_posed,  # The output updated canonical avatar, animated to the input pose
            'output_canonical_gs_dic': new_gs_tpose, # The updated canonical avatar

            # The output of the original SAM 3D Body model
            "animate_params": predicted_pose_dic,
            "pred_camera": pred_camera,
            "mhr": pose_output["mhr"],
            "avatar_tokens": avatar_tokens
        }

        return {
            'passes': [pass_output],
            'render_input': [],
            'backbone_output': backbone_output,
        }
