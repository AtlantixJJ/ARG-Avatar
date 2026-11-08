"""
Animatable Gaussian pipeline using MHR class directly (from mhr.mhr.MHR).

This version uses the MHR class from the momentum/pymomentum package directly
instead of relying on a torchscript model. This provides better flexibility
and access to the full MHR API.

Key differences from mhr_animatable_gaussians.py:
- Uses MHR.from_files() to load the model directly
- Supports configurable LOD (level of detail)
- No torchscript fallback (requires momentum package)
"""
import os
import warnings
from typing import Optional, Dict, Tuple, List, Union, Any

import torch
import torch.nn.functional as F
from einops import rearrange
import roma

MOMENTUM_ENABLED = os.environ.get("MOMENTUM_ENABLED") is None
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR
        MOMENTUM_ENABLED = True
    else:
        raise ImportError
except:
    MOMENTUM_ENABLED = False

if not MOMENTUM_ENABLED:
    MHR = object

# Try to import GaussianParameters, fallback to simple dict-like class if unavailable
from xlib.render.gaussian_splatting import GaussianParameters, get_gaussian_type

from xlib.geometry.ops import matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply


def animate_gs(animatable_gs, gs_dic, pose_dic=None, **animate_kwargs):
    to_colmap = animate_kwargs.pop("to_colmap", True)
    pose_indices = animate_kwargs.pop("pose_indices", [0])
    lbs_cache = animate_kwargs.pop("lbs_cache", None)
    offset_pose = animate_kwargs.pop("offset_pose", None)

    if lbs_cache is None:
        if pose_dic:
            animatable_gs.set_animate_params(**pose_dic)
        lbs_cache = animatable_gs.precompute_lbs_cache(pose_indices=pose_indices)
        cache_pose_indices = None
    else:
        cache_pose_indices = pose_indices

    # Animation from cache is bf16-safe (indexing, barycentric interp, quaternion ops)
    L = gs_dic["sh"].shape[2]
    if L == animatable_gs.num_verts:
        return animatable_gs.animate_from_cache(
            lbs_cache,
            gs_dic,
            pose_indices=cache_pose_indices,
            to_colmap=to_colmap,
            offset_pose=offset_pose,
        )
    else:
        return animatable_gs.animate_from_cache_barycentric(
            lbs_cache,
            gs_dic,
            animatable_gs._flat_face_ids,
            animatable_gs._flat_bary_coords,
            pose_indices=cache_pose_indices,
            to_colmap=to_colmap,
            offset_pose=offset_pose,
        )


def extract_lbs_feat(
    animatable_gs,
    lbs_cache: Dict[str, torch.Tensor],
    face_ids: torch.Tensor,
    bary_coords: torch.Tensor,
    pose_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    """Extract barycentrically blended LBS features from cache."""
    uv_per_sample_tf = lbs_cache.get("uv_per_sample_tf")
    uv_face_ids = lbs_cache.get("uv_face_ids")
    uv_bary_coords = lbs_cache.get("uv_bary_coords")

    if pose_indices is None:
        pose_indices = list(range(lbs_cache["per_vertex_tf"].shape[1]))

    if (
        uv_per_sample_tf is not None
        and uv_face_ids is not None
        and uv_bary_coords is not None
        and torch.equal(face_ids.to(device=uv_face_ids.device), uv_face_ids)
        and torch.equal(
            bary_coords.to(device=uv_bary_coords.device, dtype=uv_bary_coords.dtype),
            uv_bary_coords,
        )
    ):
        return uv_per_sample_tf[:, pose_indices].reshape(
            uv_per_sample_tf.shape[0],
            len(pose_indices),
            uv_per_sample_tf.shape[2],
            12,
        )

    per_vertex_tf = lbs_cache["per_vertex_tf"][:, pose_indices]
    B, N, _, _, _ = per_vertex_tf.shape
    L = face_ids.shape[0]

    tri_v_idx = animatable_gs.faces[face_ids.to(animatable_gs.device)]
    tri_tf = per_vertex_tf[:, :, tri_v_idx]
    bc = bary_coords.to(tri_tf.device).view(1, 1, L, 3, 1, 1)
    per_sample_tf = (bc * tri_tf).sum(dim=-3)
    return per_sample_tf[:, :, :, :3, :4].reshape(B, N, L, 12)


def _trs_to_matrix(trs: torch.Tensor, to_meters: bool = True) -> torch.Tensor:
    """
    Convert packed (x, y, z, qx, qy, qz, qw, s) to 4x4 matrix.
    MHR stores translations in centimeters; to_meters divides by 100.
    """
    t = trs[..., :3]
    if to_meters:
        t = t / 100.0
    q = trs[..., 3:7]
    s = trs[..., 7:8]
    rot = roma.unitquat_to_rotmat(q)
    rot_scaled = rot * s.unsqueeze(-1)

    mat = torch.zeros(*trs.shape[:-1], 4, 4, device=trs.device, dtype=trs.dtype)
    mat[..., :3, :3] = rot_scaled
    mat[..., :3, 3] = t
    mat[..., 3, 3] = 1.0
    return mat


class MHRAnimatableGaussians:
    """
    Animatable 3D Gaussians driven by the Meta Human Rig (MHR) model.

    This class uses the MHR class directly from the momentum package.

    Key Features:
    -   **Backend:** Uses MHR class from mhr.mhr module
    -   **LOD Support:** Configurable level of detail (0-6, where 0 is highest detail)
    -   **Animation:** Accepts shape, model (pose), and expression parameters
    -   **Gaussian Anchoring:** Manages 3D Gaussians anchored to MHR mesh vertices

    Workflow:
    1.  Initialize with desired LOD
    2.  `set_animate_params()`: Feed shape/model/expression params
    3.  `animate_gaussians()`: Transform T-pose Gaussians to world space
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lod: int = 1,
        mhr_model: Optional[MHR] = None,
        mhr_model_path: Optional[str] = None,
        ignore_animation_rotation: bool = False,
    ):
        """
        Args:
            device: torch device.
            lod: Level of detail for MHR model (0-6). Default is 1 (matches TScript behavior).
            mhr_model: Optional pre-loaded MHR model instance.
            mhr_model_path: Optional path to MHR assets folder or torchscript model (.pt).
            ignore_animation_rotation: If True, the rigid body rotation from LBS is not composed
                into the Gaussian's output quaternion during animation. This is useful for
                isotropic Gaussians where orientation is not meaningful for shape.
        """
        self.device = device
        self.lod = lod
        self.ignore_animation_rotation = ignore_animation_rotation

        print(f"MHRAnimatableGaussians LOD={lod}")

        # Load MHR model
        self.use_torchscript = False
        if mhr_model is not None:
            self.mhr = mhr_model.to(device)
        elif MOMENTUM_ENABLED:
            self.mhr = MHR.from_files(device=device, lod=lod)
        else:
            if mhr_model_path is not None and os.path.exists(mhr_model_path):
                self.mhr = torch.jit.load(mhr_model_path, map_location=device)
                self.use_torchscript = True
            else:
                raise ValueError(f"MHR model path {mhr_model_path} does not exist and pymomentum is not available.")

        self.mhr.eval()
        for param in self.mhr.parameters():
            param.requires_grad = False

        # Blendshape counts (torchscript models may omit expression blendshapes)
        self.num_identity_blendshapes, self.num_expr_blendshapes = self._get_blendshape_counts()

        # Static buffers from MHR
        ct = self.mhr.character_torch
        self.rest_vertices = ct.mesh.rest_vertices.to(device) / 100.0  # meters
        self.faces = ct.mesh.faces.long()
        self.inverse_bind_pose = ct.linear_blend_skinning.inverse_bind_pose.to(device)
        self.skin_indices_flattened = ct.linear_blend_skinning.skin_indices_flattened.to(device).long()
        self.skin_weights_flattened = ct.linear_blend_skinning.skin_weights_flattened.to(device)
        self.vert_indices_flattened = ct.linear_blend_skinning.vert_indices_flattened.to(device).long()
        self.joint_parents = ct.skeleton.joint_parents.to(device).long()

        self.num_joints = self.inverse_bind_pose.shape[0]
        self.num_verts = self.rest_vertices.shape[0]

        gaussian_type = get_gaussian_type()
        scale_dim = 3 if gaussian_type == "3d" else 2

        # Animation state
        self.offset_xyz = torch.zeros((1, 1, self.num_verts, 3), device=device)
        self.scale = torch.ones((1, 1, self.num_verts, scale_dim), device=device)
        self.rotation = torch.tensor([1, 0, 0, 0], device=device, dtype=torch.float32)[None, None, None, :].repeat(1, 1, self.num_verts, 1)
        self.opacity = torch.ones((1, 1, self.num_verts, 1), device=device)
        self.sh = torch.zeros((1, 1, self.num_verts, 1, 3), device=device)

        self.shape_params = None
        self.model_params = None
        self.expr_params = None
        self.apply_pose_correctives = True

        # Optional UV barycentric buffers for animate_gaussians_barycentric()
        self._uv_sample_topology = None
        self._flat_face_ids = None    # [L] int — flat sample face indices
        self._flat_bary_coords = None # [L, 3] float — flat barycentric weights

        rest_pose = self._compute_rest_pose(
            torch.zeros(1, self.num_identity_blendshapes, device=device)
        )
        # Store as [V, 3] to match downstream anchor-point expectations.
        self.query_points = rest_pose[0] if rest_pose.ndim == 3 else rest_pose

    def to(self, device):
        """Move buffers and model to device."""
        self.device = device
        self.mhr = self.mhr.to(device)
        for attr in ["rest_vertices", "inverse_bind_pose", "skin_indices_flattened",
                     "skin_weights_flattened", "vert_indices_flattened", "joint_parents",
                     "offset_xyz", "scale", "rotation", "opacity", "sh", "query_points", "faces"]:
            setattr(self, attr, getattr(self, attr).to(device))
        for attr in ["_flat_face_ids", "_flat_bary_coords"]:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(self, attr, val.to(device))
        # Move sample topology tensors to device
        topo = getattr(self, "_uv_sample_topology", None)
        if topo is not None:
            for attr in ["face_ids", "bary_coords", "anchor_points", "component_ids"]:
                val = getattr(topo, attr, None)
                if val is not None and isinstance(val, torch.Tensor):
                    setattr(topo, attr, val.to(device))
        return self

    # ----------- Core MHR helpers -----------
    def _compute_joint_parameters(self, model_params: torch.Tensor) -> torch.Tensor:
        """
        model_params: [B, 204]
        identity: [B, 45] (unused, kept for API compatibility)
        returns: joint_parameters [B, J, 8] from parameter transform
        """

        pad_size = self.num_identity_blendshapes

        if not self.use_torchscript:
            pad_size += self.num_expr_blendshapes
        padding = torch.zeros(model_params.shape[0], pad_size).to(model_params).requires_grad_(False)
        joint_parameters = torch.cat([model_params, padding], dim=1)
        return self.mhr.character_torch.model_parameters_to_joint_parameters(joint_parameters)

    def _compute_rest_pose(self, identity: torch.Tensor) -> torch.Tensor:
        """
        Returns rest pose vertices after blendshapes (before pose correctives), in meters.

        Args:
            identity: [B, 45] identity blendshape coefficients
            expr: [B, 72] face expression coefficients
        """
        # Ensure identity and expr have compatible batch dimensions
        if identity.ndim == 1:
            identity = identity.unsqueeze(0)
        
        # a mismatch between torchscript and pymomentum MHR model
        if not self.use_torchscript:
            pad = torch.zeros(identity.shape[0], self.num_expr_blendshapes, device=identity.device)
            identity = torch.cat([identity, pad], dim=1)

        rest_pose = self.mhr.character_torch.blend_shape.forward(identity)
        return rest_pose / 100.0  # Convert from cm to meters

    def _apply_pose_correctives(self, joint_parameters: torch.Tensor, rest_pose: torch.Tensor) -> torch.Tensor:
        if self.apply_pose_correctives and self.mhr.pose_correctives_model is not None:
            corr = self.mhr.pose_correctives_model(joint_parameters)
            rest_pose = rest_pose + corr / 100.0  # corr is in cm
        return rest_pose

    def _compute_skeleton_state(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        return self.mhr.character_torch.joint_parameters_to_skeleton_state(joint_parameters)

    def _blend_transforms(self, joint_mats: torch.Tensor) -> torch.Tensor:
        """
        joint_mats: [BN, J, 4, 4]
        returns per-vertex transform matrices [BN, V, 4, 4]
        """
        ibp = _trs_to_matrix(self.inverse_bind_pose, to_meters=True)  # [J,4,4]
        comp = torch.matmul(joint_mats.index_select(1, self.skin_indices_flattened), ibp.index_select(0, self.skin_indices_flattened))
        weights = self.skin_weights_flattened.view(1, -1, 1, 1)
        comp = comp * weights  # [BN, K, 4, 4]
        per_vertex = torch.zeros(joint_mats.shape[0], self.num_verts, 4, 4, device=joint_mats.device, dtype=joint_mats.dtype)
        per_vertex.index_add_(1, self.vert_indices_flattened, comp)
        return per_vertex

    # ----------- Public API -----------
    def set_animate_params(
        self,
        shape_params: torch.Tensor,
        model_params: torch.Tensor,
        expr_params: Optional[torch.Tensor] = None,
        apply_pose_correctives: bool = True,
        detach: bool = False,
        **kwargs,
    ):
        """
        Store animation parameters.
        Args:
            shape_params: [B, 45]
            model_params: [B, N, 204] or [B, 204]
            expr_params: [B, N, 72] or None (defaults to zeros)
            detach: If True, detaches inputs before storing them on this module.
                This avoids accidentally retaining autograd graphs across iterations
                because these tensors are stored as persistent attributes.
        """
        if detach:
            shape_params = shape_params.detach()
            model_params = model_params.detach()
            if expr_params is not None:
                expr_params = expr_params.detach()

        assert shape_params.ndim == 2, f"shape_params must be [B, 45], got {tuple(shape_params.shape)}"
        assert model_params.ndim in (2, 3), (
            f"model_params must be [B, N, 204] or [B, 204], got {tuple(model_params.shape)}"
        )
        if expr_params is not None:
            assert expr_params.ndim in (2, 3), (
                f"expr_params must be [B, N, 72], got {tuple(expr_params.shape)}"
            )

        if model_params.ndim == 2:
            model_params = model_params[:, None, :]
            if expr_params is not None and expr_params.ndim == 2:
                expr_params = expr_params[:, None, :]

        B, N, _ = model_params.shape
        assert shape_params.shape[0] == B, (
            f"Batch mismatch: shape_params batch {shape_params.shape[0]} vs model_params batch {B}"
        )
        assert shape_params.shape[1] == self.num_identity_blendshapes, (
            f"shape_params dim must be {self.num_identity_blendshapes}, got {shape_params.shape[1]}"
        )

        expected_model_dim = self.mhr.character_torch.parameter_transform.parameter_transform.shape[1]
        expected_model_dim = int(expected_model_dim) - self.num_identity_blendshapes
        if not self.use_torchscript:
            expected_model_dim -= self.num_expr_blendshapes
        assert model_params.shape[-1] == expected_model_dim, (
            f"model_params dim must be {expected_model_dim}, got {model_params.shape[-1]}"
        )

        if expr_params is None:
            expr_params = torch.zeros(B, N, self.num_expr_blendshapes, device=model_params.device)
        else:
            assert expr_params.shape[0] == B, (
                f"Batch mismatch: expr_params batch {expr_params.shape[0]} vs model_params batch {B}"
            )
            assert expr_params.shape[1] == N, (
                f"Sequence mismatch: expr_params frames {expr_params.shape[1]} vs model_params frames {N}"
            )
            expected_expr = self.num_expr_blendshapes
            if expected_expr == 0:
                expr_params = expr_params[..., :0]
            elif expr_params.shape[-1] != expected_expr:
                if expr_params.shape[-1] > expected_expr:
                    expr_params = expr_params[..., :expected_expr]
                else:
                    pad = expected_expr - expr_params.shape[-1]
                    expr_params = F.pad(expr_params, (0, pad))
        self.shape_params = shape_params.to(self.device).float()
        self.model_params = model_params.to(self.device).float()
        self.expr_params = expr_params.to(self.device).float()
        self.apply_pose_correctives = apply_pose_correctives

    def _get_blendshape_counts(self) -> tuple[int, int]:
        """Infer identity/expression blendshape counts, guarding torchscript models."""
        ct = self.mhr.character_torch
        total = None
        if hasattr(ct, "blend_shape") and hasattr(ct.blend_shape, "shape_vectors"):
            try:
                total = int(ct.blend_shape.shape_vectors.shape[0])
            except Exception:
                total = None

        identity = None
        expr = None
        if hasattr(self.mhr, "get_num_identity_blendshapes"):
            try:
                identity = int(self.mhr.get_num_identity_blendshapes())
            except Exception:
                identity = None
        if hasattr(self.mhr, "get_num_face_expression_blendshapes"):
            try:
                expr = int(self.mhr.get_num_face_expression_blendshapes())
            except Exception:
                expr = None

        if identity is None and total is not None:
            identity = total
        if expr is None:
            expr = 0

        return identity or 0, expr or 0

    def _prepare_pose(self, pose_indices):
        assert self.shape_params is not None, "Call set_animate_params first."
        B = self.shape_params.shape[0]
        N = len(pose_indices)
        shape = self.shape_params[:, None, :].expand(B, N, -1).reshape(B * N, -1)
        model = self.model_params[:, pose_indices].reshape(B * N, -1)
        expr = self.expr_params[:, pose_indices].reshape(B * N, -1)
        return shape, model, expr, B, N

    def _compute_pose_transforms(self, shape, model, expr):
        joint_params = self._compute_joint_parameters(model)
        skel_state = self._compute_skeleton_state(joint_params)
        rest_pose = self._compute_rest_pose(shape)
        rest_pose = self._apply_pose_correctives(joint_params, rest_pose)
        joint_mats = _trs_to_matrix(skel_state, to_meters=True)  # [BN, J, 4,4]
        per_vertex_tf = self._blend_transforms(joint_mats)  # [BN, V, 4,4]
        return rest_pose, per_vertex_tf, joint_mats

    def set_uv_barycentric_buffers(self, sample_topology) -> None:
        """Store UV barycentric buffers for animate_from_cache_barycentric()."""
        self._uv_sample_topology = sample_topology
        self._flat_face_ids = sample_topology.face_ids.to(self.device)
        self._flat_bary_coords = sample_topology.bary_coords.to(self.device)

    def _resolve_offset_xyz(self, tpose_gs_dic: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "offset_xyz" in tpose_gs_dic:
            return tpose_gs_dic["offset_xyz"]
        if "xyz" not in tpose_gs_dic:
            raise KeyError("Expected `offset_xyz` or `xyz` in tpose_gs_dic.")

        xyz = tpose_gs_dic["xyz"]
        if xyz.shape[2] != self.num_verts:
            raise ValueError(
                "Missing `offset_xyz` for non-vertex Gaussians. "
                "Barycentric branches require explicit offsets."
            )
        anchor_points = self.query_points[None, None].to(xyz)
        return xyz - anchor_points

    def precompute_lbs_cache(
        self, 
        pose_dics: List[Dict] = None, 
        pose_indices: List[int] = None,
        include_uv_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Pre-compute per-vertex LBS transforms for a list of poses.
        
        Returns:
            {
                "rest_pose": [B, N, V, 3],
                "per_vertex_tf": [B, N, V, 4, 4],
                "posed_mesh_verts": [B, N, V, 3],
                "rigid_quat": [B, N, V, 4],  # per-vertex quaternions
                "uv_posed_bary": [B, N, L, 3],        # optional UV-sampled positions
                "uv_rigid_quat": [B, N, L, 4],        # optional UV-sampled quaternions
                "uv_per_sample_tf": [B, N, L, 3, 4],  # optional UV-sampled transforms
            }
        """
        if pose_dics is not None:
            shape = pose_dics[0]["shape_params"]
            model = torch.cat([p["model_params"] for p in pose_dics], dim=1)
            expr = torch.cat([p["expr_params"] for p in pose_dics], dim=1)
            self.set_animate_params(shape, model, expr)
            if pose_indices is None:
                pose_indices = list(range(model.shape[1]))
        
        if pose_indices is None:
            if self.model_params is not None:
                pose_indices = list(range(self.model_params.shape[1]))
            else:
                pose_indices = [0]
            
        shape_bn, model_bn, expr_bn, B, N = self._prepare_pose(pose_indices)
        
        with torch.amp.autocast(device_type="cuda", enabled=False):
            rest_pose, per_vertex_tf, _ = self._compute_pose_transforms(
                shape_bn.float(), model_bn.float(), expr_bn.float()
            )
            # rest_pose: [BN, V, 3], per_vertex_tf: [BN, V, 4, 4]
            
            # Compute posed MESH vertices
            rest_h = torch.cat([rest_pose, torch.ones_like(rest_pose[..., :1])], dim=-1)
            posed_verts = torch.einsum("bvij,bvj->bvi", per_vertex_tf, rest_h)[..., :3]
            
            # Extract per-vertex rotation quaternions
            rot_mat = per_vertex_tf[..., :3, :3]
            rigid_quat = matrix_to_quaternion(rot_mat.float())
            rigid_quat = rigid_quat * rigid_quat[..., :1].sign() # ensure canonical quaternion
            
        lbs_cache = {
            "rest_pose": rest_pose.view(B, N, self.num_verts, 3),
            "per_vertex_tf": per_vertex_tf.view(B, N, self.num_verts, 4, 4),
            "posed_mesh_verts": posed_verts.view(B, N, self.num_verts, 3),
            "rigid_quat": rigid_quat.view(B, N, self.num_verts, 4),
        }
        if include_uv_cache:
            if self._flat_face_ids is None or self._flat_bary_coords is None:
                raise ValueError("UV cache requested before barycentric buffers were initialized.")

            face_ids = self._flat_face_ids
            bary_coords = self._flat_bary_coords
            L = face_ids.shape[0]
            tri_v_idx = self.faces[face_ids.to(self.device)]
            posed_mesh_verts_all = lbs_cache["posed_mesh_verts"].float()
            rigid_quat_all = lbs_cache["rigid_quat"].float()
            per_vertex_tf_all = lbs_cache["per_vertex_tf"].float()

            tri_posed_v = posed_mesh_verts_all[:, :, tri_v_idx]
            tri_rigid_q = rigid_quat_all[:, :, tri_v_idx]
            tri_tf = per_vertex_tf_all[:, :, tri_v_idx]

            bc_xyz = bary_coords.to(device=tri_posed_v.device, dtype=tri_posed_v.dtype).view(1, 1, L, 3, 1)
            bc_tf = bary_coords.to(device=tri_tf.device, dtype=tri_tf.dtype).view(1, 1, L, 3, 1, 1)

            lbs_cache["uv_posed_bary"] = (bc_xyz * tri_posed_v).sum(dim=-2)
            lbs_cache["uv_rigid_quat"] = F.normalize((bc_xyz * tri_rigid_q).sum(dim=-2), dim=-1)
            lbs_cache["uv_per_sample_tf"] = (bc_tf * tri_tf).sum(dim=-3)[:, :, :, :3, :4]
            lbs_cache["uv_face_ids"] = face_ids.clone()
            lbs_cache["uv_bary_coords"] = bary_coords.clone()

        return lbs_cache

    def animate_from_cache(
        self,
        lbs_cache: Dict[str, torch.Tensor],
        tpose_gs_dic: dict,
        pose_indices: List[int] = None,
        to_colmap: bool = True,
        offset_pose: Optional[torch.Tensor] = None,
    ) -> Tuple[GaussianParameters, torch.Tensor]:
        """Animate Gaussians using pre-computed vertex-level LBS cache.
        Args:
            offset_pose: [B, 1, V, 3] override on tpose_gs_dic["offset_xyz"]
        """
        if pose_indices is None:
            pose_indices = list(range(lbs_cache["per_vertex_tf"].shape[1]))
        device, dtype = lbs_cache["per_vertex_tf"].device, lbs_cache["per_vertex_tf"].dtype
            
        per_vertex_tf = lbs_cache["per_vertex_tf"][:, pose_indices] # [B, N, V, 4, 4]
        rigid_quat = lbs_cache["rigid_quat"][:, pose_indices] # [B, N, V, 4]
        posed_mesh_verts = lbs_cache["posed_mesh_verts"][:, pose_indices] # [B, N, V, 3]

        offset_xyz = tpose_gs_dic['offset_xyz']
        rotation = tpose_gs_dic["rotation"] # [B, 1, V, 4]
        scale = tpose_gs_dic["scale"]
        opacity = tpose_gs_dic["opacity"]
        sh = tpose_gs_dic["sh"]

        if offset_pose is None:
            offset_pose = torch.tensor(0.0, device=device, dtype=dtype)

        B, N, V, _, _ = per_vertex_tf.shape

        total_offset = offset_xyz + offset_pose

        # rest_pose + offset_xyz -> base_mean
        rest_pose = lbs_cache["rest_pose"][:, pose_indices] # [B, N, V, 3]
        base_mean = rest_pose + total_offset
        base_h = torch.cat([base_mean, torch.ones_like(base_mean[..., :1])], dim=-1) # [B, N, V, 4]

        # apply per_vertex_tf via matmul -> posed_xyz
        posed_xyz = torch.einsum("bnvij,bnvj->bnvi", per_vertex_tf, base_h)[..., :3]

        # Compose rotation
        if self.ignore_animation_rotation:
            posed_quat = rotation.expand(B, N, V, 4)
        else:
            posed_quat = quaternion_multiply(rigid_quat, rotation.expand(B, N, V, 4))

        # pose-dependent offsets: [B, L, D] stored in tpose_gs_dic; unsqueeze to broadcast over N
        offset_pose_rotation = tpose_gs_dic.get("offset_pose_rotation")
        if offset_pose_rotation is not None:
            posed_quat = quaternion_multiply(posed_quat, offset_pose_rotation.float().unsqueeze(1))

        expanded_scale = scale.expand(B, N, V, -1)
        offset_pose_scale_ = tpose_gs_dic.get("offset_pose_scale_")
        if offset_pose_scale_ is not None:
            expanded_scale = expanded_scale * offset_pose_scale_.float().unsqueeze(1)

        gs_param = GaussianParameters(
            xyz=posed_xyz,
            scale=expanded_scale,
            rotation=posed_quat,
            opacity=opacity.expand(B, N, V, 1),
            sh=sh.expand(B, N, V, -1, 3),
            active_sh_degree=int(sh.shape[-2] ** 0.5) - 1,
        )
        
        if to_colmap:
            sign = torch.tensor([1, 1, -1, -1], device=gs_param.xyz.device, dtype=gs_param.xyz.dtype)
            gs_param.xyz = gs_param.xyz * sign[1:].view(1, 1, 1, -1)
            gs_param.rotation = gs_param.rotation * sign.view(1, 1, 1, -1)
            posed_mesh_verts = posed_mesh_verts * sign[1:].view(1, 1, 1, -1)
            
        return gs_param, posed_mesh_verts

    def animate_from_cache_barycentric(
        self,
        lbs_cache: Dict[str, torch.Tensor],
        tpose_gs_dic: dict,
        face_ids: torch.Tensor,        # [L]
        bary_coords: torch.Tensor,     # [L, 3]
        pose_indices: List[int] = None,
        to_colmap: bool = True,
        offset_pose: Optional[torch.Tensor] = None,
    ) -> Tuple[GaussianParameters, torch.Tensor]:
        """Animate barycentric samples using pre-computed cache."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if pose_indices is None:
                pose_indices = list(range(lbs_cache["per_vertex_tf"].shape[1]))

            uv_posed_bary = lbs_cache.get("uv_posed_bary")
            uv_rigid_quat = lbs_cache.get("uv_rigid_quat")
            uv_face_ids = lbs_cache.get("uv_face_ids")
            uv_bary_coords = lbs_cache.get("uv_bary_coords")

            use_uv_cache = (
                uv_posed_bary is not None
                and uv_rigid_quat is not None
                and uv_face_ids is not None
                and uv_bary_coords is not None
                and torch.equal(face_ids.to(device=uv_face_ids.device), uv_face_ids)
                and torch.equal(
                    bary_coords.to(device=uv_bary_coords.device, dtype=uv_bary_coords.dtype),
                    uv_bary_coords,
                )
            )

            if use_uv_cache:
                posed_mesh_verts_all = lbs_cache["posed_mesh_verts"][:, pose_indices].float()
                posed_bary = uv_posed_bary[:, pose_indices].float()
                rigid_quat_blended = uv_rigid_quat[:, pose_indices].float()
                B, N, L, _ = posed_bary.shape
            else:
                posed_mesh_verts_all = lbs_cache["posed_mesh_verts"][:, pose_indices].float() # [B, N, V, 3]
                rigid_quat_all = lbs_cache["rigid_quat"][:, pose_indices].float() # [B, N, V, 4]

                B, N, _, _ = posed_mesh_verts_all.shape
                L = face_ids.shape[0]

                tri_v_idx = self.faces[face_ids.to(self.device)]
                tri_posed_v = posed_mesh_verts_all[:, :, tri_v_idx]
                tri_rigid_q = rigid_quat_all[:, :, tri_v_idx]

                bc = bary_coords.to(device=tri_posed_v.device, dtype=tri_posed_v.dtype).view(1, 1, L, 3, 1)
                posed_bary = (bc * tri_posed_v).sum(dim=-2) # [B, N, L, 3]
                rigid_quat_blended = (bc * tri_rigid_q).sum(dim=-2) # [B, N, L, 4]
                rigid_quat_blended = F.normalize(rigid_quat_blended, dim=-1)

            offset_xyz = tpose_gs_dic["offset_xyz"].float()
            rotation = tpose_gs_dic["rotation"].float() # [B, 1, L, 4]
            scale = tpose_gs_dic["scale"].float() # [B, 1, L, 3]

            if offset_pose is None:
                offset_pose = tpose_gs_dic.get("offset_pose")

            if offset_pose is None:
                offset_pose = torch.zeros_like(offset_xyz)
            else:
                offset_pose = offset_pose.float()

            total_offset = offset_xyz + offset_pose # [B, 1, L, 3]

            def rotate_vector_by_quaternion(q, v):
                q_w = q[..., :1]
                q_vec = q[..., 1:]
                t = 2 * torch.linalg.cross(q_vec, v)
                return v + (q_w * t) + torch.linalg.cross(q_vec, t)

            offset_posed = rotate_vector_by_quaternion(rigid_quat_blended, total_offset.expand(B, N, L, 3))
            posed_xyz = posed_bary + offset_posed

            if self.ignore_animation_rotation:
                posed_rotation = rotation.expand(B, N, L, 4)
            else:
                posed_rotation = quaternion_multiply(rigid_quat_blended, rotation.expand(B, N, L, 4))

            # pose-dependent offsets: [B, L, D] stored in tpose_gs_dic; unsqueeze to broadcast over N
            offset_pose_rotation = tpose_gs_dic.get("offset_pose_rotation")
            if offset_pose_rotation is not None:
                posed_rotation = quaternion_multiply(posed_rotation, offset_pose_rotation.float().unsqueeze(1))

            expanded_scale = scale.expand(B, N, L, -1)
            offset_pose_scale_ = tpose_gs_dic.get("offset_pose_scale_")
            if offset_pose_scale_ is not None:
                expanded_scale = expanded_scale * offset_pose_scale_.float().unsqueeze(1)

            opacity = tpose_gs_dic["opacity"].float()
            sh = tpose_gs_dic["sh"].float()

            gs_param = GaussianParameters(
                xyz=posed_xyz,
                scale=expanded_scale,
                rotation=posed_rotation,
                opacity=opacity.expand(B, N, L, 1),
                sh=sh.expand(B, N, L, -1, 3),
                active_sh_degree=int(sh.shape[-2] ** 0.5) - 1,
            )

            if to_colmap:
                sign = torch.tensor([1, 1, -1, -1], device=gs_param.xyz.device, dtype=gs_param.xyz.dtype)
                gs_param.xyz = gs_param.xyz * sign[1:].view(1, 1, 1, -1)
                gs_param.rotation = gs_param.rotation * sign.view(1, 1, 1, -1)
                posed_mesh_verts_all = posed_mesh_verts_all * sign[1:].view(1, 1, 1, -1)

            return gs_param, posed_mesh_verts_all

    def get_root_rigid_in_colmap(
        self,
        model_params_b: torch.Tensor,
    ) -> tuple:
        """Compute the rigid transform from zero-root pose to full pose, in colmap space."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            mp_full = model_params_b.reshape(-1, model_params_b.shape[-1]).float()
            mp_zero = mp_full.clone()
            mp_zero[:, :6] = 0

            jp_full = self._compute_joint_parameters(mp_full)
            jp_zero = self._compute_joint_parameters(mp_zero)

            ss_full = self._compute_skeleton_state(jp_full)
            ss_zero = self._compute_skeleton_state(jp_zero)

            T_full = _trs_to_matrix(ss_full[:, 1], to_meters=True)
            T_zero = _trs_to_matrix(ss_zero[:, 1], to_meters=True)
            M = T_full @ torch.linalg.inv(T_zero)
            R_mhr = M[:, :3, :3]
            t_mhr = M[:, :3, 3]

        flip = torch.tensor([1.0, -1.0, -1.0], device=R_mhr.device, dtype=torch.float32)
        R_colmap = R_mhr.float() * flip.view(1, 1, 3) * flip.view(1, 3, 1)
        t_colmap = t_mhr.float() * flip.view(1, 3)
        return R_colmap, t_colmap
    def state_dict(self) -> Dict[str, torch.Tensor]:
        keys = ["offset_xyz", "scale", "rotation", "opacity", "sh"]
        return {k: getattr(self, k) for k in keys}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v.to(self.device))
        return self

    def forward_mhr(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: Optional[torch.Tensor] = None,
        apply_correctives: bool = True,
    ):
        """
        Direct forward pass through MHR model.

        Args:
            identity_coeffs: [B, 45] identity blendshape coefficients
            model_parameters: [B, 204] model parameters
            face_expr_coeffs: [B, 72] face expression coefficients (optional)
            apply_correctives: whether to apply pose correctives

        Returns:
            verts: [B, V, 3] vertices in centimeters
            skel_state: [B, J, 8] skeleton state
        """
        return self.mhr(identity_coeffs, model_parameters, face_expr_coeffs, apply_correctives)
