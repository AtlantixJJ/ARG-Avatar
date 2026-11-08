"""Camera and related transformation.
"""
import math
import torch
import numpy as np
from PIL import Image
from typing import NamedTuple, Union, Optional, List, Tuple, Dict, Any
from scipy.spatial.transform import Rotation
from einops import rearrange, einsum
from xlib.render.camera_utils import campos_from_angle
from numpy.typing import NDArray

#from xlib.geometry.ops import matrix_to_quaternion



class GSCamera:
    """
    Camera class compatible with the Gaussian Splatting rasterizer.
    
    This class encapsulates camera extrinsics (world-to-camera), intrinsics (projection),
    and image metadata. It handles the conversion between different coordinate systems 
    (e.g., Colmap/OpenCV to the coordinate system expected by the rasterizer).

    Attributes:
        world_view_transform (Tensor): [4, 4] World-to-Camera matrix (transposed for rasterizer).
        full_proj_transform (Tensor): [4, 4] World-to-Clip matrix (transposed).
        camera_center (Tensor): [3] Camera position in world space.
        K (Tensor): [3, 3] Intrinsics matrix.
        full_K (Tensor): [3, 3] Full intrinsics matrix.
        image_width, image_height (int): Resolution.
    """

    def __init__(self,
                 world_view_transform: Optional[torch.Tensor] = None, # world to camera
                 full_proj_transform: Optional[torch.Tensor] = None, # world to screen
                 K: Optional[torch.Tensor] = None,           # NEW
                 image_width: int = 512,
                 image_height: int = 512,

                 full_K: Optional[torch.Tensor] = None,       # NEW
                 full_image_width: Optional[int] = None,       # original image size before crop
                 full_image_height: Optional[int] = None,

                 image: Optional[torch.Tensor] = None,
                 mask: Optional[torch.Tensor] = None,
                 depth: Optional[torch.Tensor] = None,
                 normal: Optional[torch.Tensor] = None,
                 image_path: Optional[str] = None,
                 mask_path: Optional[str] = None,
                 depth_path: Optional[str] = None,
                 normal_path: Optional[str] = None,

                 znear: float = 0.01,
                 zfar: float = 100):
        self.image, self.mask, self.depth, self.normal = image, mask, depth, normal
        self.image_width, self.image_height = image_width, image_height
        self.image_path = image_path
        self.mask_path, self.depth_path = mask_path, depth_path
        self.normal_path = normal_path
        self.K = K
        self.full_K = full_K if full_K is not None else self.K
        self.full_image_width = full_image_width if full_image_width is not None else image_width
        self.full_image_height = full_image_height if full_image_height is not None else image_height
        self.znear, self.zfar = znear, zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center: Optional[torch.Tensor] = None
        if world_view_transform is not None:
            orig_type = world_view_transform.dtype
            view_inv = torch.inverse(world_view_transform.float()).to(orig_type)
            self.camera_center = view_inv[3][:3]

    @property
    def fov_x(self) -> float:
        if self.K is None: return 0.
        length = max(self.image_width, self.image_height)
        return focal2fov(self.K[0, 0], length)

    @property
    def fov_y(self) -> float:
        if self.K is None: return 0.
        length = max(self.image_width, self.image_height)
        return focal2fov(self.K[1, 1], length)

    @property
    def full_fov_x(self) -> float:
        # The image we use are all fx = fy, but due to cropping fov_x and fov_y are different.
        # So we use the maximum of fov_x and fov_y as the fov
        if self.full_K is None: return self.fov_x
        length = max(self.full_image_width, self.full_image_height)
        return focal2fov(self.full_K[0, 0], length)

    @property
    def full_fov_y(self) -> float:
        # The image we use are all fx = fy, but due to cropping fov_x and fov_y are different.
        # So we use the maximum of fov_x and fov_y as the fov
        if self.full_K is None: return self.fov_y
        length = max(self.full_image_width, self.full_image_height)
        return focal2fov(self.full_K[1, 1], length)

    @property
    def cx(self) -> Optional[float]:
        return self.K[0, 2] if self.K is not None else None

    @property
    def cy(self) -> Optional[float]:
        return self.K[1, 2] if self.K is not None else None

    @staticmethod
    def from_extrinsics_intrinsics(
        world2cam: torch.Tensor,    # [4,4]
        K: torch.Tensor,            # [3,3] intrinsics for this image
        full_K: Optional[torch.Tensor] = None,  # [3,3] intrinsics of original full image
        **kwargs                    # image_width, image_height, image, mask, etc.
    ) -> 'GSCamera':
        width = kwargs.get('image_width', 512)
        height = kwargs.get('image_height', 512)
        znear = kwargs.get('znear', 0.01)
        zfar = kwargs.get('zfar', 100)
        proj_matrix = perspective_matrix_from_K(K, width, height, znear, zfar)
        full_proj_transform = world2cam.T @ proj_matrix.to(world2cam).T
        return GSCamera(
            world_view_transform=world2cam.T.float(),
            full_proj_transform=full_proj_transform.float(),
            K=K, full_K=full_K, **kwargs)

    def __repr__(self) -> str:
        return f'GSCamera(K={self.K.tolist() if self.K is not None else None} world2cam={self.world_view_transform.shape if self.world_view_transform is not None else None} proj={self.full_proj_transform.shape if self.full_proj_transform is not None else None} device={self.world_view_transform.device if self.world_view_transform is not None else "None"} image_path={self.image_path} depth_path={self.depth_path} mask_path={self.mask_path}\n'

    def to(self, device: Union[str, torch.device]) -> 'GSCamera':
        """Move to device."""
        if self.image is not None:
            self.image = self.image.to(device)
        if self.world_view_transform is not None:
            self.world_view_transform = self.world_view_transform.to(device)
        if self.full_proj_transform is not None:
            self.full_proj_transform = self.full_proj_transform.to(device)
        if self.camera_center is not None:
            self.camera_center = self.camera_center.to(device)
        if self.K is not None:
            self.K = self.K.to(device)
        if self.full_K is not None:
            self.full_K = self.full_K.to(device)
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary of attributes."""
        attrs = ['image_path', 'mask_path', 'depth_path', 'normal_path',
                 'image', 'mask', 'depth', 'normal',
                 'image_height', 'image_width',
                 'K', 'full_K',
                 'full_image_width', 'full_image_height',
                 'camera_center',
                 'world_view_transform', 'full_proj_transform']
        return {f'_camdata_{k}': getattr(self, k) for k in attrs \
                if getattr(self, k) is not None}

    #@property
    #def quat(self):
    #    cam2world = torch.linalg.inv(self.world_view_transform.T)
    #    return matrix_to_quaternion(cam2world[:3, :3])

    @staticmethod
    def from_dict(dic: Dict[str, Any]) -> Union['GSCamera', List['GSCamera']]:
        """Load the attributes from a dictionary. Support a batch of cameras."""
        idx = len('_camdata_')
        
        # Check if we have an old-style dict
        if '_camdata_K' not in dic and '_camdata_fov_x' in dic:
            # Convert to new style in-place in a copy
            dic = dic.copy()
            fov_x = dic.pop('_camdata_fov_x')
            fov_y = dic.pop('_camdata_fov_y')
            # These might be single values or lists for batch
            H = dic.get('_camdata_image_height', 512)
            W = dic.get('_camdata_image_width', 512)
            cx = dic.pop('_camdata_cx', None)
            cy = dic.pop('_camdata_cy', None)
            
            is_batch = not isinstance(fov_x, float)
            
            def get_k(fx, fy, h, w, c_x, c_y):
                return intrinsics_from_fov(fx, fy, h, w, c_x, c_y)

            if is_batch:
                cx = cx if cx is not None else [None] * len(fov_x)
                cy = cy if cy is not None else [None] * len(fov_x)
                H = H if isinstance(H, list) else [H] * len(fov_x)
                W = W if isinstance(W, list) else [W] * len(fov_x)
                dic['_camdata_K'] = [get_k(*args) for args in zip(fov_x, fov_y, H, W, cx, cy)]
            else:
                dic['_camdata_K'] = get_k(fov_x, fov_y, H, W, cx, cy)

            if '_camdata_full_fov_x' in dic:
                full_fov_x = dic.pop('_camdata_full_fov_x')
                full_fov_y = dic.pop('_camdata_full_fov_y')
                if is_batch:
                    dic['_camdata_full_K'] = [get_k(*args) for args in zip(full_fov_x, full_fov_y, H, W, cx, cy)]
                else:
                    dic['_camdata_full_K'] = get_k(full_fov_x, full_fov_y, H, W, cx, cy)

        # Now, load from the (potentially modified) dict
        is_batch = isinstance(dic.get('_camdata_K'), list)
        
        if not is_batch:
            cam_data = {k[idx:]: v for k, v in dic.items() if k.startswith('_camdata_')}
            # Hack for nested lists in some saved formats
            for k, v in cam_data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                    cam_data[k] = v[0]
            return GSCamera(**cam_data)
        else:
            n_cams = len(dic['_camdata_K'])
            cams = []
            for i in range(n_cams):
                cam_data = {}
                for k, v in dic.items():
                    if k.startswith('_camdata_'):
                        key = k[idx:]
                        if isinstance(v, list) and len(v) == n_cams:
                            cam_data[key] = v[i]
                        else:
                            cam_data[key] = v
                cams.append(GSCamera(**cam_data))
            return cams

    def load_image(self) -> Optional[torch.Tensor]:
        """Deprecated. Load the image only when requested."""
        if self.image is not None:
            return self.image
        if self.image_path is None:
            return None
        image = np.array(Image.open(self.image_path))
        image = torch.from_numpy(image) / 255.0
        self.image = image.permute(2, 0, 1)
        self.image_width = self.image.shape[2]
        self.image_height = self.image.shape[1]
        return self.image

    def load_depth(self) -> Optional[torch.Tensor]:
        """Load the depth only when requested."""
        if self.depth_path is None:
            return None
        if self.depth is not None:
            return self.depth
        depth = np.array(Image.open(self.depth_path))
        self.depth = torch.from_numpy(depth).float()
        return self.depth

    def load_mask(self) -> Optional[torch.Tensor]:
        """Load the mask only when requested."""
        if self.mask_path is None:
            return None
        if self.mask is not None:
            return self.mask
        mask = np.array(Image.open(self.mask_path))
        self.mask = torch.from_numpy(mask).float()
        return self.mask

    @staticmethod
    def from_look_at(cam_pos: torch.Tensor, fov_x: float, fov_y: float, look_at: Optional[torch.Tensor] = None, **kwargs) -> 'GSCamera':
        """Creates a Camera object from a look-at point and camera position."""
        look_at = torch.zeros(3).to(cam_pos) if look_at is None else look_at
        world2cam = extrinsics_from_lookat(cam_pos, look_at)[0]
        #cam2world = kaolin2colmap_cam2world(torch.linalg.inv(world2cam))
        #world2cam = torch.linalg.inv(cam2world)
        return GSCamera.from_matrix(world2cam, fov_x, fov_y, **kwargs)

    def reset_full_proj_transform(self) -> None:
        """Reset the full projection transform."""
        proj_matrix = perspective_matrix_from_K(
            self.K, self.image_width, self.image_height, self.znear, self.zfar
        )
        self.full_proj_transform = self.world_view_transform @ proj_matrix.to(self.world_view_transform).T

    @staticmethod
    def from_matrix(
        world2cam: torch.Tensor,
        fov_x: float, fov_y: float,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        **kwargs
    ) -> 'GSCamera':
        """Creates a Camera object from world-to-camera matrix and intrinsics.
        Args:
            world2cam: [4, 4] world-to-camera matrix.
            fov_x: Field of view in x-axis (radians).
            fov_y: Field of view in y-axis (radians).
            cx: Principal point x in pixels. None means image center.
            cy: Principal point y in pixels. None means image center.
        """
        width = kwargs.get('image_width', 512)
        height = kwargs.get('image_height', 512)
        K = intrinsics_from_fov(fov_x, fov_y, height, width, cx, cy)
        return GSCamera.from_extrinsics_intrinsics(
            world2cam, K=K, full_K=None, **kwargs
        )


######################################
##### Gaussian Splatting Cameras #####
######################################



def to_homogenized(points: torch.Tensor) -> torch.Tensor:
    """Convert points to homogenized coordinates."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def from_homogenized(points: torch.Tensor) -> torch.Tensor:
    """Convert points from homogenized coordinates."""
    return points[..., :-1] / points[..., -1:]


def world2cam_projection(w2c: torch.Tensor, intrinsics: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Convert world to camera space and project to screen space.
    Args:
        w2c: torch.Tensor, [B, 4, 4], world to camera matrix. (Already transposed!)
        intrinsics: torch.Tensor, [B, 3, 3], intrinsics matrix.
        points: torch.Tensor, [B, N, 3], world space points.
    Returns:
        torch.Tensor, [B, N, 2], screen space points.
    """
    cam_points = from_homogenized(to_homogenized(points) @ w2c)
    cam_points = cam_points @ intrinsics.permute(0, 2, 1)
    return from_homogenized(cam_points)


def unproject(
    coordinates: torch.Tensor, # (..., 3), :2 are x, y, 2 is z
    intrinsics: torch.Tensor, # (..., 3, 3)
    extrinsics: torch.Tensor, # (..., 4, 4)
) -> torch.Tensor:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    z = coordinates[..., 2:].clone() # (L,)
    coordinates[..., 2] = 1
    ray_directions = einsum(
        intrinsics.inverse(), coordinates,
        "... i j, ... j -> ... i") # (..., 3)
    coordinates[..., 2:] = z
    # Apply the supplied depth values.
    points = ray_directions * z

    # Apply extrinsics transformation
    return einsum(extrinsics[..., :3, :3], points, '... i j, ... j -> ... i') \
            + extrinsics[..., :3, 3]


def perspective_matrix_from_K(
    K: torch.Tensor,        # [3,3]
    width: int, height: int,
    znear: float = 0.01, zfar: float = 100
) -> torch.Tensor:
    """Build 4x4 perspective projection matrix directly from intrinsics K.

    Equivalent to perspective_matrix_with_offset but takes K directly
    instead of fov/cx/cy/focal decomposition.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Compute asymmetric frustum bounds at znear
    left   = -(cx) / fx * znear
    right  = (width - cx) / fx * znear
    top    = -(cy) / fy * znear
    bottom = (height - cy) / fy * znear

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (bottom - top)  # note: bottom > top in pixel coords
    # [CHECK this] CLAUDE proposes this fix
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = -(bottom + top) / (bottom - top)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal: float, pixels: int) -> float:
    return 2 * math.atan(pixels / (2 * focal))


def intrinsics_to_fov(intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert intrinsics to FoV.
    Args:
        intrinsics: a tensor of shape [batch, 3, 3]
    Returns:
        FoVx, FoVy: a tensor of shape [batch], in radians.
    """
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    return 2 * torch.atan2(cx, fx), 2 * torch.atan2(cy, fy)


def intrinsics_from_fov(
        fov_x: float,
        fov_y: float,
        H: float = 1.0,
        W: float = 1.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None) -> torch.Tensor:
    """Get the camera intrinsics matrix from FoV.
    Notice that this transforms points into screen space [0, size]^2 rather than NDC.
    .----> x
    |
    |
    v y
    Args:
        fov_x: Field of View in x-axis (radians).
        fov_y: Field of View in y-axis (radians).
        cx: Principal point x in pixels. None means centered (H/2).
        cy: Principal point y in pixels. None means centered (W/2).
    """
    fx = 0.5 * H / math.tan(fov_y / 2)
    fy = 0.5 * W / math.tan(fov_x / 2)
    cx = cx if cx is not None else 0.5 * H
    cy = cy if cy is not None else 0.5 * W

    return torch.Tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])


def extrinsics_from_lookat(
    cam_pos: torch.Tensor,  # [3] or [B, 3]
    look_at: torch.Tensor,  # [3] or [B, 3]
    up: Optional[torch.Tensor] = None) -> torch.Tensor:  # [3] or [B, 3]
    """Create rotation and translation matrices from camera position and look-at point.
    
    Args:
    cam_pos: Camera position in world coordinates
    look_at: Point to look at in world coordinates 
    up: Up vector, defaults to [0,1,0]
    
    Returns:
    R: Rotation matrix [3,3] or [B,3,3]
    T: Translation vector [3] or [B,3]
    """

    up = torch.tensor([[0.0, 1.0, 0.0]], device=cam_pos.device) if up is None else up
    
    # Handle both batched and unbatched inputs
    if cam_pos.dim() == 1:
        cam_pos = cam_pos.unsqueeze(0)
        look_at = look_at.unsqueeze(0)
    if up.dim() == 1:
        up = up.unsqueeze(0)

    # Create camera coordinate system
    forward = look_at - cam_pos  # Forward direction
    forward = forward / torch.norm(forward.float(), dim=-1, keepdim=True).to(forward.dtype)
    
    # right = torch.cross(up, forward)
    right = torch.cross(up, forward, dim=-1)  # Right direction
    right = right / torch.norm(right.float(), dim=-1, keepdim=True).to(right.dtype)
    
    # up = torch.cross(forward, right)
    up = torch.cross(forward, right, dim=-1)  # Recompute up for orthogonality
    
    # Stack vectors to create rotation matrix
    # point transform: x' = Rx; coordinate transform: x' = R^T x
    # we are doing coordinate transform, so the rotation matrix is transposed
    R = torch.stack([right, up, forward], dim=-1).permute(0, 2, 1)
    
    # Translation is negative of rotated camera position
    T = -torch.einsum('...ij,...j->...i', R, cam_pos)
    
    E = torch.eye(4, device=cam_pos.device)[None].repeat(cam_pos.shape[0], 1, 1)
    E[..., :3, :3] = R
    E[..., :3, 3] = T
    return E


def angle2matrix(angles: torch.Tensor) -> torch.Tensor:
    """Convert angles to rotation matrices.
    Args:
        angles: (N, 3), [yaw, pitch, roll]
    Returns:
        R: (N, 3, 3), rotation matrices
    """
    return torch.from_numpy(Rotation.from_euler('yxz', angles.cpu().numpy()).as_matrix()).float().to(angles.device)


def get_lookat_camera(
        look_at: Optional[torch.Tensor] = None,
        azims: Optional[torch.Tensor] = None,
        elevs: Optional[torch.Tensor] = None,
        fov=0.2443,
        radius=2.7,
        H=512, W=512,
        device='cpu') -> List[GSCamera]:
    """
    Sample a camera pose looking at a point with a Gaussian or uniform distribution.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians
        elev: polar angle (angle from the y axis) in radians
        look_at: 3-vector, point to look at
        azim_std: standard deviation of azimuthal angle in radians
        elev_std: standard deviation of polar angle in radians
        fov: field of view in radians
        radius: distance from camera to look_at point
        n_sample: number of samples to return
        resolution: image resolution of the camera
        device: device to put output on
        noise: 'gaussian' or 'uniform'
    """

    if azims is None or elevs is None:
        raise ValueError("azims and elevs must be provided")
    
    if isinstance(azims, np.ndarray):
        azims = torch.from_numpy(azims).float()
        elevs = torch.from_numpy(elevs).float()
    look_at = torch.zeros(3).float() if look_at is None else look_at
    azims = azims.to(device)
    elevs = elevs.to(device)
    look_at = look_at.to(device)
    
    common_kwargs = dict(
        look_at=look_at,
        fov_x=fov, fov_y=fov,
        image_width=W, image_height=H)
    cam_positions = campos_from_angle(azims, elevs, radius).to(device)
    
    cams = [GSCamera.from_look_at(cam_pos=x, **common_kwargs)
            for x in cam_positions]
    return cams
