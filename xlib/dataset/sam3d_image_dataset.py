import os
import cv2
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torchvision.transforms import ToTensor
from tqdm import tqdm

from xlib.dataset.common import (
    apply_affine_to_intrinsics,
    bbox_from_binary_mask,
    binary_mask_from_pil,
    glob_images,
)
from xlib.utils import imread_pil
from xlib.model.sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from xlib.model.sam_3d_body.data.utils.prepare_batch import prepare_batch


MAX_CACHE_ITEMS = 1000
BG_COLOR = (0.95, 0.94, 0.96)
BG_COLOR_UINT8 = tuple(int(round(c * 255)) for c in BG_COLOR)
BG_COLOR_TENSOR = torch.tensor(BG_COLOR, dtype=torch.float32)


class GetBorderValue:
    """Set fixed background color for affine padding."""

    def __call__(self, results: Dict) -> Dict:
        results["border_value"] = BG_COLOR_UINT8
        return results


class SAM3DImageDataset(torch.utils.data.Dataset):
    """Image dataset using SAM-3D-Body prepare_batch + estimator-style transform."""

    POSE_ZERO_SPECS = {
        "pred_pose_raw": (266,),
        "shape": (45,),
        "scale": (28,),
        "hand": (108,),
        "face": (72,),
        "mhr_model_params": (204,),
        "pred_cam_t": (3,),
        "pred_cam": (3,),
        "pred_keypoints_2d": (70, 2),
        "pred_keypoints_3d": (70, 3),
        "pred_joint_coords": (127, 3),
    }
    ALL_CACHE_KEYS = [
        "K_proc",
        "K_raw",
        "bbox_center",
        "bbox_scale",
        "img_size",
        "ori_img_size",
        "affine_trans",
        "mask_score",
        "person_valid",
        "bg_color",
    ]
    STACK_FIELDS = (
        "images",
        "hr_images",
        "masks",
        "bbox_center",
        "bbox_scale",
        "img_size",
        "ori_img_size",
        "affine_trans",
        "mask_score",
        "person_valid",
        "K_proc",
        "K_raw",
        "bg_color",
    )
    TENSOR_FIELDS = ("raw_sizes", "crop_bboxes")

    def __init__(
        self,
        root_dir: str,
        hierarchy_file: str,
        image_folder: str = "input_images",
        mask_folder: str = "input_masks",
        sup_image_folder: str = "images",
        sup_mask_folder: str = "masks",
        num_input_views: int = 3,
        num_sup_views: int = 0,
        input_size: Tuple[int, int] = (512, 512),
        hr_image_size: Optional[Tuple[int, int]] = None,
        allow_cross_pose: bool = True,
        default_fov: float = 60.0,
        split: Optional[str] = None,
        use_sam3db_cache: bool = True,

        # deterministic image list mode
        image_list: Optional[str] = None,

        # single subject mode
        subject_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        max_poses_per_subject: Optional[int] = None,
        subject_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__()


        self.root_dir = root_dir
        self.split = split
        self.hr_image_size = hr_image_size
        self.hierarchy_file = (
            os.path.join(root_dir, hierarchy_file)
            if split is None
            else os.path.join(root_dir, f"{split}_hierarchy.json")
        )
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.sup_image_folder = sup_image_folder
        self.sup_mask_folder = sup_mask_folder
        self.num_input_views = int(num_input_views)
        self.num_sup_views = int(num_sup_views)
        self.n_views = self.num_input_views + self.num_sup_views
        self.allow_cross_pose = allow_cross_pose
        self.default_fov_rad = float(np.deg2rad(default_fov))

        self.hierarchy: Dict[str, Dict[str, List[str]]] = {}
        self.hierarchy_inverse: Dict[str, Tuple[str, str]] = {}
        self.scene_cache: Dict[str, Optional[Dict[str, Any]]] = {} # scene.pth cache
        self._view_cache: Dict[Tuple[str, int], Dict[str, Any]] = {} # view cache
        self.use_sam3db_cache = use_sam3db_cache # sam3db cache
        self.sam3db_cache: Dict[str, Optional[Dict]] = {} # sam3db cache

        self._load_hierarchy()
        self.max_poses_per_subject = max_poses_per_subject
        if subject_id is not None:
            assert cluster_id is not None, "cluster_id is required when subject_id is set"
            self._filter_to_subject(subject_id, cluster_id, subject_sequences)

        self._path_index: Dict[str, Dict[str, List[str]]] = {}
        self._ensure_path_index()

        self._image_list: Optional[Dict] = None
        self._image_list_uses_paths = False
        self._image_list_root = self.root_dir

        self._use_view_cache = subject_id is not None

        if len(self.hierarchy_inverse) == 0:
            if subject_id is not None:
                msg = f"Subject '{subject_id}' cluster '{cluster_id}'"
                if subject_sequences:
                    msg += f" with sequences {subject_sequences}"
                msg += f" not found in {self.hierarchy_file}"
                logging.warning(msg)
            elif subject_names is not None:
                logging.warning(
                    "None of the requested subject_names were found in %s",
                    self.hierarchy_file,
                )
            
            raise ValueError(f"No valid subjects found in {self.hierarchy_file}")
        self.sam3d_image_size = input_size

        self.transform = Compose(
            [
                GetBBoxCenterScale(padding=1.1),
                GetBorderValue(),
                TopdownAffine(input_size=self.sam3d_image_size, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

        if image_list is not None:
            if isinstance(image_list, str):
                self.load_image_list(image_list)
            else:
                # Handle list or ListConfig
                self.set_image_list(list(image_list))

    def _sample_multi_input_views(self, pose_name: str) -> List[Dict]:
        """Sample input and supervision views for one pose.

        Input views: first input image of pose_name is always the anchor; remaining
        num_input_views-1 are sampled randomly from all input images across all cluster
        poses such that each chosen view has a distinct cam_idx (no duplicate camera
        indices in the input set). num_input_views==-1 returns every input image.

        Sup views: sampled randomly (no replacement) from all images (input+sup folders)
        across all cluster poses. num_sup_views==-1 returns every such image.
        """
        subject_id, cluster_id = self.hierarchy_inverse[pose_name]
        all_cluster_poses = (
            self.hierarchy[subject_id][cluster_id] if self.allow_cross_pose else [pose_name]
        )

        # Collect all input images across all poses, sorted per pose by filename
        all_input_views: List[Dict] = []
        for pn in all_cluster_poses:
            views = [
                {"pose_name": pn, "cam_idx": i, "filename": os.path.basename(rel)}
                for i, rel in enumerate(self._path_index[pn]["ii"])
            ]
            views.sort(key=lambda v: v["filename"])
            all_input_views.extend(views)

        if self.num_input_views == -1:
            input_views = list(all_input_views)
        else:
            # Anchor: first input image from current pose
            anchor_idx = next(
                (i for i, v in enumerate(all_input_views) if v["pose_name"] == pose_name), 0
            )
            used_indices = {anchor_idx}
            used_cam_idx = {all_input_views[anchor_idx]["cam_idx"]}
            input_views = [all_input_views[anchor_idx]]
            remaining = max(0, self.num_input_views - 1)
            for _ in range(remaining):
                candidate_indices = [
                    i
                    for i, v in enumerate(all_input_views)
                    if i not in used_indices and v["cam_idx"] not in used_cam_idx
                ]
                if not candidate_indices:
                    logging.warning(
                        "No more input views with unused cam_idx for %s "
                        "(wanted %d input views, got %d); distinct cams exhausted",
                        pose_name,
                        self.num_input_views,
                        len(input_views),
                    )
                    break
                pick = int(np.random.choice(candidate_indices))
                used_indices.add(pick)
                used_cam_idx.add(all_input_views[pick]["cam_idx"])
                input_views.append(all_input_views[pick])

        # Sup views: all images (input + sup folders) from all cluster poses,
        # excluding any (pose_name, cam_idx) already selected as an input view.
        selected_input_keys = {(v["pose_name"], v["cam_idx"]) for v in input_views}
        all_sup_views: List[Dict] = []
        for pn in all_cluster_poses:
            entry = self._path_index[pn]
            n_cams = len(entry["ii"]) + len(entry["i"])
            for cam_idx in range(n_cams):
                if (pn, cam_idx) not in selected_input_keys:
                    all_sup_views.append({"pose_name": pn, "cam_idx": cam_idx})

        if self.num_sup_views == -1:
            sup_views = list(all_sup_views)
        else:
            replace = False 
            if len(all_sup_views) < self.num_sup_views:
                logging.warning(f"Only {len(all_sup_views)} sup views available for {pose_name}, using all")
                replace = True
            idxs = np.random.choice(len(all_sup_views), self.num_sup_views, replace=replace)
            sup_views = [all_sup_views[i] for i in idxs]

        return input_views + sup_views

    def __len__(self) -> int:
        if self._image_list is not None:
            return len(self._image_list)
        return len(self.hierarchy_inverse)

    def _load_hierarchy(self) -> None:
        if not os.path.exists(self.hierarchy_file):
            raise FileNotFoundError(f"Hierarchy file not found: {self.hierarchy_file}")

        with open(self.hierarchy_file, "r") as f:
            self.hierarchy = json.load(f)

        self.hierarchy_inverse = {}
        for subject_id, clusters in self.hierarchy.items():
            for cluster_id, pose_names in clusters.items():
                for pose_name in pose_names:
                    self.hierarchy_inverse[pose_name] = (subject_id, cluster_id)

    def _filter_to_subject(
        self,
        subject_id: str,
        cluster_id: str,
        subject_sequences: Optional[List[str]] = None
    ) -> None:
        """Get only the poses belonging to the given subject_id and cluster_id."""
        assert subject_id in self.hierarchy, f"Subject {subject_id} not found in hierarchy"
        assert cluster_id in self.hierarchy[subject_id], f"Cluster {cluster_id} not found in subject {subject_id}"
        poses = self.hierarchy[subject_id][cluster_id]

        if subject_sequences is not None:
            poses = [p for p in poses if p in subject_sequences]

        if self.max_poses_per_subject is not None and self.max_poses_per_subject >= 0:
            poses = poses[: self.max_poses_per_subject]
        self.hierarchy = {subject_id: {cluster_id: poses}}
        self.hierarchy_inverse = {pose: (subject_id, cluster_id) for pose in poses}

    def _get_index_path(self) -> str:
        return os.path.join(self.root_dir, "path_index.npz")

    def _build_pose_index_entries(self, pose_dir: str) -> Dict[str, List[str]]:
        input_imgs = glob_images(os.path.join(pose_dir, self.image_folder))
        mask_dir = os.path.join(pose_dir, self.mask_folder)
        ii = []
        for p in sorted(input_imgs):
            rel = os.path.relpath(p, os.path.join(pose_dir, self.image_folder))
            msk = os.path.join(mask_dir, rel.replace(".jpg", ".png"))
            if os.path.exists(msk):
                ii.append(rel)
        sup_dir = os.path.join(pose_dir, self.sup_image_folder)
        if os.path.isdir(sup_dir):
            sup_mask_dir = os.path.join(pose_dir, self.sup_mask_folder)
            i = []
            for p in sorted(glob_images(sup_dir)):
                rel = os.path.relpath(p, sup_dir)
                msk = os.path.join(sup_mask_dir, rel.replace(".jpg", ".png"))
                if os.path.exists(msk):
                    i.append(rel)
        else:
            i = []
        return {"ii": ii, "i": i}

    def _path_index_entry_exists(self, pose_dir: str, entry: Dict[str, List[str]]) -> bool:
        for rel in entry["ii"]:
            img_path = os.path.join(pose_dir, self.image_folder, rel)
            msk_path = os.path.join(pose_dir, self.mask_folder, rel.replace(".jpg", ".png"))
            if not os.path.exists(img_path) or not os.path.exists(msk_path):
                return False
        for rel in entry["i"]:
            img_path = os.path.join(pose_dir, self.sup_image_folder, rel)
            msk_path = os.path.join(pose_dir, self.sup_mask_folder, rel.replace(".jpg", ".png"))
            if not os.path.exists(img_path) or not os.path.exists(msk_path):
                return False
        return True

    def _ensure_path_index(self) -> None:
        """Ensure the path index is built and loaded."""

        logging.info(f"Preparing path index for {len(self.hierarchy_inverse)} poses")
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            data = np.load(index_path)
            keys = data["keys"].tolist()
            ii_flat = data["ii_flat"].tolist()
            i_flat = data["i_flat"].tolist()
            ii_offsets = data["ii_offsets"].tolist()
            i_offsets = data["i_offsets"].tolist()
            self._path_index = {
                k: {
                    "ii": ii_flat[ii_offsets[j]:ii_offsets[j + 1]],
                    "i": i_flat[i_offsets[j]:i_offsets[j + 1]],
                }
                for j, k in enumerate(keys)
            }
        else:
            self._path_index = {}

        # Rebuild entries that are missing or have stale file references.
        new_entries = False
        for pose_name in tqdm(list(self.hierarchy_inverse.keys())):
            pose_dir = os.path.join(self.root_dir, pose_name)
            if pose_name not in self._path_index:
                self._path_index[pose_name] = self._build_pose_index_entries(pose_dir)
                new_entries = True

        if new_entries:
            keys = list(self._path_index.keys())
            ii_flat, i_flat, ii_offsets, i_offsets = [], [], [0], [0]
            for k in keys:
                ii_flat.extend(self._path_index[k]["ii"])
                i_flat.extend(self._path_index[k]["i"])
                ii_offsets.append(len(ii_flat))
                i_offsets.append(len(i_flat))
            np.savez_compressed(
                index_path,
                keys=np.array(keys),
                ii_flat=np.array(ii_flat),
                ii_offsets=np.array(ii_offsets),
                i_flat=np.array(i_flat),
                i_offsets=np.array(i_offsets),
            )

    def _get_view_paths(self, pose_dir: str, cam_idx: int) -> Tuple[str, str]:
        pose_name = os.path.relpath(pose_dir, self.root_dir)
        entry = self._path_index[pose_name]
        ii_rels = entry["ii"]
        if cam_idx < len(ii_rels):
            rel = ii_rels[cam_idx]
            return (
                os.path.join(pose_dir, self.image_folder, rel),
                os.path.join(pose_dir, self.mask_folder, rel.replace('.jpg', '.png')),
            )
        rel = entry["i"][cam_idx - len(ii_rels)]
        return (
            os.path.join(pose_dir, self.sup_image_folder, rel),
            os.path.join(pose_dir, self.sup_mask_folder, rel.replace('.jpg', '.png')),
        )

    def _load_scene(self, pose_dir: str) -> Optional[Dict[str, Any]]:
        """Load scene.pth from a pose directory. Returns None on missing/corrupt file."""
        if pose_dir in self.scene_cache: # cache hit
            return self.scene_cache[pose_dir]

        if len(self.scene_cache) > MAX_CACHE_ITEMS:
            self.scene_cache = {}

        scene_path = os.path.join(pose_dir, "scene.pth")
        if not os.path.exists(scene_path):
            self.scene_cache[pose_dir] = None
            return None
        scene_data = torch.load(scene_path, map_location="cpu", weights_only=True)
        self.scene_cache[pose_dir] = scene_data
        return scene_data

    def _load_intrinsics_from_scene(
        self, pose_dir: str, cam_idx: int, img_h: int, img_w: int
    ) -> np.ndarray:
        """Load or build raw intrinsics [3, 3] from scene data."""
        scene_data = self._load_scene(pose_dir)

        if scene_data is not None:
            if "input_K" in scene_data and "sup_K" in scene_data:
                input_K = scene_data["input_K"]
                sup_K = scene_data["sup_K"]
                if cam_idx < len(input_K):
                    return np.array(input_K[cam_idx], dtype=np.float64)
                else:
                    sup_idx = cam_idx - len(input_K)
                    assert 0 <= sup_idx < len(sup_K), (
                        f"Camera index {cam_idx} out of range for supervision "
                        f"sequence with length {len(sup_K)}"
                    )
                    return np.array(sup_K[sup_idx], dtype=np.float64)

            if "input_fov_x" in scene_data and "input_fov_y" in scene_data:
                input_fov_x = scene_data["input_fov_x"]
                input_fov_y = scene_data["input_fov_y"]
                fov_x = fov_y = self.default_fov_rad
                cx, cy = img_w / 2, img_h / 2
                if cam_idx < len(input_fov_x):
                    fov_x = float(input_fov_x[cam_idx])
                    fov_y = float(input_fov_y[cam_idx])
                    if "input_cx" in scene_data and "input_cy" in scene_data:
                        cx = float(scene_data["input_cx"][cam_idx])
                        cy = float(scene_data["input_cy"][cam_idx])
                elif "sup_fov_x" in scene_data and "sup_fov_y" in scene_data:
                    sup_fov_x = scene_data["sup_fov_x"]
                    sup_fov_y = scene_data["sup_fov_y"]
                    sup_idx = cam_idx - len(input_fov_x)
                    assert 0 <= sup_idx < len(sup_fov_x), (
                        f"sup_idx={sup_idx} < 0 for cam_idx={cam_idx} "
                        f"and len(input_fov_x)={len(input_fov_x)}"
                    )
                    fov_x = float(sup_fov_x[sup_idx])
                    fov_y = float(sup_fov_y[sup_idx])
                    if "sup_cx" in scene_data and "sup_cy" in scene_data:
                        cx = float(scene_data["sup_cx"][sup_idx])
                        cy = float(scene_data["sup_cy"][sup_idx])
                fx = img_w / (2 * np.tan(fov_x / 2))
                fy = img_h / (2 * np.tan(fov_y / 2))
                return np.array(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
                )

        # Total fallback: default FoV and centered principal point
        fx = img_w / (2 * np.tan(self.default_fov_rad / 2))
        fy = img_h / (2 * np.tan(self.default_fov_rad / 2))
        cx, cy = img_w / 2, img_h / 2
        return np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

    @staticmethod
    def _get_raw_input_pose_view(view_data: Any) -> Optional[Dict[str, torch.Tensor]]:
        """Extract raw sam3db per-view record.
        
        Keeps all raw keys as-is (no rename, no squeeze/unsqueeze).
        Casts to float, stays on CPU.
        """
        if not isinstance(view_data, dict):
            return None

        # Required keys for reconstruction/animation
        required_keys = list(SAM3DImageDataset.POSE_ZERO_SPECS.keys())
        # Optional keys
        optional_keys = [
            "pred_keypoints_2d", "pred_keypoints_3d", "pred_joint_coords",
            "cam_rot", "cam_trans",
        ]

        for k in required_keys:
            if k not in view_data:
                return None

        out = {}
        for k in required_keys + optional_keys:
            if k in view_data:
                v = view_data[k]
                if isinstance(v, torch.Tensor):
                    v = v.float()
                    # Normalize to per-view shape [1, ...] so view-wise torch.cat() is correct.
                    # Cached sam3db entries are often stored as [...], e.g. [204], [45], [70, 2].
                    if v.ndim == 0:
                        v = v.unsqueeze(0)
                    elif v.shape[0] != 1:
                        v = v.unsqueeze(0)
                    out[k] = v
                else:
                    out[k] = v

        # Flag whether root rotation was already factorized out of mhr_model_params
        if "cam_rot" in out and "cam_trans" in out:
            out["is_factorized"] = torch.tensor([1.0])
        else:
            out["cam_rot"] = torch.zeros(1, 3, 3)
            out["cam_trans"] = torch.zeros(1, 3)
            out["is_factorized"] = torch.tensor([0.0])
        return out

    @staticmethod
    def _image_cache_key(img_path: str) -> str:
        return os.path.join(
            os.path.basename(os.path.dirname(img_path)),
            os.path.basename(img_path),
        )

    @staticmethod
    def _get_cached_view_entry(
        sam3db_data: Optional[Dict], img_key: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        if sam3db_data is None or img_key not in sam3db_data:
            return None
        return sam3db_data[img_key]
    
    @staticmethod
    def _safe_load(img_path: str, msk_path: str) -> Any:
        try:
            return imread_pil(img_path), imread_pil(msk_path)
        except Exception as e:
            logging.error("Error in _safe_load: %s, when loading image %s or mask %s", e, img_path, msk_path)
            raise

    def _load_sam3db(self, pose_dir: str) -> Optional[Dict]:
        if not self.use_sam3db_cache:
            return None
        if pose_dir not in self.sam3db_cache:
            sam3db_path = os.path.join(pose_dir, "sam3db.pth")
            if not os.path.exists(sam3db_path):
                #print(f"No sam3db.pth at {sam3db_path}")
                return None
            if len(self.sam3db_cache) > MAX_CACHE_ITEMS:
                self.sam3db_cache = {}
            self.sam3db_cache[pose_dir] = torch.load(sam3db_path, map_location="cpu", weights_only=True)
        return self.sam3db_cache[pose_dir]

    def _process_view(self, pose_dir: str, cam_idx: int) -> Dict[str, Any]:
        """Process a single view and return all derived tensors."""
        cache_key = (pose_dir, cam_idx)
        if self._use_view_cache and cache_key in self._view_cache:
            return self._view_cache[cache_key]

        img_path, msk_path = self._get_view_paths(pose_dir, cam_idx)

        # Check for cached non-image metadata in sam3db
        sam3db_data = self._load_sam3db(pose_dir)
        img_key = self._image_cache_key(img_path)
        cached_view = self._get_cached_view_entry(sam3db_data, img_key)

        if cached_view is not None:
            try:
                results = self._process_view_from_cache(
                    img_path, msk_path, cached_view, img_key, sam3db_data
                )
            except Exception as e:
                logging.error("Error processing cached view %s: %s", img_path, e)
                raise
        else:
            results = self._process_view_full(
                pose_dir, cam_idx, img_path, msk_path, img_key, sam3db_data
            )

        # Additive multiview pose exposure if present in sam3db.pth
        if sam3db_data is not None and "multiview" in sam3db_data:
            mv = sam3db_data["multiview"]
            if img_key in mv.get("per_view", {}):
                results["multiview_pose"] = {
                    "body": mv["body"],
                    "camera": mv["per_view"][img_key],
                    "meta": mv.get("meta", {}),
                }

        if self._use_view_cache:
            if len(self._view_cache) > MAX_CACHE_ITEMS:
                self._view_cache = {}
            self._view_cache[cache_key] = results
        return results

    def _process_view_from_cache(
        self,
        img_path: str,
        msk_path: str,
        cached: Dict[str, torch.Tensor],
        img_key: str,
        sam3db_data: Dict
    ) -> Dict[str, Any]:
        """Fast path: load image, apply cached affine warp, skip prepare_batch."""
        img_pil, msk_pil = self._safe_load(img_path, msk_path)
        img_np = np.array(img_pil)

        # Apply cached affine transform to image
        aff_np = cached["affine_trans"].numpy().astype(np.float64)
        h_out, w_out = self.sam3d_image_size
        border_value = BG_COLOR_UINT8
        img_warped = cv2.warpAffine(
            img_np, aff_np, (w_out, h_out),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        img_t = torch.from_numpy(img_warped).permute(2, 0, 1).float() / 255.0

        # Load mask and warp it too
        msk_np = binary_mask_from_pil(msk_pil)
        msk_warped = cv2.warpAffine(
            msk_np[..., 0], aff_np, (w_out, h_out),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        msk_t = torch.from_numpy(msk_warped).float().unsqueeze(0) / 255.0  # [1, H, W]

        # Pose
        input_pose = self._get_raw_input_pose_view(sam3db_data[img_key])

        results = {
            "images": img_t,
            "masks": msk_t,
            "image_paths": img_path,
            "input_pose": input_pose,
            "raw_images": img_pil,
            "raw_masks": msk_pil,
        }

        if self.hr_image_size is not None:
            hr_h, hr_w = self.hr_image_size
            std_h, std_w = self.sam3d_image_size
            S = np.array([[hr_w / std_w, 0, 0], [0, hr_h / std_h, 0], [0, 0, 1]])
            hr_affine = S @ np.vstack([aff_np, [0, 0, 1]])
            hr_affine = hr_affine[:2]
            hr_warped = cv2.warpAffine(
                img_np, hr_affine, (hr_w, hr_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
            hr_img_t = torch.from_numpy(hr_warped).permute(2, 0, 1).float() / 255.0
            results["hr_images"] = hr_img_t

        for key in self.ALL_CACHE_KEYS:
            results[key] = cached[key]
        results["bg_color"] = BG_COLOR_TENSOR
        results.update({
            "raw_sizes": cached["raw_size"].float(),
            "crop_bboxes": cached["crop_bbox"].float(),
        })
        return results

    def _process_view_full(
        self,
        pose_dir: str,
        cam_idx: int,
        img_path: str,
        msk_path: str,
        img_key: str,
        sam3db_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Existing full pipeline for cache misses."""
        img_pil, msk_pil = self._safe_load(img_path, msk_path)

        img_np = np.array(img_pil)
        msk_np = binary_mask_from_pil(msk_pil)
        bbox_xyxy = bbox_from_binary_mask(msk_np)
        x_min, y_min, x_max, y_max = bbox_xyxy
        assert x_max > x_min and y_max > y_min, f"Invalid bbox {bbox_xyxy} for {img_path}"

        K_raw = self._load_intrinsics_from_scene(pose_dir, cam_idx, img_pil.height, img_pil.width)
        cam_int = torch.from_numpy(K_raw).float().unsqueeze(0)

        prepared = prepare_batch(
            img=img_np,
            transform=self.transform,
            boxes=np.array([bbox_xyxy], dtype=np.float32),
            masks=np.expand_dims(msk_np, axis=0),
            masks_score=None,
            cam_int=cam_int,
        )

        img_t = prepared["img"][0, 0].float()
        msk_t = (prepared["mask"][0, 0] / 255.).float()
        aff_t = prepared["affine_trans"][0, 0].float()
        K_proc = apply_affine_to_intrinsics(K_raw, aff_t)
        K_raw_t = torch.from_numpy(K_raw).float()

        input_pose = None
        if sam3db_data is not None and img_key in sam3db_data and self.use_sam3db_cache:
            input_pose = self._get_raw_input_pose_view(sam3db_data[img_key])

        results = {
            "images": img_t,
            "masks": msk_t,
            "K_proc": K_proc,
            "K_raw": K_raw_t,
            "bbox_center": prepared["bbox_center"][0, 0].float(),
            "bbox_scale": prepared["bbox_scale"][0, 0].float(),
            "img_size": prepared["img_size"][0, 0].float(),
            "ori_img_size": prepared["ori_img_size"][0, 0].float(),
            "affine_trans": aff_t,
            "mask_score": prepared["mask_score"][0, 0].float(),
            "person_valid": prepared["person_valid"][0, 0].float(),
            "raw_sizes": (img_pil.height, img_pil.width),
            "crop_bboxes": bbox_xyxy,
            "bg_color": BG_COLOR_TENSOR,
            "image_paths": img_path,
            "input_pose": input_pose,
            "raw_images": img_pil,
            "raw_masks": msk_pil,
        }

        if self.hr_image_size is not None:
            hr_h, hr_w = self.hr_image_size
            std_h, std_w = self.sam3d_image_size
            S = np.array([[hr_w / std_w, 0, 0], [0, hr_h / std_h, 0], [0, 0, 1]])
            hr_affine = S @ np.vstack([aff_t.numpy(), [0, 0, 1]])
            hr_affine = hr_affine[:2]
            border_value = BG_COLOR_UINT8
            hr_warped = cv2.warpAffine(
                img_np, hr_affine, (hr_w, hr_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
            hr_img_t = torch.from_numpy(hr_warped).permute(2, 0, 1).float() / 255.0
            results["hr_images"] = hr_img_t

        return results

    def set_image_list(self, image_list: Optional[Dict]) -> None:
        """Set or clear the deterministic image list (dict keyed by subject_name)."""
        self._image_list_uses_paths = False
        self._image_list_root = self.root_dir
        if image_list is None:
            self._image_list = None
            return
        if isinstance(image_list, dict) and "data_root" in image_list and "items" in image_list:
            self._image_list_root = image_list["data_root"]
            root_abs = os.path.realpath(self.root_dir)
            list_root_abs = os.path.realpath(self._image_list_root)
            if root_abs != list_root_abs:
                raise ValueError(
                    "Validation list data_root must match dataset root_dir: "
                    f"{self._image_list_root} != {self.root_dir}"
                )
            self._image_list = image_list["items"]
            self._image_list_uses_paths = True
            return
        self._image_list = image_list

    def load_image_list(self, path: str) -> None:
        """Load image list from YAML or JSON file."""
        if path.endswith((".yaml", ".yml")):
            with open(path, "r") as f:
                image_list = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                image_list = json.load(f)
        self.set_image_list(image_list)

    def _resolve_path_list_view(self, local_path: str) -> Tuple[str, int]:
        """Map a path-list item back to the pose directory and old cam_idx."""
        if not isinstance(local_path, str):
            raise TypeError(f"Path-list image entry must be a string, got {type(local_path)}")
        if os.path.isabs(local_path):
            raise ValueError(f"Path-list image entry must be local to data_root: {local_path}")

        raw_parts = local_path.split(os.sep)
        if ".." in raw_parts:
            raise ValueError(f"Invalid local image path in validation list: {local_path}")
        parts = os.path.normpath(local_path).split(os.sep)
        if parts == ["."] or ".." in parts:
            raise ValueError(f"Invalid local image path in validation list: {local_path}")

        candidates = []
        for folder_idx, folder in enumerate(parts):
            if folder_idx == 0 or folder_idx == len(parts) - 1:
                continue
            if folder not in {self.image_folder, self.sup_image_folder}:
                continue
            pose_name = os.path.join(*parts[:folder_idx])
            rel_path = os.path.join(*parts[folder_idx + 1:])
            if pose_name not in self._path_index:
                continue
            entry = self._path_index[pose_name]
            if folder == self.image_folder and rel_path in entry["ii"]:
                cam_idx = entry["ii"].index(rel_path)
                candidates.append((pose_name, cam_idx))
            if folder == self.sup_image_folder and rel_path in entry["i"]:
                cam_idx = len(entry["ii"]) + entry["i"].index(rel_path)
                candidates.append((pose_name, cam_idx))

        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely resolve validation-list image path {local_path} "
                f"under {self.root_dir}; found {len(candidates)} matches"
            )

        pose_name, cam_idx = candidates[0]
        return os.path.join(self.root_dir, pose_name), cam_idx

    def _get_sample(self, idx: int) -> Dict[str, Any]:
        if self._image_list is not None:
            sample_identifier = list(self._image_list.keys())[idx]
            entry = self._image_list[sample_identifier]
            all_input_images = entry.get("input_images", [])
            sup_views_all = entry.get("images", [])
            input_views = list(all_input_images[: self.num_input_views])
            sup_views = sup_views_all if self.num_sup_views < 0 else sup_views_all[: self.num_sup_views]
            sampled_views = input_views + sup_views
            n_input = len(input_views)
        else:
            sample_identifier, (subject_id, cluster_id) = list(self.hierarchy_inverse.items())[idx]
            sampled_views = self._sample_multi_input_views(sample_identifier)
            n_input = self.num_input_views
        n_actual_req = len(sampled_views)
        sample = {}

        for sampled_view in sampled_views:
            if self._image_list_uses_paths:
                pose_dir, cam_idx = self._resolve_path_list_view(sampled_view)
            else:
                pn = str(sampled_view["pose_name"])
                pose_dir = os.path.join(self.root_dir, pn)
                cam_idx = sampled_view["cam_idx"]

            view = self._process_view(pose_dir, cam_idx)

            bg_color = view["bg_color"]
            view["bg_color"] = bg_color[0] if bg_color.ndim == 2 else bg_color
            if len(sample) == 0:
                sample = {key: [] for key in view if key != "input_pose"}
                sample["input_pose_dic"] = []
                sample["sample_identifier"] = sample_identifier
                sample["input_view_mask"] = torch.arange(n_actual_req) < n_input
                sample["idx"] = idx

            for key in sample:
                if key in {"input_pose_dic", "sample_identifier", "input_view_mask", "idx", "multiview_pose"}:
                    continue
                sample[key].append(view[key])

            input_pose = dict(view["input_pose"]) if view["input_pose"] is not None else self._zero_input_pose()
            input_pose["has_pose"] = torch.tensor(view["input_pose"] is not None)
            sample["input_pose_dic"].append(input_pose)

        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a multi-view SAM3D sample."""
        if self._image_list is not None:
            sample_name = list(self._image_list.keys())[idx]
        else:
            sample_name = list(self.hierarchy_inverse.keys())[idx]
        retries = 0
        last_error = None
        while retries < 3:
            try:
                sample = self._get_sample(idx)
                break
            except OSError as e:
                last_error = e
                logging.error("Error in __getitem__: %s, idx=%d", e, idx)
                retries += 1
        if retries >= 3:
            raise RuntimeError(
                f"Failed to get sample for subject {sample_name} after 3 retries"
            ) from last_error

        sample["input_pose_dic"] = {
            key: torch.stack([view[key].squeeze(0) for view in sample["input_pose_dic"]], dim=0)
            for key in sample["input_pose_dic"][0]
        }
        for key in self.STACK_FIELDS:
            if key in sample:
                sample[key] = torch.stack(sample[key], dim=0)
        sample["masks"] = sample["masks"][:, :1]
        for key in self.TENSOR_FIELDS:
            sample[key] = torch.stack(
                [
                    value.float() if isinstance(value, torch.Tensor)
                    else torch.tensor(value, dtype=torch.float32)
                    for value in sample[key]
                ],
                dim=0,
            )
        return sample

    @classmethod
    def _zero_input_pose(cls) -> Dict[str, torch.Tensor]:
        d = {key: torch.zeros(shape) for key, shape in cls.POSE_ZERO_SPECS.items()}
        d["cam_rot"] = torch.zeros(3, 3)
        d["cam_trans"] = torch.zeros(3)
        d["is_factorized"] = torch.tensor(0.0)
        return d


if __name__ == "__main__":
    import argparse
    from xlib.utils import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/_shared/data_neuman.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    for ds_cfg in cfg.val.ds_list:
        dataset = SAM3DImageDataset(
            root_dir=ds_cfg.root_dir,
            hierarchy_file=ds_cfg.hierarchy_file,
            image_list=ds_cfg.image_list,
            input_size=ds_cfg.input_size,
            allow_cross_pose=ds_cfg.allow_cross_pose,
            default_fov=ds_cfg.default_fov,
        )
