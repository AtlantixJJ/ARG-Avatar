#!/usr/bin/env python
"""Run this project's own data pipeline over NeuMan frames to produce sam3db.pth + scene.pth.

This ports the approach of `generate_sam3d_poses.py` (the actual pipeline that produced
training/eval detections) rather than `scripts/inference/estimate_motion_mhr.py`'s. The two
differ in one important way: `estimate_motion_mhr.py` is a *demo* tool for a raw video with
no existing annotations, so it independently re-detects (ViTDet), re-segments (SAM2), and
re-estimates FOV (MoGe2) per frame. NeuMan already has masks/ (from `generate_neuman_masks.py`)
and a real per-scene camera, so this instead goes through `xlib.dataset.sam3d_image_dataset.
SAM3DImageDataset` directly -- the exact same class `configs/_shared/data_val_list.yaml` uses
for training and evaluation. For any frame not yet in `sam3db.pth`, that class derives the
crop bbox straight from the mask (`bbox_from_binary_mask`); camera intrinsics come from
`scene.pth` if present, else a `default_fov` guess (60 degrees, `--default_fov`) applied
independently to width and height (asymmetric fx != fy). This script always takes that
fallback path -- no `scene.pth` is seeded from Apple's COLMAP calibration, so every frame's
camera comes from `default_fov`. Only the model's own pose regression (`get_input_pose_dic`
-> `forward_pose_branch_full`) actually runs per frame; there is no independent
detector/segmentor/FOV-estimator involved.

    python scripts/download_evaluation/generate_neuman_poses.py --gpus 0,1,2,3

or one scene while iterating:

    python scripts/download_evaluation/generate_neuman_poses.py --subjects bike --gpus 0

Per scene, ``sam3db.pth`` gets one entry per referenced frame under ``images/<file>``,
plus a duplicate of each input frame's entry under ``input_images/<file>`` (matching
``SAM3DImageDataset``'s default ``image_folder="input_images"`` / ``sup_image_folder="images"``
split). ``scene.pth`` is then just those same frames' ``K_raw`` gathered in val-list order.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from scripts.download_evaluation.neuman_manifest import SUBJECTS
from scripts.inference.infer_argavatar import DEFAULT_CONFIG_PATH, parse_gpus, run_parallel
from xlib.dataset.collate import collate_with_pil_images
from xlib.dataset.sam3d_image_dataset import SAM3DImageDataset
from xlib.model.sam_3d_body.utils import recursive_to
from xlib.render.gaussian_splatting import set_gaussian_type
from xlib.train.train_helper import get_unwrapped_model, resolve_sam3d_cfg, setup_model_sam3d
from xlib.utils import load_config

DEFAULT_ROOT = Path("data/AnimatableHuman/Rendering/NeuMan")
DEFAULT_VAL_LIST = Path("expr/val_lists/Neuman_val_list.yaml")
DEFAULT_CHECKPOINT = "checkpoints/ARGAvatar-Final.pth"
DEFAULT_FOV = 60.0  # SAM3DImageDataset's fallback FOV, applied to every frame's camera.

POSE_KEEP_KEYS = [
    "pred_pose_raw", "shape", "scale", "hand", "face", "pred_cam",
    "pred_keypoints_2d", "pred_keypoints_3d", "pred_joint_coords",
    "mhr_model_params", "pred_cam_t", "cam_rot", "cam_trans",
]

# batch field -> saved sam3db entry field. Matches SAM3DImageDataset.ALL_CACHE_KEYS
# (read back verbatim) plus the two TENSOR_FIELDS it renames on the way in.
CACHE_FIELDS = {
    "K_proc": "K_proc", "K_raw": "K_raw", "bbox_center": "bbox_center",
    "bbox_scale": "bbox_scale", "img_size": "img_size", "ori_img_size": "ori_img_size",
    "affine_trans": "affine_trans", "mask_score": "mask_score", "person_valid": "person_valid",
    "bg_color": "bg_color", "crop_bboxes": "crop_bbox", "raw_sizes": "raw_size",
}


def load_val_list(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def subject_frames(val_list: dict) -> Dict[str, Tuple[List[str], List[str]]]:
    """Return {subject: (input_filenames, cross_filenames)} from the val list."""
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for key, entry in val_list["items"].items():
        subject = key.split("/")[0]
        input_fnames = [Path(r).name for r in entry.get("input_images", [])]
        cross_fnames = [Path(r).name for r in entry.get("images", [])]
        out[subject] = (input_fnames, cross_fnames)
    return out


def _image_key(batch: dict, v_idx: int, b_idx: int = 0) -> str:
    """"<folder>/<file>" for view v_idx, matching SAM3DImageDataset._image_cache_key."""
    # default_collate transposes a per-sample List[str] of length V into V lists of length B.
    img_path = batch["image_paths"][v_idx][b_idx]
    return "/".join(Path(img_path).parts[-2:])


def _extract_entry(batch: dict, pose_dic: dict, v_idx: int, b_idx: int = 0) -> dict:
    entry = {}
    for k in POSE_KEEP_KEYS:
        if k in pose_dic and isinstance(pose_dic[k], torch.Tensor):
            entry[k] = pose_dic[k][b_idx, v_idx].detach().cpu().float()
    for batch_key, entry_key in CACHE_FIELDS.items():
        entry[entry_key] = batch[batch_key][b_idx, v_idx].detach().cpu().float()
    return entry


def _pose_worker(rank: int, gpu_ids: list, subjects: list, by_subject: dict, root: str,
                  shard_dir: str, args) -> None:
    """Estimate pose for this rank's stripe of subjects. Writes one shard."""
    torch.set_grad_enabled(False)
    gpu_id = gpu_ids[rank] if gpu_ids else None
    device = torch.device(f"cuda:{gpu_id}") if gpu_id is not None else torch.device("cpu")
    if gpu_id is not None:
        torch.cuda.set_device(device)

    train_args = load_config(args.config)
    cfg = resolve_sam3d_cfg(train_args, args.config)
    set_gaussian_type(train_args.model.get("gaussian_type", "3d"))
    if args.checkpoint_path:
        train_args.model["load_checkpoint"] = args.checkpoint_path
    model = setup_model_sam3d(train_args, cfg, device)
    model.eval()
    unwrapped = get_unwrapped_model(model)

    n_workers = max(1, len(gpu_ids))
    my_subjects = subjects[rank::n_workers]
    shard = {}
    for subject in my_subjects:
        input_fnames, cross_fnames = by_subject[subject]
        wanted = input_fnames + cross_fnames
        if args.limit:
            wanted = wanted[: args.limit]

        # Every wanted frame goes under "images" (sup_image_folder) so every entry lands
        # under the "images/<file>" key -- the input/cross split is re-applied afterwards
        # by duplicating the input frames' entries under "input_images/<file>".
        image_list = {
            "data_root": root,
            "items": {f"{subject}/0": {"images": [f"{subject}/images/{f}" for f in wanted]}},
        }
        ds = SAM3DImageDataset(
            root_dir=root, hierarchy_file="val_hierarchy.json",
            input_size=[512, 512], allow_cross_pose=True, default_fov=args.default_fov,
            num_input_views=0, num_sup_views=-1, use_sam3db_cache=True,
        )
        # SAM3DImageDataset.__init__ mishandles a dict image_list (it only accepts a path
        # string or a bare list there); set_image_list() is the correct entry point for a
        # {"data_root", "items"} dict.
        ds.set_image_list(image_list)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                             collate_fn=collate_with_pil_images)
        for batch in loader:
            batch = recursive_to(batch, device)
            pose_dic = unwrapped.get_input_pose_dic(batch)
            pose_dic = unwrapped._ensure_factorized_input_pose_dic(dict(pose_dic))
            n_views = batch["images"].shape[1]
            for v_idx in range(n_views):
                key = _image_key(batch, v_idx)
                shard[(subject, key)] = _extract_entry(batch, pose_dic, v_idx)
        print(f"[pose rank {rank}] {subject}: {len(wanted)} frames done", flush=True)
    torch.save(shard, Path(shard_dir) / f"pose_{rank}.pth")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint_path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--val_list", type=Path, default=DEFAULT_VAL_LIST)
    parser.add_argument("--default_fov", type=float, default=DEFAULT_FOV,
                        help="FOV guess used for every frame's camera (see docstring)")
    parser.add_argument("--subjects", nargs="*", default=None, metavar="NAME",
                        help="Only these scenes (default: every scene in --val_list)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap frames processed per subject, for a quick test run (0 = no cap)")
    parser.add_argument("--gpus", type=str, default="all")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Don't recompute frames already present in an existing sam3db.pth")
    parser.add_argument("--force", action="store_true", help="Recompute even if already present")
    args = parser.parse_args()

    val_list = load_val_list(args.val_list)
    by_subject = subject_frames(val_list)
    subjects = args.subjects if args.subjects else [s for s in SUBJECTS if s in by_subject]

    existing_by_subject: Dict[str, dict] = {}
    todo_subjects: List[str] = []
    for subject in subjects:
        input_fnames, cross_fnames = by_subject[subject]
        sam3db_path = args.root / subject / "sam3db.pth"
        existing = {}
        if sam3db_path.is_file():
            existing = torch.load(sam3db_path, map_location="cpu", weights_only=False)
        existing_by_subject[subject] = existing

        wanted = input_fnames + cross_fnames
        if args.limit:
            wanted = wanted[: args.limit]
        already_done = args.skip_existing and not args.force and all(
            f"images/{f}" in existing for f in wanted
        )
        if not already_done:
            todo_subjects.append(subject)

    if not todo_subjects:
        print("Nothing to do -- every requested subject already has full sam3db coverage.")
    else:
        print(f"Estimating pose for {len(todo_subjects)} subject(s): {', '.join(todo_subjects)}")
        gpus = parse_gpus(args.gpus)
        shard_dir = Path("expr/_neuman_pose_shards")
        shard_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_parallel(_pose_worker, gpus, todo_subjects, by_subject, str(args.root),
                         str(shard_dir), args)
            n_workers = max(1, len(gpus))
            for rank in range(n_workers):
                shard_path = shard_dir / f"pose_{rank}.pth"
                if not shard_path.is_file():
                    continue
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                for (subject, key), entry in shard.items():
                    existing_by_subject[subject][key] = entry
        finally:
            shutil.rmtree(shard_dir, ignore_errors=True)

    for subject in subjects:
        input_fnames, cross_fnames = by_subject[subject]
        existing = existing_by_subject[subject]

        for fname in input_fnames:
            src_key = f"images/{fname}"
            if src_key in existing:
                existing[f"input_images/{fname}"] = copy.deepcopy(existing[src_key])

        missing = [f for f in input_fnames + cross_fnames if f"images/{f}" not in existing]
        if missing:
            print(f"{subject}: WARNING -- {len(missing)} frame(s) still missing a pose, "
                  f"e.g. {missing[0]}")

        sam3db_path = args.root / subject / "sam3db.pth"
        sam3db_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(existing, sam3db_path)
        print(f"{subject}: wrote {len(existing)} entries -> {sam3db_path}")

        try:
            input_K = [existing[f"input_images/{f}"]["K_raw"] for f in input_fnames]
            sup_K = [existing[f"images/{f}"]["K_raw"] for f in cross_fnames]
        except KeyError:
            print(f"{subject}: skipping scene.pth -- not every frame has a pose yet")
            continue
        scene_path = args.root / subject / "scene.pth"
        torch.save({"input_K": input_K, "sup_K": sup_K}, scene_path)
        print(f"{subject}: wrote scene.pth ({len(input_K)} input, {len(sup_K)} sup)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
