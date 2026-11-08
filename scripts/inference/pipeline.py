"""One reusable ARGAvatar pipeline object, shared by every inference entrypoint.

This module is a library, not an entrypoint -- it has no command line. The
entrypoints that use it are ``scripts/inference/infer_argavatar.py`` (the one inference
CLI), ``app/`` (the Gradio demo) and the two ``scripts/test/`` smoke tests.

``infer_argavatar.py`` also holds the low-level reconstruction, animation and rendering
helpers. This module wraps them in an object that holds the loaded model once and
exposes it as a few plain method calls, so the entrypoints do not each re-derive the
model/estimator setup and the ``[B=1, V, ...]`` batch construction.

Nothing here reimplements reconstruction. Every method delegates to the functions in
``infer_argavatar`` (and through them to ``xlib.train.train_helper``), so the demo path
and the batch path cannot drift apart::

    pipeline = ARGAvatarPipeline.build()
    avatar = pipeline.reconstruct(["front.png", "back.png"])
    pipeline.render_views(avatar, "out/")
    pipeline.render_orbit_video(avatar, "out/orbit.mp4")
    pipeline.render_motion(avatar, "motion/dance.pth", "out/")

``ARGAvatarPipeline.build()`` takes the same knobs the CLI does; ``PipelineOptions``
carries them, and is also what gets passed straight through to the ``infer_argavatar``
helpers that expect an argparse namespace.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")

from scripts.inference import canonical_view as canonical_view_mod
from scripts.inference.canonical_view import render_canonical_view
from scripts.inference.infer_argavatar import (
    _animate_motion_frame_gs,
    build_subject_batch,
    collect_subject_views,
    extract_avatar_data,
    fps_arg,
    get_estimator,
    get_hr_image_size,
    load_driving_motion,
    render_motion_frame_crop,
    render_motion_frame_original,
    render_motion_frame_window,
    resolve_checkpoint_path,
)
from evaluation.argavatar.eval_argavatar import (
    render_in_raw_space,
    _snapshot_render_entries,
    _write_vis_snapshots,
)
from xlib.model_input_utils import slice_batch
from xlib.render.gaussian_splatting import set_gaussian_type
from xlib.train.train_helper import (
    get_unwrapped_model,
    prepare_all_pose_outputs,
    render_posed_gs,
    resolve_sam3d_cfg,
    setup_model_sam3d,
)
from xlib.utils import load_config
from scripts.inference.motion_output import GaussianSequenceWriter, VideoChunkEncoder

DEFAULT_CONFIG_PATH = "configs/ARGAvatar-Final.yaml"


@dataclass
class PipelineOptions:
    """Everything the ``infer_argavatar`` helpers read off the CLI namespace.

    Field names match the ``infer_argavatar.py`` flags exactly -- instances of this class
    are handed to those functions in place of the argparse namespace.
    """

    config: str = DEFAULT_CONFIG_PATH
    checkpoint_path: str = ""
    min_views: int = 1
    max_views: int = 6
    max_render_height: int = 1024
    bbox_thr: float = 0.5

    detector_name: str = "vitdet"
    detector_path: str = ""
    segmentor_name: str = "sam2"
    segmentor_path: str = "thirdparty/sam2"
    fov_name: str = "moge2"
    fov_path: str = ""

    driving_format: str = "video"
    driving_fps: float = 30.0
    driving_max_frames: int = 0
    export_motion_3d: bool = False
    motion_chunk_frames: int = 15


@dataclass
class Avatar:
    """One reconstructed subject: canonical Gaussians plus what is needed to re-pose them.

    ``gs_tpose`` are the canonical (rest-pose) Gaussians. ``avatar_tokens`` drive the
    per-frame pose-dependent offset during animation; models built without a pose_dep
    branch simply animate without one. ``batch`` is retained because the raw-space and
    canonical renders need the input cameras.
    """

    name: str
    gs_tpose: dict
    avatar_tokens: Optional[torch.Tensor]
    offset_pose: Optional[torch.Tensor]
    batch: dict
    view_paths: list = field(default_factory=list)
    # The raw final forward pass, kept only so save_avatar() can call extract_avatar_data.
    last_pass: dict = field(default_factory=dict, repr=False)


class ARGAvatarPipeline:
    """A loaded ARGAvatar model plus its SAM-3D-Body front end, held across calls."""

    def __init__(self, model, estimator, options: PipelineOptions, hr_image_size, device):
        self.model = model
        self.estimator = estimator
        self.options = options
        self.hr_image_size = hr_image_size
        self.device = device

    # -- construction ---------------------------------------------------------

    @classmethod
    def build(cls, options: Optional[PipelineOptions] = None, device: Optional[torch.device] = None,
              **overrides) -> "ARGAvatarPipeline":
        """Load the model, checkpoint and SAM-3D-Body front end onto ``device``.

        This is the expensive call -- roughly a minute, and several GB of VRAM. Build one
        and reuse it; ``app.py`` keeps a single instance for the whole server lifetime.
        """
        options = options or PipelineOptions()
        for key, value in overrides.items():
            if not hasattr(options, key):
                raise TypeError(f"unknown pipeline option: {key!r}")
            setattr(options, key, value)

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.set_grad_enabled(False)

        train_args = load_config(options.config)
        cfg = resolve_sam3d_cfg(train_args, options.config)
        set_gaussian_type(train_args.model.get("gaussian_type", "3d"))

        options.checkpoint_path = resolve_checkpoint_path(options.checkpoint_path, train_args)
        train_args.model["load_checkpoint"] = options.checkpoint_path

        model = setup_model_sam3d(train_args, cfg, device)
        model.eval()
        estimator = get_estimator(model, cfg, options, device)
        return cls(model, estimator, options, get_hr_image_size(train_args), device)

    @property
    def unwrapped(self):
        return get_unwrapped_model(self.model)

    @property
    def subfix(self) -> str:
        """`<config stem>_<checkpoint stem>`, the suffix used in output filenames."""
        model_name = Path(self.options.config).stem
        checkpoint_name = Path(self.options.checkpoint_path).stem
        return f"{model_name}_{checkpoint_name}"

    # -- reconstruction -------------------------------------------------------

    @torch.inference_mode()
    def reconstruct(self, images: Sequence[str], name: str = "subject") -> Avatar:
        """Reconstruct one subject from one or more views of it.

        ``images`` are paths, one person per image; a single path is simply ``V=1``. Views
        with no detected person are dropped, and fewer than ``min_views`` usable views is
        an error (the batch entrypoint skips such subjects; an interactive caller wants to
        be told).
        """
        image_paths = [str(p) for p in images]
        if not image_paths:
            raise ValueError("reconstruct() needs at least one image")

        view_batches, raw_imgs, raw_masks, view_names = collect_subject_views(
            self.estimator, image_paths, self.device, self.hr_image_size, self.options
        )
        if len(view_batches) < self.options.min_views:
            raise RuntimeError(
                f"no person detected in enough views: got {len(view_batches)} usable of "
                f"{len(image_paths)} image(s), need >= {self.options.min_views}"
            )

        batch = build_subject_batch(view_batches, raw_imgs, raw_masks, view_names, self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            result = self.unwrapped.two_phase_model_forward(batch)
        last_pass = result["passes"][-1]
        avatar = Avatar(
            name=name,
            gs_tpose=last_pass["output_canonical_gs_dic"],
            avatar_tokens=last_pass.get("avatar_tokens"),
            offset_pose=last_pass.get("offset_pose"),
            batch=batch,
            view_paths=view_names,
            last_pass=last_pass,
        )
        return avatar

    # -- rendering ------------------------------------------------------------

    @torch.inference_mode()
    def render_views(self, avatar: Avatar, out_dir: str) -> list:
        """Re-render each input view from its own camera, into ``out_dir``."""
        os.makedirs(out_dir, exist_ok=True)
        render_entries = render_in_raw_space(
            self.model, avatar.batch, avatar.gs_tpose, avatar.offset_pose,
            self.device, self.options.max_render_height,
        )
        snapshots = _snapshot_render_entries(render_entries)
        # _write_vis_snapshots writes <out_dir>/<name>/render/ and .../render_human_input/
        _write_vis_snapshots(out_dir, avatar.name, snapshots)
        return sorted(str(p) for p in (Path(out_dir) / avatar.name / "render").glob("*.png"))

    def _rest_pose_dic(self, avatar: Avatar) -> dict:
        """The subject's shape at zero pose and zero expression -- the canonical A-pose."""
        input_pose_dic = self.unwrapped.get_input_pose_dic(avatar.batch)
        factorized = self.unwrapped._ensure_factorized_input_pose_dic(dict(input_pose_dic))
        all_pose_outputs = prepare_all_pose_outputs(avatar.batch, factorized)
        pose_dic = all_pose_outputs[0]["pose_dic"]
        return {
            "shape_params": pose_dic["shape_params"],
            "model_params": torch.zeros_like(pose_dic["model_params"]),
            "expr_params": torch.zeros_like(pose_dic["expr_params"]),
        }

    @torch.inference_mode()
    def render_canonical(self, avatar: Avatar, out_dir: str, canvas_size: int = 2048) -> list:
        """Write the eight rest-pose orbit stills (`canonical_render_a<angle>.png`)."""
        os.makedirs(out_dir, exist_ok=True)
        render_canonical_view(
            model=self.unwrapped,
            batch=avatar.batch,
            rest_pose_dic=self._rest_pose_dic(avatar),
            gs_tpose_b=slice_batch(avatar.gs_tpose, 0),
            subject_dir=out_dir,
            max_render_size=self.options.max_render_height,
            device=self.device,
            canonical_canvas_size=canvas_size,
        )
        return sorted(str(p) for p in Path(out_dir).glob("canonical_render_a*.png"))

    @torch.inference_mode()
    def render_orbit_video(self, avatar: Avatar, out_path: str, n_frames: int = 90,
                           canvas_size: int = 768, fps: float = 30.0) -> str:
        """Render a smooth 360-degree turntable of the rest-pose avatar to ``out_path``.

        The same camera construction as ``render_canonical`` -- ``canonical_view._orbit_camera``
        -- just sampled at ``n_frames`` angles instead of eight, and encoded to mp4.
        """
        out_path = str(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        intrinsics = torch.eye(3, device=self.device)
        intrinsics[0, 0] = intrinsics[1, 1] = canvas_size * 1.2
        intrinsics[0, 2] = intrinsics[1, 2] = canvas_size / 2.0

        lbs_cache = slice_batch(
            self.unwrapped.mhr_animatable_gs.precompute_lbs_cache([self._rest_pose_dic(avatar)]), 0
        )
        from xlib.model.mhr_animatable_gaussians import animate_gs

        gs_posed, _ = animate_gs(
            self.unwrapped.mhr_animatable_gs, slice_batch(avatar.gs_tpose, 0), None,
            lbs_cache=lbs_cache, pose_indices=[0], to_colmap=True, offset_pose=None,
        )
        center, depth = canonical_view_mod._canonical_bbox_frame(
            gs_posed, intrinsics, canvas_size, canvas_size
        )
        background = torch.ones(3, device=self.device, dtype=torch.float32)

        with tempfile.TemporaryDirectory(prefix="orbit_") as tmp_dir:
            for idx in range(n_frames):
                theta = 2.0 * math.pi * idx / n_frames
                camera, _ = canonical_view_mod._orbit_camera(
                    center, depth, theta, intrinsics, canvas_size, canvas_size
                )
                result = render_posed_gs(
                    gs_posed, b_idx=0, cam=camera, bg_color=background,
                )
                rgba = canonical_view_mod._render_result_to_rgba(result)
                Image.fromarray(rgba[..., :3], mode="RGB").save(Path(tmp_dir) / f"{idx + 1:06d}.png")
            _encode_mp4(tmp_dir, out_path, fps)
        return out_path

    @torch.inference_mode()
    def render_motion(self, avatar: Avatar, motion_path: str, out_dir: str,
                      max_frames: Optional[int] = None, export_3d: Optional[bool] = None) -> dict:
        """Drive the avatar through one MHR motion file.

        ``motion_path`` is a ``sam3db``-format dict as written by ``estimate_motion_mhr.py``.
        Returns ``{"crop": <path>, "original": <path>}`` -- the human-crop-space video (the
        space the model was supervised in) and one at each source frame's own resolution.
        With ``driving_format="images"`` these are folders of one PNG per frame instead.
        With 3D export enabled, ``sequence`` contains the posed PLY manifest path.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        limit = self.options.driving_max_frames if max_frames is None else max_frames
        motion_name, sam3db, frame_keys = load_driving_motion(motion_path, limit)

        sequence = self._motion_sequence(out_dir, export_3d)
        as_images = self.options.driving_format == "images"
        with tempfile.TemporaryDirectory(prefix="motion_crop_") as crop_tmp, \
                tempfile.TemporaryDirectory(prefix="motion_orig_") as orig_tmp:
            if as_images:
                # One PNG per frame, named after the motion's own frame keys.
                crop_dir = out_dir / motion_name / "crop"
                orig_dir = out_dir / motion_name / "original"
                crop_dir.mkdir(parents=True, exist_ok=True)
                orig_dir.mkdir(parents=True, exist_ok=True)
            else:
                crop_dir, orig_dir = Path(crop_tmp), Path(orig_tmp)
            for idx, frame_key in enumerate(frame_keys, start=1):
                entry = sam3db[frame_key]
                gs_posed = _animate_motion_frame_gs(
                    self.unwrapped, avatar.gs_tpose, avatar.avatar_tokens, entry, self.device
                )
                if sequence:
                    sequence.write(gs_posed, frame_key)
                frame_name = f"{Path(frame_key).stem}.png" if as_images else f"{idx:06d}.png"
                Image.fromarray(
                    render_motion_frame_crop(gs_posed, entry, self.device, self.options.max_render_height)
                ).save(crop_dir / frame_name)
                Image.fromarray(
                    render_motion_frame_original(gs_posed, entry, self.device)
                ).save(orig_dir / frame_name)

            if as_images:
                outputs = {"crop": str(crop_dir), "original": str(orig_dir)}
            else:
                outputs = {}
                for tmp_dir, label in ((crop_tmp, "crop"), (orig_tmp, "original")):
                    out_path = out_dir / f"{motion_name}_{label}.mp4"
                    _encode_mp4(tmp_dir, str(out_path), self.options.driving_fps)
                    outputs[label] = str(out_path)
        if sequence:
            outputs["sequence"] = sequence.finish()
        return outputs

    @torch.inference_mode()
    def stream_motion(self, avatar: Avatar, motion_path: str, out_dir: str,
                      max_frames: Optional[int] = None, export_3d: Optional[bool] = None):
        """Drive the avatar through one MHR motion file, yielding each frame as it renders.

        Only the source-resolution render (``render_motion``'s ``original`` output) is
        produced. Yields ``(index, total, frame_rgba, video_path)`` where ``frame_rgba`` is
        the just-rendered ``[H, W, 4]`` uint8 array and ``video_path`` is ``None`` until the
        final item, which repeats the last frame and carries the encoded mp4.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        limit = self.options.driving_max_frames if max_frames is None else max_frames
        motion_name, sam3db, frame_keys = load_driving_motion(motion_path, limit)
        total = len(frame_keys)

        sequence = self._motion_sequence(out_dir, export_3d)
        with tempfile.TemporaryDirectory(prefix="motion_orig_") as orig_tmp:
            frame = None
            for idx, frame_key in enumerate(frame_keys, start=1):
                entry = sam3db[frame_key]
                gs_posed = _animate_motion_frame_gs(
                    self.unwrapped, avatar.gs_tpose, avatar.avatar_tokens, entry, self.device
                )
                if sequence:
                    sequence.write(gs_posed, frame_key)
                frame = render_motion_frame_original(gs_posed, entry, self.device)
                Image.fromarray(frame).save(Path(orig_tmp) / f"{idx:06d}.png")
                yield idx, total, frame, None
            out_path = out_dir / f"{motion_name}_original.mp4"
            _encode_mp4(orig_tmp, str(out_path), self.options.driving_fps)
        if sequence:
            print(f"3D motion sequence: {sequence.finish()}", flush=True)
        yield total, total, frame, str(out_path)

    def _motion_sequence(self, out_dir: Path, export_3d: Optional[bool]):
        enabled = self.options.export_motion_3d if export_3d is None else export_3d
        return GaussianSequenceWriter(out_dir, self.options.driving_fps) if enabled else None

    @torch.inference_mode()
    def stream_motion_video(self, avatar: Avatar, motion_path: str, out_dir: str,
                            max_frames: Optional[int] = None, export_3d: Optional[bool] = None):
        """Yield (completed, total, TS chunk, final MP4) without per-frame PNG files.

        Chunk items have final_mp4=None. The final item has chunk=None and the saved MP4.
        Optional 3D export writes posed PLYs plus a manifest, without changing these yields.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        limit = self.options.driving_max_frames if max_frames is None else max_frames
        name, motion, keys = load_driving_motion(motion_path, limit)
        total = len(keys)
        if not total:
            raise ValueError("Motion has no frames")
        sequence = self._motion_sequence(out_dir, export_3d)
        with VideoChunkEncoder(out_dir, self.options.driving_fps, self.options.motion_chunk_frames) as encoder:
            for index, key in enumerate(keys, 1):
                entry = motion[key]
                posed = _animate_motion_frame_gs(
                    self.unwrapped, avatar.gs_tpose, avatar.avatar_tokens, entry, self.device
                )
                if sequence:
                    sequence.write(posed, key)
                frame = render_motion_frame_original(
                    posed, entry, self.device, max_render_height=self.options.max_render_height,
                    include_alpha=False,
                )
                for completed, chunk in encoder.add(frame, index):
                    yield completed, total, chunk, None
            for completed, chunk in encoder.finish(total):
                yield completed, total, chunk, None
            video = encoder.combine(out_dir / f"{name}_original.mp4")
        if sequence:
            print(f"3D motion sequence: {sequence.finish()}", flush=True)
        yield total, total, None, video

    @torch.inference_mode()
    def render_motion_window(self, avatar: Avatar, motion_path: str,
                             crop_xyxy: tuple, out_hw: tuple, out_dir: str,
                             max_frames: Optional[int] = None) -> list:
        """Drive the avatar through one MHR motion file, rendered into a fixed pixel window.

        Unlike ``render_motion`` (human-crop space / raw source-frame space), this renders
        through ``crop_xyxy`` -- a caller-chosen rect in the driving view's own raw pixel
        coordinates, fixed for the whole clip -- resized to ``out_hw``. Writes one RGBA PNG
        per frame (``%06d.png``) to ``out_dir`` and returns the sorted list of paths; the
        caller owns encoding to video, since a guidance-corpus build needs an exact, recorded
        ffmpeg invocation matching its target track's encode.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        limit = self.options.driving_max_frames if max_frames is None else max_frames
        _, sam3db, frame_keys = load_driving_motion(motion_path, limit)

        paths = []
        for idx, frame_key in enumerate(frame_keys, start=1):
            entry = sam3db[frame_key]
            gs_posed = _animate_motion_frame_gs(
                self.unwrapped, avatar.gs_tpose, avatar.avatar_tokens, entry, self.device
            )
            rgba = render_motion_frame_window(gs_posed, entry, self.device, crop_xyxy, out_hw)
            out_path = out_dir / f"{idx:06d}.png"
            Image.fromarray(rgba, mode="RGBA").save(out_path)
            paths.append(str(out_path))
        return paths

    # -- export ---------------------------------------------------------------

    def save_avatar(self, avatar: Avatar, out_path: str) -> str:
        """Write the canonical Gaussians + first-frame pose, the ``avatar_*.pth`` format."""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(extract_avatar_data(avatar.last_pass), out_path)
        return out_path

    @torch.inference_mode()
    def posed_gaussians(self, avatar: Avatar, motion_path: str = "", frame: int = 0):
        """The Gaussians in world space: rest pose by default, or one motion frame.

        This is what ``infer_argavatar.py --save_ply`` / ``--ply_frame`` export.
        """
        from xlib.model.mhr_animatable_gaussians import animate_gs

        if motion_path:
            _, sam3db, frame_keys = load_driving_motion(motion_path, 0)
            if not 0 <= frame < len(frame_keys):
                raise IndexError(f"frame {frame} out of range for {len(frame_keys)}-frame motion")
            entry = sam3db[frame_keys[frame]]
            return _animate_motion_frame_gs(
                self.unwrapped, avatar.gs_tpose, avatar.avatar_tokens, entry, self.device
            )

        lbs_cache = slice_batch(
            self.unwrapped.mhr_animatable_gs.precompute_lbs_cache([self._rest_pose_dic(avatar)]), 0
        )
        gs_posed, _ = animate_gs(
            self.unwrapped.mhr_animatable_gs, slice_batch(avatar.gs_tpose, 0), None,
            lbs_cache=lbs_cache, pose_indices=[0], to_colmap=True, offset_pose=None,
        )
        return gs_posed


# -- Gaussian export ---------------------------------------------------------

# INRIA 3DGS vertex properties, in the order the reference implementation writes them.
_PLY_PROPS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)
PLY_FORMATS = ("3dgs", "spark")


def _rendered_gaussians(gs):
    """The ``[N, ...]`` Gaussians the renderer draws: batch 0, view 0.

    ``animate_gs`` returns ``[B, V, N, ...]`` fields, and a multi-view subject's motion-frame
    deformation is computed once per input view, so scale and rotation can carry ``V``
    copies while ``xyz`` carries one. ``render_posed_gs`` always draws ``index_gs(gs, 0, 0)``;
    exporting the same slice keeps every field the same length and the file identical to
    what is rendered.
    """
    from xlib.render.gaussian_splatting_3d import index_gs

    return index_gs(gs, 0, 0) if gs.xyz.dim() > 2 else gs


def _flatten(tensor: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Collapse any leading batch/pose dimensions, leaving [N, last_dim] on the CPU."""
    return tensor.detach().float().cpu().reshape(-1, last_dim)


def write_gaussian_ply(gs, out_path: str, centerize: bool = True) -> str:
    """Write ``gs`` in the INRIA 3DGS PLY layout. Returns ``out_path``.

    The layout is ``x y z nx ny nz f_dc_* opacity scale_* rot_*``, binary little-endian,
    rotation as **wxyz** -- what SuperSplat, the original viewer and most web viewers
    expect. ``gs`` is the object ``xlib.model.mhr_animatable_gaussians.animate_gs``
    returns. Its stored quantities are *activated* (positive scales, opacity in (0, 1)),
    whereas the 3DGS format stores the pre-activation values -- so scales are log'd and
    opacity is logit'd here, exactly as ``export_spark_ply`` does.
    """
    gs = _rendered_gaussians(gs)
    xyz = _flatten(gs.xyz, 3)
    scale = _flatten(gs.scale, 3).clamp_min(1e-12)
    rotation = torch.nn.functional.normalize(_flatten(gs.rotation, 4), dim=-1, eps=1e-8)
    opacity = _flatten(gs.opacity, 1).clamp(1e-6, 1 - 1e-6)

    if getattr(gs, "sh", None) is not None and gs.sh.numel() > 0:
        sh = gs.sh.detach().float().cpu()
        # [..., n_coeffs, 3] -> the DC term of each Gaussian
        f_dc = sh.reshape(-1, sh.shape[-2], 3)[:, 0]
    else:
        f_dc = torch.zeros((xyz.shape[0], 3), dtype=torch.float32)

    if centerize:
        xyz = xyz - xyz.mean(dim=0, keepdim=True)

    # animate_gs stores rotation as xyzw (see the argavatar_rotation_order comment in
    # spark_utils); the 3DGS format is wxyz.
    rotation_wxyz = torch.cat([rotation[:, 3:4], rotation[:, 0:3]], dim=-1)

    rows = torch.cat(
        [
            xyz,
            torch.zeros_like(xyz),  # nx ny nz -- unused by 3DGS viewers, written as zeros
            f_dc,
            torch.logit(opacity),
            scale.log(),
            rotation_wxyz,
        ],
        dim=-1,
    ).numpy().astype("<f4", copy=False)
    assert rows.shape[1] == len(_PLY_PROPS), (rows.shape, len(_PLY_PROPS))

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {rows.shape[0]}",
        *[f"property float {name}" for name in _PLY_PROPS],
        "end_header",
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        rows.tofile(handle)
    return out_path


def export_gaussians(gs, out_path: str, fmt: str = "3dgs", centerize: bool = True) -> str:
    """Write ``gs`` as a ``.ply``: ``3dgs`` (INRIA layout) or ``spark``.

    ``spark`` is ``xlib.misc.spark_utils.export_spark_ply``, a compact 14-float layout
    (rotation **xyzw**, no normals) kept because existing tooling in this repo reads it.
    """
    if fmt == "spark":
        from xlib.misc.spark_utils import GaussianParameters, export_spark_ply

        gs = _rendered_gaussians(gs)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        export_spark_ply(
            GaussianParameters(gs.xyz, gs.scale, gs.rotation, gs.opacity, getattr(gs, "sh", None)),
            out_path,
            centerize=centerize,
        )
        return out_path
    if fmt != "3dgs":
        raise ValueError(f"unknown PLY format {fmt!r}; expected one of {PLY_FORMATS}")
    return write_gaussian_ply(gs, out_path, centerize=centerize)


def _encode_mp4(frame_dir: str, out_path: str, fps: float) -> None:
    """Encode frames to browser-ready H.264 with even dimensions and fast-start metadata."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", fps_arg(fps),
            "-i", os.path.join(frame_dir, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart",
            out_path,
        ],
        check=True,
    )


def expand_image_args(patterns: Iterable[str]) -> list:
    """Resolve CLI image arguments -- files, directories, or globs -- to a sorted list.

    ``infer_argavatar.py --images`` takes one subject's views this way; ``--input_dir``
    is the two-level parent-of-subject-folders layout instead.
    """
    import glob as glob_mod

    from scripts.inference.infer_argavatar import VALID_EXTS

    resolved = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            matches = [
                os.path.join(pattern, f)
                for f in sorted(os.listdir(pattern))
                if os.path.splitext(f)[1].lower() in VALID_EXTS
            ]
        elif os.path.isfile(pattern):
            matches = [pattern]
        else:
            matches = sorted(
                p for p in glob_mod.glob(pattern)
                if os.path.splitext(p)[1].lower() in VALID_EXTS
            )
        if not matches:
            raise FileNotFoundError(f"no images matched: {pattern}")
        resolved.extend(matches)

    seen, unique = set(), []
    for path in resolved:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique
