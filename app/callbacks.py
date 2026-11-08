"""Callback handlers and pipeline orchestration for the ARGAvatar Gradio app."""
from __future__ import annotations

import os
import filecmp
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Final

import gradio as gr
from PIL import Image, ImageOps

from app.state import AppState, InputMode, MotionSource
from scripts.inference.pipeline import (
    ARGAvatarPipeline,
)
from app.viewer import splat_viewer_html
from scripts.inference.pipeline import write_gaussian_ply

ASSETS: Final[Path] = Path("assets")
WORK_ROOT: Final[Path] = Path("expr/app")
MAX_VIDEO_SIZE_MB: Final[int] = 20
MAX_VIDEO_SIZE_BYTES: Final[int] = MAX_VIDEO_SIZE_MB * 1024 * 1024

VIDEO_EXTENSIONS: Final[set[str]] = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}

_PIPELINE: ARGAvatarPipeline | None = None
_NO_LOAD: bool = False


def set_pipeline(pipeline: ARGAvatarPipeline | None) -> None:
    """Set the global ARGAvatar pipeline instance."""
    global _PIPELINE
    _PIPELINE = pipeline


def get_pipeline() -> ARGAvatarPipeline:
    """Retrieve the global ARGAvatar pipeline or raise a user-facing error."""
    if _PIPELINE is None:
        if _NO_LOAD:
            raise gr.Error("Model is not loaded (running in --no_load mode).")
        raise gr.Error("Model is not loaded yet — wait for startup to finish and retry.")
    return _PIPELINE


def set_no_load(no_load: bool) -> None:
    """Configure whether the app runs in mock/no-load mode."""
    global _NO_LOAD
    _NO_LOAD = no_load


def is_no_load_mode() -> bool:
    """Check if the app is running in mock/no-load mode."""
    return _NO_LOAD


def pipeline_status_line() -> str:
    """One-line description of what the app has loaded, for the header badge."""
    if _NO_LOAD:
        return "**Status:** UI-only (`--no_load`) — reconstruction and animation are mocked"
    if _PIPELINE is None:
        return "**Status:** model not loaded"
    checkpoint = Path(_PIPELINE.options.checkpoint_path or "unknown").name
    return f"**Status:** model loaded — `{checkpoint}` on `{_PIPELINE.device}`"


def new_work_dir() -> Path:
    """Allocate an exclusive directory for one request's files."""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=WORK_ROOT))


def cleanup_work_dirs(work_root: Path = WORK_ROOT, max_age_hours: int = 24) -> int:
    """Remove working directories older than max_age_hours on startup."""
    if not work_root.is_dir():
        return 0
    cutoff_time = time.time() - (max_age_hours * 3600)
    cleaned_count = 0
    for child in work_root.iterdir():
        if child.is_dir():
            try:
                if child.stat().st_mtime < cutoff_time:
                    shutil.rmtree(child, ignore_errors=True)
                    cleaned_count += 1
            except Exception:
                pass
    return cleaned_count


def bundled_motions() -> dict[str, str]:
    """Retrieve mapping of display names to `.pth` file paths in assets/example_motion/."""
    motion_dir = ASSETS / "example_motion"
    if not motion_dir.is_dir():
        return {}
    return {path.stem: str(path) for path in sorted(motion_dir.glob("*.pth"))}


def bundled_motion_previews() -> dict[str, str]:
    """Retrieve mapping of display names to `.mp4` preview paths in assets/example_motion/."""
    motion_dir = ASSETS / "example_motion"
    if not motion_dir.is_dir():
        return {}
    return {path.stem: str(path) for path in sorted(motion_dir.glob("*.mp4"))}


def get_motion_preview(motion_name: str | None) -> str | None:
    """Return the `.mp4` preview path for a given motion name."""
    if not motion_name:
        return None
    return bundled_motion_previews().get(motion_name)


def example_subjects(subdir: str = "example_images") -> dict[str, list[str]]:
    """Retrieve `{"<dataset>/<clip>": [image_paths]}` for each clip folder under
    `assets/<subdir>/<dataset>/`, with NeuMan, THuman2.1 and DNA-Rendering first."""
    directory = ASSETS / subdir
    if not directory.is_dir():
        return {}
    subjects: dict[str, list[str]] = {}
    dataset_priority = {"neuman": 0, "thuman21": 1, "dnarendering": 2}
    subject_dirs = sorted(
        directory.glob("*/*"),
        key=lambda path: (
            dataset_priority.get(path.parent.name, len(dataset_priority)),
            path.as_posix(),
        ),
    )
    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue
        images = sorted(
            str(path)
            for path in subject_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            subjects[subject_dir.relative_to(directory).as_posix()] = images
    return subjects


def extract_image_paths(files: Any) -> list[str]:
    """Extract list of file paths from Gradio upload or gallery inputs.

    Handles str, list of strings or dicts, and objects with name or path attributes.
    """
    if not files:
        return []
    if isinstance(files, str):
        return [files] if files else []
    if isinstance(files, (list, tuple)):
        paths: list[str] = []
        for f in files:
            if isinstance(f, str):
                paths.append(f)
            elif isinstance(f, (list, tuple)) and len(f) > 0 and f[0] is not None:
                paths.append(str(f[0]))
            elif isinstance(f, dict):
                p = (
                    f.get("path")
                    or f.get("name")
                    or (f.get("image", {}).get("path") if isinstance(f.get("image"), dict) else None)
                )
                if p:
                    paths.append(str(p))
            elif hasattr(f, "name") and getattr(f, "name", None):
                paths.append(str(f.name))
            elif hasattr(f, "path") and getattr(f, "path", None):
                paths.append(str(f.path))
            elif f is not None:
                paths.append(str(f))
        return [p for p in paths if p]
    if hasattr(files, "name") and getattr(files, "name", None):
        return [str(files.name)]
    if hasattr(files, "path") and getattr(files, "path", None):
        return [str(files.path)]
    return [str(files)]


def validate_video_file(video_path: str, max_size_bytes: int = MAX_VIDEO_SIZE_BYTES) -> None:
    """Check that the video file exists and does not exceed the size limit."""
    if not os.path.isfile(video_path):
        raise gr.Error(f"Video file not found: {video_path}")
    file_size = os.path.getsize(video_path)
    if file_size > max_size_bytes:
        raise gr.Error(
            f"Uploaded video ({file_size / (1024 * 1024):.1f}MB) exceeds the {MAX_VIDEO_SIZE_MB}MB maximum limit."
        )


def is_same_file(p1: str, p2: str) -> bool:
    """Match original image paths with Gradio's byte-identical cached copies."""
    try:
        return Path(p1).resolve() == Path(p2).resolve() or filecmp.cmp(p1, p2, shallow=False)
    except OSError:
        return p1 == p2


def _evenly_spaced_subset(paths: list[str], k: int) -> list[str]:
    """Pick k items from paths, evenly spaced by index (paths must already be ordered)."""
    if len(paths) <= k:
        return list(paths)
    if k <= 1:
        return [paths[0]]
    step = (len(paths) - 1) / (k - 1)
    indices = [int(round(i * step)) for i in range(k)]
    return [paths[idx] for idx in sorted(set(indices))]


def extract_frames_from_video(
    video_path: str,
    output_dir: Path,
    max_views: int = 6,
) -> list[str]:
    """Extract individual frames from an uploaded video and sample up to max_views evenly.

    Every decoded frame is left on disk under `output_dir` (not just the returned
    subsample) so a later call can re-select from the full set, e.g. via
    `auto_select_frames_fps` -- see `on_video_uploaded` / `on_auto_select_frames`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                str(output_dir / "frame_%06d.png"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        err_msg = (exc.stderr or exc.stdout or "ffmpeg process returned non-zero exit status")[:500]
        raise gr.Error(f"Failed to extract video frames via ffmpeg: {err_msg}") from exc

    all_frames = sorted(str(p) for p in output_dir.glob("frame_*.png"))
    if not all_frames:
        raise gr.Error("Failed to extract frames from the uploaded video.")

    return _evenly_spaced_subset(all_frames, max_views)


def toggle_view_selection(
    clicked_path: str,
    current_files: Any,
    associated_video: str | None = None,
    state: AppState | None = None,
) -> tuple[list[str], Any, AppState]:
    """Toggle one view in/out of the active selection, for the video-mode extracted-frames
    gallery and the Subject Library (both deal in image paths only)."""
    if state is None:
        state = AppState()

    current_paths = extract_image_paths(current_files)
    if not current_paths and state.active_views:
        current_paths = list(state.active_views)
    current_paths = [p for p in current_paths if Path(p).suffix.lower() in IMAGE_EXTENSIONS]

    already_selected_idx = None
    for i, p in enumerate(current_paths):
        if is_same_file(p, clicked_path):
            already_selected_idx = i
            break

    if already_selected_idx is not None:
        # Clicked again -> de-select
        new_paths = [p for i, p in enumerate(current_paths) if i != already_selected_idx]
    else:
        # Clicked to select -> add to the active selection
        new_paths = current_paths + [clicked_path]

    state.active_views = new_paths
    if new_paths:
        state.input_video_path = associated_video
        return (
            new_paths,
            gr.update(
                value=new_paths,
                visible=True,
                label=f"Selected Views ({len(new_paths)} view(s))",
            ),
            state,
        )
    else:
        state.input_video_path = None
        state.active_subject_id = None
        return (
            [],
            gr.update(
                value=None,
                visible=False,
            ),
            state,
        )


def normalize_uploaded_images(paths: list[str]) -> list[str]:
    """Convert HEIC and EXIF-rotated uploads to upright RGB PNGs for the pipeline."""
    normalized = []
    work_dir = None
    for path in paths:
        # Existing callbacks also accept logical paths in no-load mode.
        if not Path(path).is_file():
            normalized.append(path)
            continue
        is_heif = Path(path).suffix.lower() in {".heic", ".heif"}
        if is_heif:
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError as exc:
                raise gr.Error("HEIC photos need pillow-heif on the server. "
                               "Upload JPEG/PNG or ask the server owner to install pillow-heif.") from exc
        try:
            with Image.open(path) as image:
                if is_heif or image.getexif().get(274, 1) != 1:
                    work_dir = work_dir or new_work_dir()
                    output = work_dir / f"upload-{uuid.uuid4().hex}.png"
                    ImageOps.exif_transpose(image).convert("RGB").save(output)
                    path = str(output)
        except (OSError, ValueError) as exc:
            raise gr.Error("Could not read this photo. Try uploading a JPEG or PNG.") from exc
        normalized.append(path)
    return normalized


def on_images_uploaded(
    gallery_value: Any,
    current_subs: list[dict] | None,
    state: AppState,
) -> tuple[Any, list[dict], AppState]:
    """Handle images dropped into the image-mode upload gallery.

    The gallery is interactive, so `gallery_value` already holds its full current content
    (prior images plus whatever was just added). The first upload into an empty box starts
    a new Subject Library entry; further uploads while that entry is still the one backing
    the box extend it in place rather than spawning duplicates.
    """
    if current_subs is None:
        current_subs = []
    paths = [p for p in extract_image_paths(gallery_value) if Path(p).suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        state.active_views = []
        state.active_subject_id = None
        return gr.update(value=None), current_subs, state

    paths = normalize_uploaded_images(paths)
    updated_subs = list(current_subs)
    target = next((s for s in updated_subs if s.get("id") == state.active_subject_id), None)
    if target is not None:
        target["paths"] = paths
    else:
        new_id = len(updated_subs) + 1
        updated_subs.append({
            "id": new_id,
            "name": f"User Subject {new_id}",
            "paths": paths,
            "video_path": None,
        })
        state.active_subject_id = new_id
        state.input_video_path = None

    state.active_views = paths
    return gr.update(value=paths), updated_subs, state


def on_image_deleted(
    gallery_value: Any, state: AppState, evt: gr.EventData = None,
) -> AppState:
    """Handle a single image removed from the image-mode upload gallery via its native
    delete control. Only narrows the current selection -- the Subject Library entry (if
    any) keeps every originally uploaded view until removed there explicitly."""
    paths = [p for p in extract_image_paths(gallery_value) if Path(p).suffix.lower() in IMAGE_EXTENSIONS]
    # Gradio Gallery dispatches delete before removing the item from its value.
    # Generic EventData is used because Gallery sends {file, index}, unlike File.delete.
    if evt is not None:
        index = evt._data.get("index")
        deleted_path = evt._data.get("file", {}).get("path")
        if (isinstance(index, int) and 0 <= index < len(paths)
                and deleted_path and is_same_file(paths[index], deleted_path)):
            paths.pop(index)
    state.active_views = paths
    if not paths:
        state.active_subject_id = None
        state.input_video_path = None
    return state


def on_video_uploaded(
    video_path: str | None,
    current_subs: list[dict] | None,
    max_views: int | float,
    state: AppState,
    progress=gr.Progress(),
) -> tuple[Any, list[dict], AppState]:
    """Handle a video dropped into the video-mode upload box: copy it into a fresh work
    dir and extract evenly spaced frames (6 by default) for reconstruction."""
    if current_subs is None:
        current_subs = []
    if not video_path:
        state.active_views = []
        state.input_video_path = None
        return gr.update(value=None, visible=False), current_subs, state

    validate_video_file(video_path)
    work_dir = new_work_dir()
    local_video = work_dir / f"input_video{Path(video_path).suffix or '.mp4'}"
    shutil.copyfile(video_path, local_video)
    state.input_video_path = str(local_video)

    num_views = max(1, int(max_views))
    progress(0.2, desc=f"Extracting {num_views} evenly distributed frames from uploaded video")
    frames_dir = work_dir / "extracted_frames"
    extracted_frames = extract_frames_from_video(
        str(local_video), frames_dir, max_views=num_views
    )
    state.active_views = extracted_frames
    state.all_extracted_frames = sorted(str(p) for p in frames_dir.glob("frame_*.png"))

    updated_subs = list(current_subs)
    new_id = len(updated_subs) + 1
    updated_subs.append({
        "id": new_id,
        "name": f"User Subject {new_id}",
        "paths": extracted_frames,
        "video_path": str(local_video),
    })

    return gr.update(
        value=extracted_frames,
        visible=True,
        label=f"Extracted Video Frames ({len(extracted_frames)} views selected for reconstruction)",
    ), updated_subs, state


def on_video_cleared(state: AppState) -> tuple[Any, AppState]:
    """Handle clearing the video-mode upload box."""
    state.active_views = []
    state.input_video_path = None
    state.all_extracted_frames = []
    return gr.update(value=None, visible=False), state


# ---------------------------------------------------------------------------
# LPIPS farthest-point-sampling frame selection ("Auto-Select Frames" button)
#
# Port of the `fps` mode in /home/jianjinx/data2/SAM3DGS/scripts/HumanVid/
# 06_filter_frames_by_lpips.py: extract L2-normalised LPIPS backbone features per frame,
# then greedily pick the frame farthest (in cosine distance) from the already-selected
# set. Chooses visually diverse frames instead of naive even time-spacing, e.g. skipping
# over a run of near-duplicate frames where the subject briefly stopped turning.
# ---------------------------------------------------------------------------

_LPIPS_MODEL_CACHE: dict[str, Any] = {}


def _get_lpips_model(device: Any) -> Any:
    """Lazily build (and cache per-device) the LPIPS AlexNet backbone."""
    import lpips as lpips_module

    key = str(device)
    model = _LPIPS_MODEL_CACHE.get(key)
    if model is None:
        model = lpips_module.LPIPS(net="alex").to(device)
        model.eval()
        model.requires_grad_(False)
        _LPIPS_MODEL_CACHE[key] = model
    return model


def _load_lpips_tensor(path: str, device: Any, resize_width: int = 224) -> Any:
    from PIL import Image
    from torchvision import transforms

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if resize_width > 0 and w != resize_width:
        new_h = max(1, round(h * resize_width / w))
        img = img.resize((resize_width, new_h), Image.BILINEAR)
    t = transforms.ToTensor()(img) * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def _extract_lpips_features(
    paths: list[str],
    loss_fn: Any,
    device: Any,
    resize_width: int = 224,
    batch_size: int = 16,
) -> Any:
    """Global-average-pooled, L2-normalised LPIPS backbone feature per path -> (N, D) tensor."""
    import torch

    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            tensors = torch.cat(
                [_load_lpips_tensor(p, device, resize_width) for p in batch_paths], dim=0
            )
            feature_maps = loss_fn.net(tensors)
            pooled = torch.cat([m.mean(dim=[2, 3]) for m in feature_maps], dim=1)
            pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-8)
            feats.append(pooled)
    return torch.cat(feats, dim=0)


def _fps_select_indices(features: Any, k: int) -> list[int]:
    """Greedy farthest point sampling over L2-normalised feature vectors (cosine distance).

    Seeds with the frame farthest from the mean feature (better diversity than index 0),
    then repeatedly adds whichever remaining frame is farthest from the selected set.
    """
    import torch

    n = features.shape[0]
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]

    mean_feat = features.mean(dim=0)
    mean_feat = mean_feat / (mean_feat.norm() + 1e-8)
    seed = int(torch.argmax(1.0 - features @ mean_feat).item())

    selected = [seed]
    distances = torch.full((n,), float("inf"), device=features.device)
    for _ in range(1, k):
        last = selected[-1]
        dist = 1.0 - (features @ features[last])
        distances = torch.minimum(distances, dist)
        next_idx = int(torch.argmax(distances).item())
        selected.append(next_idx)
    return sorted(selected)


def auto_select_frames_fps(frame_paths: list[str], k: int, device: Any) -> list[str]:
    """Pick k visually diverse frames out of frame_paths via LPIPS-feature farthest point
    sampling. Returns all of frame_paths unchanged if there are k or fewer."""
    if len(frame_paths) <= k:
        return list(frame_paths)
    loss_fn = _get_lpips_model(device)
    features = _extract_lpips_features(frame_paths, loss_fn, device)
    indices = _fps_select_indices(features, k)
    return [frame_paths[i] for i in indices]


def on_auto_select_frames(
    state: AppState,
    max_views: int | float,
    progress=gr.Progress(),
) -> tuple[Any, AppState]:
    """Handle the 'Auto-Select Frames' button: re-pick the video-mode frame set from every
    extracted frame via LPIPS FPS instead of the naive even-time-spacing default."""
    candidates = state.all_extracted_frames
    if not candidates:
        raise gr.Error("Upload a video first so frames can be extracted.")

    k = max(1, int(max_views))

    if is_no_load_mode():
        selected = _evenly_spaced_subset(candidates, k)
    else:
        import torch

        device = _PIPELINE.device if _PIPELINE is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        progress(0.3, desc=f"Selecting {k} diverse frames via LPIPS FPS")
        selected = auto_select_frames_fps(candidates, k, device)

    state.active_views = selected
    return gr.update(
        value=selected,
        visible=True,
        label=f"Extracted Video Frames ({len(selected)} view(s), auto-selected via LPIPS FPS)",
    ), state


def on_frames_gallery_select(
    evt: gr.SelectData,
    cur_files: Any,
    state: AppState,
) -> tuple[list[str], Any, AppState]:
    """Handle click selection/deselection within the video-mode extracted-frames gallery."""
    current_paths = extract_image_paths(cur_files) or list(state.active_views or [])
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if isinstance(idx, int) and 0 <= idx < len(current_paths):
        clicked_path = current_paths[idx]
        return toggle_view_selection(clicked_path, cur_files, state=state)
    return extract_image_paths(cur_files), gr.update(), state


def on_input_mode_change(mode: str) -> tuple[Any, Any]:
    """Show the image-mode upload gallery or the video-mode upload box + frames gallery."""
    is_images = mode == InputMode.IMAGES.value
    return gr.update(visible=is_images), gr.update(visible=not is_images)


def run_reconstruct(
    input_mode: str,
    image_gallery_value: Any,
    video_value: Any,
    gallery_frames: Any,
    max_views: int | float,
    state: AppState,
    progress=gr.Progress(),
) -> tuple[str, Any, str, AppState]:
    """Run sparse-view 3D avatar reconstruction and export the canonical Gaussians.

    Returns ``(viewer_html, ply_download_update, status, state)``: the interactive viewer
    page for the rest-pose Gaussians, and the same ``.ply`` as a download.
    """
    max_views_int = max(1, int(max_views))
    is_video_mode = input_mode == InputMode.VIDEO.value

    work_dir = new_work_dir()

    if is_video_mode:
        candidate_views = extract_image_paths(gallery_frames) or list(state.active_views or [])
        if not candidate_views:
            if not video_value:
                raise gr.Error("Upload a video of the person first.")
            validate_video_file(video_value)
            local_video = work_dir / f"input_video{Path(video_value).suffix or '.mp4'}"
            shutil.copyfile(video_value, local_video)
            state.input_video_path = str(local_video)
            frames_dir = work_dir / "extracted_frames"
            candidate_views = extract_frames_from_video(
                str(local_video), frames_dir, max_views=max_views_int
            )
            state.active_views = candidate_views
            state.all_extracted_frames = sorted(str(p) for p in frames_dir.glob("frame_*.png"))
    else:
        candidate_views = extract_image_paths(image_gallery_value) or list(state.active_views or [])

    if not candidate_views:
        raise gr.Error("Upload at least one image or a video of the person first.")

    # Cap here rather than mutating the shared pipeline options: the pipeline object is a
    # process-wide singleton, so a per-request write to it would race across sessions.
    if len(candidate_views) > max_views_int:
        candidate_views = candidate_views[:max_views_int]
        state.active_views = list(candidate_views)

    state.work_dir = work_dir

    if is_no_load_mode():
        progress(0.2, desc="[no_load mode] Simulating detection and reconstruction")
        state.avatar = "dummy_avatar"
        time.sleep(0.5)
        return (
            splat_viewer_html(None, "[no_load mode] No Gaussians are reconstructed, so there is nothing to view."),
            gr.update(value=None, visible=False),
            "",
            state,
        )

    pipeline = get_pipeline()

    progress(0.1, desc=f"Detecting and reconstructing from {len(candidate_views)} view(s)")
    try:
        avatar = pipeline.reconstruct(candidate_views, name="subject")
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    state.avatar = avatar

    progress(0.8, desc="Exporting Gaussians")
    ply_path = work_dir / "canonical.ply"
    write_gaussian_ply(pipeline.posed_gaussians(avatar), str(ply_path))

    return (
        splat_viewer_html(ply_path),
        gr.update(value=str(ply_path), visible=ply_path.is_file()),
        "",
        state,
    )


def fit_motion_from_video(video_path: str, max_frames: int, progress: gr.Progress) -> str:
    """Run estimate_motion_mhr.py over an uploaded video, returning the sam3db motion path."""
    validate_video_file(video_path)
    motion_name = f"upload-{uuid.uuid4().hex}"
    expr_dir = Path("expr/motion_pose") / motion_name
    frames_dir = expr_dir / "images"
    frames_dir.mkdir(parents=True, exist_ok=True)

    progress(0.1, desc="Extracting frames")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                *(["-frames:v", str(max_frames)] if max_frames > 0 else []),
                str(frames_dir / "%06d.png"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        err_msg = (exc.stderr or exc.stdout or "ffmpeg failed")[:500]
        raise gr.Error(f"Failed to extract video frames via ffmpeg: {err_msg}") from exc

    n_frames = len(list(frames_dir.glob("*.png")))
    if n_frames == 0:
        raise gr.Error("No frames could be read from that video.")

    progress(0.25, desc=f"Fitting MHR poses over {n_frames} frames (this is the slow part)")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/inference/estimate_motion_mhr.py",
            "--video",
            motion_name,
            "--frames_dir",
            str(frames_dir),
            "--stage",
            "all",
            "--strategy",
            "s0",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    if result.returncode != 0:
        raw_output = result.stderr or result.stdout or "No output captured"
        tail = raw_output.strip().splitlines()[-15:]
        if not tail:
            tail = ["No output captured"]
        hint = ""
        if "kaolin" in (result.stderr or ""):
            hint = (
                "\n\nMotion fitting from video needs kaolin. Install it with:\n"
                "  python -m pip install --no-build-isolation -r requirements.txt\n"
                "The bundled motions in the dropdown play without it."
            )
        raise gr.Error("estimate_motion_mhr.py failed:\n" + "\n".join(tail) + hint)

    motion_path = expr_dir / "sam3db_s0.pth"
    if not motion_path.is_file():
        raise gr.Error(f"Motion fitting produced no {motion_path}")
    return str(motion_path)


def run_animate(
    motion_source: str,
    motion_choice: str | None,
    video: Any,
    max_frames: int | float,
    state: AppState,
    progress=gr.Progress(),
):
    """Stream playable H.264 chunks while rendering, retaining the final MP4 on disk."""
    max_frames_int = int(max_frames)
    avatar = state.avatar
    if avatar is None:
        raise gr.Error("Reconstruct an avatar on the first tab before animating it.")

    show_video = (gr.update(visible=False),)

    if is_no_load_mode():
        progress(0.3, desc="[no_load mode] Simulating animation")
        time.sleep(0.5)
        sample_videos = (
            sorted((ASSETS / "example_videos").glob("*.mp4"))
            if (ASSETS / "example_videos").is_dir()
            else []
        )
        dummy_video = str(sample_videos[0]) if sample_videos else None
        selected_motion = (
            motion_choice if motion_source == MotionSource.BUNDLED.value else "custom_video"
        )
        # Use short chunks even in UI-only mode, and keep ffmpeg artifacts out of assets/.
        chunks = [None]
        if dummy_video is not None:
            work_dir = state.work_dir or new_work_dir()
            state.work_dir = work_dir
            chunk_dir = work_dir / f"mock-animation-{uuid.uuid4().hex}"
            chunk_dir.mkdir(parents=True)
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", dummy_video,
                "-t", "2", "-an", "-vf", "scale=512:-2", "-c:v", "libx264",
                "-preset", "veryfast", "-threads", "2", "-pix_fmt", "yuv420p",
                "-g", "15", "-bf", "0", "-f", "segment", "-segment_time", "0.5",
                "-reset_timestamps", "1", str(chunk_dir / "%06d.ts"),
            ], check=True)
            chunks = [str(path) for path in sorted(chunk_dir.glob("*.ts"))]
        for chunk in chunks:
            yield (
                *show_video, chunk,
                f"[no_load mode] Simulated animation with `{selected_motion}`.", state,
            )
        return

    pipeline = get_pipeline()
    work_dir = state.work_dir or new_work_dir()
    state.work_dir = work_dir

    if motion_source == MotionSource.CUSTOM.value:
        if video is None:
            raise gr.Error("Upload a driving video or select Bundled Preset.")
        video_path = video if isinstance(video, str) else getattr(video, "name", str(video))
        validate_video_file(video_path)
        motion_path = fit_motion_from_video(video_path, max_frames_int, progress)
    else:
        motions = bundled_motions()
        if not motion_choice or motion_choice not in motions:
            raise gr.Error("Pick a bundled motion from the dropdown.")
        motion_path = motions[motion_choice]

    name = Path(motion_path).stem
    for idx, total, chunk, video_out in pipeline.stream_motion_video(
        avatar, motion_path, str(work_dir / "motion"), max_frames=max_frames_int
    ):
        if chunk is not None:
            yield (
                gr.skip(), chunk,
                f"Rendering `{name}`: frame {idx}/{total}", state,
            )
        else:
            # The playlist already contains every frame; appending the MP4 would replay them.
            yield (gr.skip(), None, f"Animated with `{name}` ({total} frame(s)).", state)


def on_motion_source_change(source: str) -> tuple[Any, Any]:
    """Show the preset panel or the custom-video panel for the chosen motion source."""
    is_preset = source == MotionSource.BUNDLED.value
    return gr.update(visible=is_preset), gr.update(visible=not is_preset)


def use_input_video(state: AppState) -> tuple[str, str, Any, Any, str]:
    """Retrieve video uploaded in Reconstruct tab and populate driving video input in Animate tab."""
    video_path = state.input_video_path
    if not video_path or not os.path.isfile(video_path):
        raise gr.Error(
            "No video was uploaded in the Reconstruct tab. "
            "Upload a video in Reconstruct or upload one below."
        )
    return (
        MotionSource.CUSTOM.value,
        video_path,
        gr.update(visible=False),
        gr.update(visible=True),
        f"Loaded input video from reconstruction: `{Path(video_path).name}`",
    )


def make_select_handler(
    paths: list[str],
    associated_video: str | None = None,
) -> Callable[[gr.SelectData, Any, AppState], tuple[Any, Any, Any, Any, AppState]]:
    """Factory for Subject Library thumbnail click handlers.

    Toggles the clicked view in/out of the active selection and always routes the result
    to the image-mode upload gallery, switching the Reconstruct tab into Images mode --
    Subject Library entries are photo sets even when they originated from a video upload
    (`associated_video` carries the source video through for the Animate tab's
    "Use Input Video" button).
    """
    def _on_select(evt: gr.SelectData, cur_gallery: Any, state: AppState) -> tuple[Any, Any, Any, Any, AppState]:
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        selected_p = paths[idx] if isinstance(idx, int) and 0 <= idx < len(paths) else paths[0]
        new_paths, _, state = toggle_view_selection(
            selected_p, cur_gallery, associated_video=associated_video, state=state
        )
        return (
            gr.update(value=new_paths),
            gr.update(value=InputMode.IMAGES.value),
            gr.update(visible=True),
            gr.update(visible=False),
            state,
        )
    return _on_select


def make_use_all_handler(
    paths: list[str],
    associated_video: str | None = None,
) -> Callable[[AppState], tuple[Any, Any, Any, Any, AppState]]:
    """Factory for 'Use all views' button click handlers."""
    def _on_use_all(state: AppState) -> tuple[Any, Any, Any, Any, AppState]:
        state.active_views = list(paths)
        state.input_video_path = associated_video
        state.active_subject_id = None
        return (
            gr.update(value=list(paths)),
            gr.update(value=InputMode.IMAGES.value),
            gr.update(visible=True),
            gr.update(visible=False),
            state,
        )
    return _on_use_all


def make_remove_user_handler(
    target_id: int,
) -> Callable[[list[dict] | None], list[dict]]:
    """Factory for removing user-uploaded subjects from state."""
    def _on_remove(cur_subs: list[dict] | None) -> list[dict]:
        filtered = [s for s in (cur_subs or []) if s.get("id") != target_id]
        result = []
        for i, s in enumerate(filtered, 1):
            result.append({
                "id": i,
                "name": f"User Subject {i}",
                "paths": s["paths"],
                "video_path": s.get("video_path"),
            })
        return result
    return _on_remove
