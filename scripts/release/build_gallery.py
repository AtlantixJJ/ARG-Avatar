#!/usr/bin/env python3
"""Maintainer tool: build the README gallery in `assets/gallery/`.

Each subject folder under `--renderings_source` holds that subject's input views
(`*__input_images__*.png`). For each subject in SUBJECT_LOOPS this reconstructs the avatar
from those views and writes:

    assets/gallery/<subject>_orbit.gif    360-degree turntable of the canonical avatar
    assets/gallery/<subject>_motion.gif   the avatar driven through one full seamless loop
                                        (`assets/example_motion/woman_dance_2_loopNN.pth`),
                                        next to the driving MHR mesh, so the GIF loops
    assets/gallery/gallery.json           which inputs and motions produced the GIFs

The mesh half comes from `<renderings_source>/woman_dance_2_mesh/mesh_loopNN_*.mp4`,
which has exactly one frame per motion frame and the same camera as the avatar render.
Reconstruction and rendering go through `scripts/inference/pipeline.ARGAvatarPipeline` --
the same calls `infer_argavatar.py` makes. Only the GIFs are committed, not the input views: the
subjects come from the training/eval datasets, so the renders carry those datasets' terms
(see THIRD_PARTY_NOTICES.md). GIFs are used because GitHub READMEs do not play
repository-hosted videos inline.

    python scripts/release/build_gallery.py
    python scripts/release/build_gallery.py --only dnarendering_048_00008
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, ".")

DEFAULT_RENDERINGS_SOURCE = Path("/home/jianjinx/data2/ARG-Avatar/expr/rendering_videos")
OUT = Path("assets/gallery")
MOTIONS = Path("assets/example_motion")

# Gallery subject folder -> the seamless loop it dances, pinned so a rebuild matches.
# The loops are a woman's dance, so the gallery shows female subjects. Each folder holds
# six input views of one DNA-Rendering sequence: view i from the i-th of six evenly
# spaced frames.
SUBJECT_LOOPS = {
    "dnarendering_010_00008": "loop01",
    "dnarendering_048_00008": "loop02",
    "dnarendering_165_00008": "loop04",
}

GIF_HEIGHT = 256
ORBIT_FRAMES = 72        # one full turn
ORBIT_GIF_FPS = 18       # -> a 4 s turn
MOTION_FPS = 30         # frame rate of the loop motions and their mesh renders
MOTION_GIF_FPS = 15     # every other frame


def run(cmd: list) -> None:
    subprocess.run([str(c) for c in cmd], check=True)


def probe_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(video)], check=True, capture_output=True, text=True,
    ).stdout.strip()
    width, height = (int(v) for v in out.split(",")[:2])
    return width, height


def ink_bbox(video: Path, height: int, pad: float = 0.06) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` around everything non-white anywhere in ``video``, in the
    coordinates of the video scaled to ``height`` px high."""
    with tempfile.TemporaryDirectory(prefix="bbox_") as tmp:
        run(["ffmpeg", "-loglevel", "error", "-i", video, "-vf", f"fps=6,scale=-2:{height}",
             f"{tmp}/%04d.png"])
        frames = [np.asarray(Image.open(f).convert("L")) for f in sorted(Path(tmp).glob("*.png"))]
    ink = np.any(np.stack(frames) < 235, axis=0)
    ys, xs = np.nonzero(ink)
    h, w = ink.shape
    margin = int(pad * max(xs.max() - xs.min(), ys.max() - ys.min()))
    return (max(xs.min() - margin, 0), max(ys.min() - margin, 0),
            min(xs.max() + margin, w), min(ys.max() + margin, h))


def mesh_and_avatar(mesh_video: Path, avatar_video: Path, out: Path, height: int = 720) -> Path:
    """``mesh | avatar`` side by side, frame for frame, each cropped to its own figure.

    Both halves share one vertical range, so they keep the same scale -- they come from
    the same camera.
    """
    boxes = [ink_bbox(v, height) for v in (mesh_video, avatar_video)]
    y0, y1 = min(b[1] for b in boxes), max(b[3] for b in boxes)
    crop_h = (y1 - y0) // 2 * 2
    crops = [f"crop={(b[2] - b[0]) // 2 * 2}:{crop_h}:{b[0]}:{y0}" for b in boxes]
    graph = (f"[0:v]scale=-2:{height},{crops[0]}[m];[1:v]scale=-2:{height},{crops[1]}[a];"
             "[m][a]hstack=shortest=1")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", mesh_video, "-i", avatar_video,
         "-filter_complex", graph, "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", out])
    return out


def to_gif(video: Path, gif: Path, fps: float, prefilter: str = "null") -> None:
    """Encode ``video`` as a looping GIF, GIF_HEIGHT high, with an optimised palette."""
    graph = (
        f"{prefilter},fps={fps},scale=-2:{GIF_HEIGHT}:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
    )
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", graph, "-loop", "0", gif])


def discover(source: Path, only: list) -> list:
    """``[(subject, input views, loop motion .pth, loop mesh .mp4)]`` for SUBJECT_LOOPS."""
    subjects = []
    for name, loop in SUBJECT_LOOPS.items():
        if only and name not in only:
            continue
        views = sorted((source / name).glob("*__input_images__*.png"))
        motion = MOTIONS / f"woman_dance_2_{loop}.pth"
        meshes = sorted((source / "woman_dance_2_mesh").glob(f"mesh_{loop}_*.mp4"))
        if not views or not motion.is_file() or not meshes:
            raise FileNotFoundError(f"{name}: need input views in {source / name}, {motion}, "
                                    f"and {source}/woman_dance_2_mesh/mesh_{loop}_*.mp4")
        subjects.append((name, views, motion, meshes[0]))
    return subjects


def main(args: argparse.Namespace) -> None:
    subjects = discover(args.renderings_source, args.only)
    if not subjects:
        raise SystemExit(f"--only matched nothing; choose from {sorted(SUBJECT_LOOPS)}")

    from scripts.inference.pipeline import ARGAvatarPipeline, PipelineOptions

    pipeline = ARGAvatarPipeline.build(PipelineOptions(driving_fps=MOTION_FPS))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "gallery.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    for stale in sorted(set(manifest) - set(SUBJECT_LOOPS)):   # subjects dropped from the gallery
        del manifest[stale]
        for gif in OUT.glob(f"{stale}_*.gif"):
            gif.unlink()

    for name, views, motion, mesh in subjects:
        print(f"[build] {name}: {len(views)} views, motion {motion.name}")
        avatar = pipeline.reconstruct([str(v) for v in views], name=name)
        with tempfile.TemporaryDirectory(prefix="gallery_") as tmp:
            # Encode at the GIF rate so the GIF keeps every one of the ORBIT_FRAMES views.
            orbit = pipeline.render_orbit_video(avatar, f"{tmp}/orbit.mp4", n_frames=ORBIT_FRAMES,
                                                canvas_size=512, fps=ORBIT_GIF_FPS)
            to_gif(Path(orbit), OUT / f"{name}_orbit.gif", ORBIT_GIF_FPS)

            videos = pipeline.render_motion(avatar, str(motion), tmp, export_3d=False)
            both = mesh_and_avatar(mesh, Path(videos["original"]), Path(tmp) / "both.mp4")
            to_gif(both, OUT / f"{name}_motion.gif", MOTION_GIF_FPS)

        manifest[name] = {
            "views": [v.name for v in views],
            "motion": str(motion),
            "mesh_render": mesh.name,
            "orbit_frames": ORBIT_FRAMES,
            "checkpoint": pipeline.options.checkpoint_path,
        }
        sizes = {f.name: f"{f.stat().st_size / 1e6:.2f} MB" for f in sorted(OUT.glob(f"{name}_*.gif"))}
        print(f"        -> {sizes}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--renderings_source", type=Path, default=DEFAULT_RENDERINGS_SOURCE,
                        help="Folder of curated renders, one subfolder per subject")
    parser.add_argument("--only", nargs="+", default=[], metavar="SUBJECT",
                        help="Rebuild only these subject folders")
    main(parser.parse_args())
