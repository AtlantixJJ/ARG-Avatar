#!/usr/bin/env python
"""ARGAvatar Gradio demo application.

    python -m app                      # http://127.0.0.1:7860
    python -m app --share              # public gradio.live link
    python -m app --checkpoint_path checkpoints/ARGAvatar-Final.pth

Run from the repository root with `PYTHONPATH=` empty. UI assembly is in `app/ui.py`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import gradio as gr
from starlette.middleware import Middleware

sys.path.insert(0, ".")

from app.callbacks import ASSETS, WORK_ROOT, cleanup_work_dirs, set_no_load, set_pipeline
from app.ui import APP_CSS, APP_JS, CONCURRENCY_LIMIT, build_ui
from app.files import SessionFileMiddleware
from scripts.inference.pipeline import DEFAULT_CONFIG_PATH, ARGAvatarPipeline, PipelineOptions


def main() -> None:
    """CLI entrypoint for running the ARGAvatar Gradio server."""
    parser = argparse.ArgumentParser(description="ARGAvatar Gradio demo")
    parser.add_argument("--model", default="",
                        help="Project name of the released model; defaults to "
                             "the final model (configs/ARGAvatar-Final.yaml)")
    parser.add_argument("--config", default="", help="Config path; overrides --model")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--max_render_height", type=int, default=1024)
    parser.add_argument("--export_motion_3d", action="store_true",
                        help="Also export posed Gaussian PLY sequences on the backend (default: video only)")
    parser.add_argument("--motion_chunk_frames", type=int, default=15,
                        help="Frames per streamed H.264 chunk (default: 15)")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Expose a public gradio.live link")
    parser.add_argument("--no_launch", action="store_true",
                        help="Build the model and the UI, then exit — the startup smoke test")
    parser.add_argument("--no_load", action="store_true",
                        help="Do not load the model pipeline; for UI layout debugging only")
    parser.add_argument("--keep_outputs", action="store_true",
                        help="Skip automatic cleanup of old workspace directories in expr/app")
    args = parser.parse_args()

    if args.motion_chunk_frames < 1:
        parser.error("--motion_chunk_frames must be positive")
    set_no_load(args.no_load)

    if not args.keep_outputs:
        cleaned = cleanup_work_dirs(WORK_ROOT, max_age_hours=24)
        if cleaned > 0:
            print(f"Cleaned up {cleaned} stale work directories under {WORK_ROOT}.")

    # ffmpeg writes every video the app produces, and extracts frames from uploads —
    # which --no_load mode still does.
    have_ffmpeg = shutil.which("ffmpeg") is not None

    if not args.no_load:
        config = args.config or (f"configs/{args.model}.yaml" if args.model else DEFAULT_CONFIG_PATH)
        if not Path(config).is_file():
            raise SystemExit(f"config not found: {config}")
        if not have_ffmpeg:
            raise SystemExit("ffmpeg is required to write the animation videos")

        print(f"Loading {config} …")
        pipeline = ARGAvatarPipeline.build(
            PipelineOptions(
                config=config,
                checkpoint_path=args.checkpoint_path,
                max_render_height=args.max_render_height,
                export_motion_3d=args.export_motion_3d,
                motion_chunk_frames=args.motion_chunk_frames,
            )
        )
        set_pipeline(pipeline)
        print(f"Ready. Checkpoint: {pipeline.options.checkpoint_path}")
    else:
        print("Running in --no_load mode: model pipeline loading is skipped for UI debugging.")
        if not have_ffmpeg:
            print("WARNING: ffmpeg not found — uploading a video will fail at frame extraction.")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    ui = build_ui()
    if args.no_launch:
        print("UI built; --no_launch given, exiting without serving.")
        return
    ui.queue(default_concurrency_limit=CONCURRENCY_LIMIT)
    ui.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        allowed_paths=[str(ASSETS.resolve()), str(WORK_ROOT.resolve())],
        app_kwargs={"middleware": [Middleware(
            SessionFileMiddleware, access=ui.file_access, public_config=ui.config,
        )]},
        max_file_size="20mb",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=APP_CSS,
        js=APP_JS,
    )


if __name__ == "__main__":
    main()
