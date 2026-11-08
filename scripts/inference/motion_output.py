"""Backend motion artifacts: bounded video chunk encoding and optional Gaussian sequences."""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


class GaussianSequenceWriter:
    def __init__(self, out_dir: Path, fps: float):
        self.directory = Path(tempfile.mkdtemp(prefix="gaussian-sequence-", dir=out_dir))
        self.fps = fps
        self.frames = []

    def write(self, gaussians, source_frame) -> None:
        # Lazy import avoids the pipeline/exporter module dependency cycle.
        from scripts.inference.pipeline import write_gaussian_ply
        filename = f"{len(self.frames) + 1:06d}.ply"
        # Keep world-space motion: centering every frame would erase root translation.
        write_gaussian_ply(gaussians, str(self.directory / filename), centerize=False)
        self.frames.append({"file": filename, "source_frame": str(source_frame)})

    def finish(self) -> str:
        manifest = self.directory / "manifest.json"
        manifest.write_text(json.dumps({
            "version": 1, "format": "3dgs-ply", "fps": self.fps,
            "coordinate_system": "COLMAP/OpenCV", "centerized": False,
            "frame_count": len(self.frames), "frames": self.frames,
        }, indent=2) + "\n")
        return str(manifest)


class VideoChunkEncoder:
    """Encode while the GPU renders, with at most two chunks queued in host memory."""
    def __init__(self, out_dir: Path, fps: float, chunk_frames: int):
        if fps <= 0 or chunk_frames < 1:
            raise ValueError("Video FPS and chunk_frames must be positive")
        self.directory = Path(tempfile.mkdtemp(prefix="video-chunks-", dir=out_dir))
        self.fps, self.chunk_frames = fps, chunk_frames
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motion-encoder")
        self.pending = deque()
        self.frames = []
        self.paths = []
        self.shape = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _submit(self, completed: int) -> None:
        path = self.directory / f"{len(self.paths):06d}.ts"
        self.paths.append(path)
        self.pending.append((completed, self.executor.submit(self._encode, self.frames, path)))
        self.frames = []

    def _encode(self, frames: list[np.ndarray], path: Path) -> str:
        height, width = frames[0].shape[:2]
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-threads", "2",
            "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-g", str(len(frames)), "-bf", "0", "-sc_threshold", "0",
            "-f", "mpegts", "-muxdelay", "0", "-muxpreload", "0", str(path),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                for frame in frames:
                    process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass  # Report ffmpeg's actual error below.
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            error = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise RuntimeError(f"Motion video encoding failed: {error.strip()}")
        # Very short/low-detail clips can contain too few TS packets for ffprobe's
        # format detection (used by Gradio). Null packets add no frames or duration.
        minimum_size = 32 * 188
        missing = max(0, (minimum_size - path.stat().st_size + 187) // 188)
        if missing:
            with path.open("ab") as output:
                output.write((b"\x47\x1f\xff\x10" + b"\xff" * 184) * missing)
        return str(path)

    def add(self, frame: np.ndarray, completed: int):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("Motion frames must be uint8 RGB or RGBA images")
        shape = frame.shape[:2]
        if self.shape is not None and self.shape != shape:
            raise ValueError("Motion frame dimensions changed during the clip")
        self.shape = shape
        self.frames.append(np.ascontiguousarray(frame[..., :3]))
        if len(self.frames) >= self.chunk_frames:
            self._submit(completed)
        while self.pending and (self.pending[0][1].done() or len(self.pending) >= 2):
            index, future = self.pending.popleft()
            yield index, future.result()

    def finish(self, completed: int):
        if self.frames:
            self._submit(completed)
        while self.pending:
            index, future = self.pending.popleft()
            yield index, future.result()

    def combine(self, out_path: Path) -> str:
        if not self.paths:
            raise ValueError("Cannot encode an empty motion")
        listing = self.directory / "concat.txt"
        listing.write_text("".join("file '" + str(path.resolve()).replace("'", "'\\''") + "'\n"
                                   for path in self.paths))
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out_path),
        ], check=True)
        return str(out_path)
