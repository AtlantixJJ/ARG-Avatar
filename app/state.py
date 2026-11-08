"""Application state models and enumerations for the ARGAvatar demo."""
from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any


class MotionSource(str, Enum):
    """Supported motion sources for the Animate tab."""

    BUNDLED = "Bundled Preset"
    CUSTOM = "Custom Video Upload"


class InputMode(str, Enum):
    """Supported input modes for the Reconstruct tab's upload box."""

    IMAGES = "Images"
    VIDEO = "Video"


@dataclasses.dataclass
class AppState:
    """Per-session application state tracking reconstructed avatar and active selections."""

    avatar: Any = None
    work_dir: Path | None = None
    input_video_path: str | None = None
    active_views: list[str] = dataclasses.field(default_factory=list)
    # id (into the session's user_subjects_state list) of the Subject Library entry that
    # the image-mode upload gallery is currently backing, so further drag-and-drop uploads
    # extend that entry instead of spawning a new one each time. None when the gallery's
    # content did not originate from an in-progress upload (e.g. a fresh session, or a
    # Subject Library selection that only toggles active_views without owning a group).
    active_subject_id: int | None = None
    # Every frame ffmpeg extracted from the current video-mode upload (not just the
    # evenly-spaced subset shown by default), so "Auto-Select Frames" can re-pick from the
    # full set via LPIPS-feature farthest point sampling without re-running ffmpeg.
    all_extracted_frames: list[str] = dataclasses.field(default_factory=list)
