#!/usr/bin/env python3
"""Download the pinned DINOv3 config and source used by the release."""
from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xlib.runtime_assets import (
    DINO_CODE_REVISION, DINO_CONFIG_REVISION, DINO_REPO, DINO_SOURCE_DIR, ROOT,
)


def main() -> None:
    from huggingface_hub import hf_hub_download

    # ConvNeXt weights are in the ARGAvatar checkpoint; only its config is needed.
    path = hf_hub_download(
        repo_id=DINO_REPO, filename="config.json", revision=DINO_CONFIG_REVISION,
        cache_dir=ROOT / "pretrained_models",
    )
    print(f"[ready] DINOv3 config: {path}")
    if (DINO_SOURCE_DIR / "hubconf.py").is_file():
        print(f"[skip] {DINO_SOURCE_DIR.name} already present")
        return

    DINO_SOURCE_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DINO_SOURCE_DIR.parent, prefix=".dinov3-") as tmp:
        archive = Path(tmp) / "source.tar.gz"
        url = f"https://codeload.github.com/facebookresearch/dinov3/tar.gz/{DINO_CODE_REVISION}"
        print(f"[get ] {url}")
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        source = Path(tmp) / f"dinov3-{DINO_CODE_REVISION}"
        if not (source / "hubconf.py").is_file():
            raise RuntimeError("DINOv3 archive does not contain hubconf.py")
        source.rename(DINO_SOURCE_DIR)
    print(f"[ready] DINOv3 source: {DINO_SOURCE_DIR}")


if __name__ == "__main__":
    main()
