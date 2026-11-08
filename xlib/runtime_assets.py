"""Pinned files needed to construct the released model without network access."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DINO_REPO = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
DINO_CONFIG_REVISION = "10d30274b4d445111e2d5bf75ac93bbd94db274b"
DINO_CODE_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
DINO_SOURCE_DIR = ROOT / "pretrained_models" / f"dinov3-{DINO_CODE_REVISION}"


def dinov3_source_dir() -> Path:
    if not (DINO_SOURCE_DIR / "hubconf.py").is_file():
        raise FileNotFoundError(
            "Pinned DINOv3 source is missing. Run bash scripts/download_weights.sh "
            "before constructing the model."
        )
    return DINO_SOURCE_DIR
