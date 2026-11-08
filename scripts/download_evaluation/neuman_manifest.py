"""What the NeuMan benchmark root is made of, and where each part comes from.

One place defines the split so the three tools that care cannot disagree:
`prepare_neuman.py` (assembles a root), `scripts/release/build_release_packs.py` (builds
the distributable pack), and the `--verify` gate (checks an assembled root).

The split matters legally as well as practically. NeuMan's imagery is Apple's, released
under their own terms and **not** redistributable here, so the pack we publish contains
only files this project produced. Everything else is fetched by the user from Apple.
"""
from __future__ import annotations
from pathlib import Path

SUBJECTS = ("bike", "citron", "jogging", "lab", "parkinglot", "seattle")

# Apple's release: https://github.com/apple/ml-neuman
NEUMAN_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/neuman/dataset.zip"
NEUMAN_ZIP_BYTES = 2_212_303_945  # ~2.1 GiB; advisory, checked only as a sanity hint

# --- from Apple's dataset.zip, copied into the prepared root unchanged -------------
# `images/` is byte-identical between Apple's release and the root the published numbers
# were produced on; it is the only official directory the evaluation reads.
OFFICIAL_DIRS = ("images",)

# --- produced by this project; this is what the detection pack contains -------------
# Per subject:
#   masks/       foreground masks, refined -- these differ from Apple's own masks, which
#                the prepared root keeps alongside as masks_neuman_orig/
#   sam3db.pth   SAM-3D-Body / MHR pose + camera per frame. The detections proper.
#   scene.pth    per-scene camera metadata
OURS_DIRS = ("masks",)
OURS_FILES = ("sam3db.pth", "scene.pth")

# At the root of the dataset, not per subject.
OURS_ROOT_FILES = ("val_hierarchy.json", "hierarchical_mapping.json")

# --- derived during preparation, shipped in neither pack ----------------------------
# `input_images/` and `input_masks/` are exact copies of the frames named as inputs by
# expr/val_lists/Neuman_val_list.yaml -- verified byte-identical to their images/ and
# masks/ counterparts -- so they are re-derived rather than downloaded twice.
# `path_index.npz` is a cache SAM3DImageDataset writes itself on first use.
DERIVED_DIRS = ("input_images", "input_masks")

# Present in the root the release numbers were produced on, read by nothing that ships
# here. Listed so `--verify` can say "absent, and that is expected" instead of staying
# silent about a difference from the original root.
UNUSED_IN_RELEASE = (
    "alignments.npy", "densepose", "depth_maps", "keypoints", "mono_depth",
    "mhr2smplx.npy", "segmentations", "smpl_output_optimized.pkl",
    "smpl_output_romp.pkl", "smpl_pred", "smplx_optimized", "sparse",
    "masks_neuman_orig", "input_masks_neuman_orig",
)

DETECTION_ARCHIVE = "argavatar-neuman-detections.tar.gz"
BUNDLED_DETECTIONS = Path(__file__).resolve().parents[2] / "assets" / "benchmark" / DETECTION_ARCHIVE
