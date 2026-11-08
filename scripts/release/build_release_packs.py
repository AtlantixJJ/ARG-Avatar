#!/usr/bin/env python3
"""Maintainer tool: build the distributable NeuMan detection pack from the live root.

Users do not run this -- `scripts/download_evaluation/` fetches what it produces. This is
how that archive is made, kept in the repo so the pack is reproducible rather than an
artefact someone once tarred by hand.

    python scripts/release/build_release_packs.py
    python scripts/release/build_release_packs.py --source /data1/datasets/AnimatableHuman

One archive lands in `expr/release_pack/`, with a SHA256 manifest:

    argavatar-neuman-detections.tar.gz      refined masks, sam3db.pth, scene.pth, hierarchy

**It contains only files this project produced.** The file list comes from
`scripts/download_evaluation/neuman_manifest.py`, and the build refuses to add anything
outside it -- Apple's NeuMan imagery must never end up in a pack we publish. The check is
not advisory: an unexpected path aborts the build.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, ".")

from scripts.download_evaluation.neuman_manifest import (
    DETECTION_ARCHIVE,
    OURS_DIRS,
    OURS_FILES,
    OURS_ROOT_FILES,
    SUBJECTS,
)

DEFAULT_SOURCE = Path("data/AnimatableHuman")
OUT_DIR = Path("expr/release_pack")

# Anything under a subject that is Apple's, or derived from Apple's frames. Present in the
# live root; must never reach a pack. `images/` and `input_images/` are Apple's frames;
# the rest are Apple's own annotations shipped in dataset.zip.
FORBIDDEN = {
    "images", "input_images", "sparse", "densepose", "depth_maps", "mono_depth",
    "keypoints", "segmentations", "smpl_pred", "smplx_optimized",
    "smpl_output_optimized.pkl", "smpl_output_romp.pkl", "alignments.npy",
    "masks_neuman_orig", "input_masks_neuman_orig",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard(arcname: str) -> None:
    """Abort if an archive path touches anything that is not ours to redistribute."""
    for part in Path(arcname).parts:
        if part in FORBIDDEN:
            raise SystemExit(
                f"REFUSING to pack {arcname!r}: {part!r} is upstream NeuMan data, not ours.\n"
                "Only files this project produced may be distributed; users fetch Apple's\n"
                "release themselves via prepare_neuman.py --download."
            )


def collect_detection_files(root: Path) -> list:
    """(source path, archive name) for everything in the detection pack."""
    entries = []
    for filename in OURS_ROOT_FILES:
        path = root / filename
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        entries.append((path, filename))

    for subject in SUBJECTS:
        for dirname in OURS_DIRS:
            directory = root / subject / dirname
            if not directory.is_dir():
                raise SystemExit(f"missing {directory}")
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    entries.append((path, str(path.relative_to(root))))
        for filename in OURS_FILES:
            path = root / subject / filename
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            entries.append((path, str(path.relative_to(root))))
    return entries


def build(entries: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    def metadata(info):
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        info.mode = 0o644
        return info

    # Stable metadata makes both the archive and checksum reproducible.
    with out_path.open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for src, arcname in sorted(entries, key=lambda entry: entry[1]):
                _guard(arcname)
                if src.is_symlink():
                    raise SystemExit(f"REFUSING to pack symlink: {src}")
                tar.add(src, arcname=arcname, filter=metadata)
                total_bytes += src.stat().st_size

    manifest = out_path.with_suffix(out_path.suffix + ".sha256")
    manifest.write_text(f"{_sha256(out_path)}  {out_path.name}\n")
    print(
        f"[make] {out_path}  {len(entries)} files, "
        f"{total_bytes / 1e6:.1f} MB raw -> {out_path.stat().st_size / 1e6:.1f} MB packed"
    )
    print(f"[make] {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="AnimatableHuman root holding Rendering/NeuMan")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    neuman_root = args.source / "Rendering" / "NeuMan"

    build(collect_detection_files(neuman_root), args.out_dir / DETECTION_ARCHIVE)

    print(
        f"\nPack is in {args.out_dir}. To publish, upload the archive and its .sha256\n"
        "file to the host, then point users at it:\n"
        f"    export ARGAVATAR_DATA_URL=https://<host>/<path>\n"
        "or, for a HuggingFace dataset repo:\n"
        f"    hf upload <org>/<repo> {args.out_dir} . --repo-type dataset\n"
        "    export ARGAVATAR_DATA_HF_REPO=<org>/<repo>"
    )


if __name__ == "__main__":
    main()
