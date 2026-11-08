#!/usr/bin/env python3
"""Assemble the NeuMan benchmark root the evaluation reads.

NeuMan is the only benchmark this release evaluates on, and its imagery is Apple's, not
ours. So the root is assembled from two sources:

  * **Apple's `dataset.zip`** (~2.1 GB) -- the frames. Downloaded from Apple, unzipped.
  * **Detections** -- refined masks, SAM-3D-Body/MHR poses (`sam3db.pth`), scene cameras,
    and the validation hierarchy. Nothing here is Apple's, and there are two ways to get
    them: `--detections` fetches our precomputed pack (what the release numbers were
    produced on); `--generate` derives them locally from Apple's images using this
    project's own SAM-3D-Body estimator (masks are a naive segmentation inversion, not
    the GSAM2-refined ones in the precomputed pack, and poses are freshly estimated so
    they won't bit-match the release numbers -- but no extra download is needed and the
    whole root is reproducible from Apple's archive plus code in this repo).

Fetch the precomputed pack (matches the released numbers):

    python scripts/download_evaluation/prepare_neuman.py --all

Or regenerate detections locally, entirely from Apple's official download link:

    python scripts/download_evaluation/prepare_neuman.py --from_scratch --gpus all

or one step at a time, which is what you want if a download failed halfway:

    python scripts/download_evaluation/prepare_neuman.py --download
    python scripts/download_evaluation/prepare_neuman.py --unzip
    python scripts/download_evaluation/prepare_neuman.py --detections   # or --generate
    python scripts/download_evaluation/prepare_neuman.py --verify

The result lands at `data/AnimatableHuman/Rendering/NeuMan`, which is what
`configs/_shared/data_val_list.yaml` points at. `--verify` is the one to run if you are
unsure whether an existing root is complete: it checks every frame named by
`expr/val_lists/Neuman_val_list.yaml` against what is actually on disk, and loads each
`sam3db.pth` to confirm it covers those frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, ".")

from scripts.download_evaluation.neuman_manifest import (
    DERIVED_DIRS,
    BUNDLED_DETECTIONS,
    DETECTION_ARCHIVE,
    NEUMAN_URL,
    OFFICIAL_DIRS,
    OURS_DIRS,
    OURS_FILES,
    OURS_ROOT_FILES,
    SUBJECTS,
)

DEFAULT_ROOT = Path("data/AnimatableHuman/Rendering/NeuMan")
DEFAULT_RAW = Path("data/neuman_raw")
VAL_LIST = Path("expr/val_lists/Neuman_val_list.yaml")


# --- steps -------------------------------------------------------------------


def step_download(raw_dir: Path, force: bool) -> Path:
    """Fetch Apple's dataset.zip. Their terms apply; see github.com/apple/ml-neuman."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "dataset.zip"
    if archive.is_file() and not force:
        print(f"[skip] {archive} already present ({archive.stat().st_size / 1e9:.1f} GB)")
        return archive
    print(f"[get ] {NEUMAN_URL}\n       -> {archive}  (~2.1 GB, this takes a while)")
    part = archive.with_suffix(".zip.part")
    subprocess.run(["curl", "-fL", "--progress-bar", "-o", str(part), NEUMAN_URL], check=True)
    part.rename(archive)
    return archive


def step_unzip(raw_dir: Path, root: Path, force: bool) -> None:
    """Unzip Apple's archive and copy the frames into the prepared root."""
    archive = raw_dir / "dataset.zip"
    if not archive.is_file():
        raise SystemExit(f"{archive} not found -- run --download first")

    extracted = raw_dir / "dataset"
    if not extracted.is_dir() or force:
        print(f"[unzip] {archive} -> {raw_dir}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
    else:
        print(f"[skip] {extracted} already extracted")

    if not extracted.is_dir():
        # Some archives nest one level deeper; find the directory holding the subjects.
        candidates = [p for p in raw_dir.rglob("bike") if p.is_dir()]
        if not candidates:
            raise SystemExit(f"could not find the subject folders under {raw_dir}")
        extracted = candidates[0].parent
        print(f"[note] using {extracted} as the extracted root")

    root.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        src_subject = extracted / subject
        if not src_subject.is_dir():
            raise SystemExit(f"{src_subject} missing from Apple's archive")
        for dirname in OFFICIAL_DIRS:
            src, dst = src_subject / dirname, root / subject / dirname
            if dst.is_dir() and not force:
                print(f"[skip] {dst} already present ({len(list(dst.iterdir()))} files)")
                continue
            print(f"[copy] {src} -> {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)


def step_detections(root: Path, archive_path: str, force: bool) -> None:
    """Unpack our detection pack over the root, then derive the input_* folders."""
    root.mkdir(parents=True, exist_ok=True)

    if archive_path:
        source = Path(archive_path)
        if not source.is_file():
            raise SystemExit(f"--detections_archive not found: {source}")
        _extract_detections(source, root, force)
    else:
        base_url = os.environ.get("ARGAVATAR_DATA_URL", "").rstrip("/")
        hf_repo = os.environ.get("ARGAVATAR_DATA_HF_REPO", "")
        if not base_url and not hf_repo:
            if not BUNDLED_DETECTIONS.is_file():
                raise SystemExit(f"Bundled detection pack is missing: {BUNDLED_DETECTIONS}")
            _extract_detections(BUNDLED_DETECTIONS, root, force)
        else:
            _download_detections(root, base_url, hf_repo, force)

    derive_input_folders(root, force)


def _download_detections(root: Path, base_url: str, hf_repo: str, force: bool) -> None:
    """Optional download overrides; a normal checkout uses the bundled pack."""
    with tempfile.TemporaryDirectory(prefix="neuman-pack-") as tmp:
        local = Path(tmp) / DETECTION_ARCHIVE
        if hf_repo:
            from huggingface_hub import hf_hub_download

            print(f"[get ] {hf_repo}:{DETECTION_ARCHIVE}")
            shutil.copyfile(
                hf_hub_download(repo_id=hf_repo, filename=DETECTION_ARCHIVE, repo_type="dataset"),
                local,
            )
        else:
            url = f"{base_url}/{DETECTION_ARCHIVE}"
            print(f"[get ] {url}")
            subprocess.run(["curl", "-fL", "--progress-bar", "-o", str(local), url], check=True)
        _extract_detections(local, root, force)


def _extract_detections(archive: Path, root: Path, force: bool) -> None:
    manifest = archive.with_suffix(archive.suffix + ".sha256")
    if archive == BUNDLED_DETECTIONS or manifest.is_file():
        expected = manifest.read_text().split()[0]
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if expected != actual:
            raise SystemExit(f"Detection pack checksum mismatch: {archive}")
    print(f"[unpack] {archive} -> {root}")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            target = root / member.name
            if target.exists() and not force:
                continue
            tar.extract(member, path=root, filter="data")
    print(f"[unpack] {len(members)} entries")


def step_generate(root: Path, raw_dir: Path, val_list: Path, args) -> None:
    """Derive detections locally instead of fetching the precomputed pack.

    Writes `val_hierarchy.json` / `hierarchical_mapping.json` (trivial for NeuMan: one
    subject per scene, so `{scene: {"0": [scene]}}`), inverts `segmentations/` from the
    raw extracted archive into `masks/`, runs this project's own SAM-3D-Body estimator
    over every frame the val list references to produce `sam3db.pth` + `scene.pth`, then
    derives `input_images/` / `input_masks/` the same way `--detections` does.
    """
    from scripts.download_evaluation.generate_neuman_masks import invert_scene

    root.mkdir(parents=True, exist_ok=True)

    hierarchy = {s: {"0": [s]} for s in SUBJECTS}
    for filename in OURS_ROOT_FILES:
        path = root / filename
        if path.is_file() and not args.force:
            print(f"[skip] {path} already present")
            continue
        with open(path, "w") as f:
            json.dump(hierarchy, f, indent=2)
        print(f"[write] {path}")

    extracted = raw_dir / "dataset"
    for scene in SUBJECTS:
        seg_dir = extracted / scene / "segmentations"
        if not seg_dir.is_dir():
            raise SystemExit(f"{seg_dir} missing -- run --unzip first")
        n = invert_scene(seg_dir, root / scene / "masks", args.force)
        print(f"[masks] {scene}: wrote {n}")

    derive_input_folders(root, args.force)

    pose_cmd = [
        sys.executable, "scripts/download_evaluation/generate_neuman_poses.py",
        "--root", str(root), "--val_list", str(val_list),
        "--config", args.config, "--checkpoint_path", args.checkpoint_path,
        "--gpus", args.gpus,
    ]
    pose_cmd.append("--force" if args.force else "--skip_existing")
    print(" ".join(pose_cmd))
    subprocess.run(pose_cmd, check=True)


def derive_input_folders(root: Path, force: bool) -> None:
    """Recreate `input_images/` and `input_masks/` from the val list's input frames.

    These are exact copies of the corresponding `images/` and `masks/` entries, so they
    are derived rather than shipped -- it would otherwise mean distributing Apple's
    frames a second time.
    """
    items = load_val_list()["items"]
    per_subject: dict = {}
    for entry in items.values():
        for rel in entry.get("input_images", []):
            subject, _, filename = Path(rel).parts[-3:]
            per_subject.setdefault(subject, set()).add(filename)

    for subject, filenames in sorted(per_subject.items()):
        for src_dir, dst_dir in (("images", "input_images"), ("masks", "input_masks")):
            dst = root / subject / dst_dir
            dst.mkdir(parents=True, exist_ok=True)
            copied = 0
            for filename in sorted(filenames):
                src_path = root / subject / src_dir / filename
                dst_path = dst / filename
                if not src_path.is_file():
                    raise SystemExit(
                        f"cannot derive {dst_path}: {src_path} is missing. "
                        f"Run --{'unzip' if src_dir == 'images' else 'detections'} first."
                    )
                if dst_path.exists() and not force:
                    continue
                shutil.copyfile(src_path, dst_path)
                copied += 1
            if copied:
                print(f"[derive] {dst} ({copied} files)")


# --- verification -------------------------------------------------------------


def load_val_list() -> dict:
    import yaml

    if not VAL_LIST.is_file():
        raise SystemExit(f"{VAL_LIST} not found -- run from the repository root")
    with open(VAL_LIST) as handle:
        return yaml.safe_load(handle)


def step_verify(root: Path) -> int:
    """Check the assembled root against the val list. Returns a process exit code."""
    import torch

    problems = []

    if not root.is_dir():
        print(f"FAIL: {root} does not exist", file=sys.stderr)
        return 1

    for filename in OURS_ROOT_FILES:
        if not (root / filename).is_file():
            problems.append(f"missing {root / filename}")

    hierarchy_path = root / "val_hierarchy.json"
    if hierarchy_path.is_file():
        with open(hierarchy_path) as handle:
            hierarchy = json.load(handle)
        missing_subjects = sorted(set(SUBJECTS) - set(hierarchy))
        if missing_subjects:
            problems.append(f"val_hierarchy.json omits: {', '.join(missing_subjects)}")

    val_list = load_val_list()
    items = val_list["items"]

    # Every frame the val list names must be on disk, with a mask beside it.
    referenced: dict = {}
    for key, entry in items.items():
        for field in ("input_images", "images"):
            for rel in entry.get(field, []):
                referenced.setdefault(Path(rel).parts[0], []).append(rel)

    total_frames = 0
    for subject in SUBJECTS:
        rels = referenced.get(subject, [])
        total_frames += len(rels)
        missing_images = [r for r in rels if not (root / r).is_file()]
        if missing_images:
            problems.append(
                f"{subject}: {len(missing_images)}/{len(rels)} frames missing, "
                f"e.g. {missing_images[0]} (run --download --unzip)"
            )
        missing_masks = [
            r for r in rels
            if not (root / r.replace("/images/", "/masks/")).is_file()
        ]
        if missing_masks:
            problems.append(
                f"{subject}: {len(missing_masks)}/{len(rels)} masks missing, "
                f"e.g. {missing_masks[0].replace('/images/', '/masks/')} (run --detections)"
            )

        for dirname in OURS_DIRS + DERIVED_DIRS:
            if not (root / subject / dirname).is_dir():
                problems.append(f"{subject}: missing {dirname}/")
        for filename in OURS_FILES:
            if not (root / subject / filename).is_file():
                problems.append(f"{subject}: missing {filename} (run --detections)")

        # The detections must actually cover the frames the val list evaluates.
        sam3db_path = root / subject / "sam3db.pth"
        if sam3db_path.is_file():
            try:
                sam3db = torch.load(sam3db_path, map_location="cpu", weights_only=True)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{subject}: sam3db.pth failed to load: {type(exc).__name__}: {exc}")
            else:
                wanted = {f"images/{Path(r).name}" for r in rels}
                uncovered = sorted(wanted - set(sam3db))
                if uncovered:
                    problems.append(
                        f"{subject}: sam3db.pth covers {len(wanted) - len(uncovered)}/{len(wanted)} "
                        f"evaluated frames, e.g. {uncovered[0]} is absent"
                    )
                else:
                    print(f"detections OK: {subject} ({len(wanted)} frames)")

    if problems:
        print(f"\n{len(problems)} problem(s) with {root}:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(
        f"\nOK -- {root} is complete: {len(SUBJECTS)} subjects, {total_frames} evaluated frames, "
        "every one with an image, a mask and an MHR detection."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Apple's NeuMan terms govern the imagery: https://github.com/apple/ml-neuman",
    )
    parser.add_argument("--all", action="store_true", help="--download --unzip --detections --verify")
    parser.add_argument("--from_scratch", action="store_true",
                        help="--download --unzip --generate --verify (no precomputed pack, "
                             "just Apple's official download + this project's own code)")
    parser.add_argument("--download", action="store_true", help="Fetch Apple's dataset.zip")
    parser.add_argument("--unzip", action="store_true", help="Unzip it and copy the frames in")
    parser.add_argument("--detections", action="store_true", help="Unpack our detection pack")
    parser.add_argument("--generate", action="store_true",
                        help="Derive detections locally instead of fetching the pack "
                             "(naive-invert masks, run SAM-3D-Body for poses)")
    parser.add_argument("--verify", action="store_true", help="Check the assembled root")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Where to assemble the benchmark")
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW,
                        help="Where Apple's zip is downloaded and unpacked")
    parser.add_argument("--val_list", type=Path, default=VAL_LIST,
                        help="Val list --generate reads frame names from")
    parser.add_argument("--detections_archive", default="",
                        help="Use a local copy of the detection pack instead of downloading")
    parser.add_argument("--config", default="configs/ARGAvatar-Final.yaml",
                        help="--generate: model config for the SAM-3D-Body estimator")
    parser.add_argument("--checkpoint_path", default="checkpoints/ARGAvatar-Final.pth",
                        help="--generate: checkpoint for the SAM-3D-Body estimator")
    parser.add_argument("--gpus", default="all", help="--generate: GPUs to run pose estimation on")
    parser.add_argument("--force", action="store_true", help="Redo steps whose output exists")
    args = parser.parse_args()

    if not any((args.all, args.from_scratch, args.download, args.unzip,
                args.detections, args.generate, args.verify)):
        parser.error("choose at least one step (or --all / --from_scratch)")

    # expr/val_lists/Neuman_val_list.yaml pins `data_root`, and SAM3DImageDataset refuses
    # to load a list whose data_root does not resolve to the dataset's root_dir. The check
    # compares realpaths, so a symlink is the supported way to keep the data elsewhere.
    if args.root.resolve() != DEFAULT_ROOT.resolve():
        print(
            f"note: --root is {args.root}, not the {DEFAULT_ROOT} that\n"
            f"      expr/val_lists/Neuman_val_list.yaml pins. Evaluation will refuse this root\n"
            f"      unless the default path resolves to it:\n"
            f"          mkdir -p {DEFAULT_ROOT.parent}\n"
            f"          ln -s {args.root.resolve()} {DEFAULT_ROOT}\n"
        )

    if args.all or args.from_scratch or args.download:
        step_download(args.raw_dir, args.force)
    if args.all or args.from_scratch or args.unzip:
        step_unzip(args.raw_dir, args.root, args.force)
    if args.all or args.detections:
        step_detections(args.root, args.detections_archive, args.force)
    if args.from_scratch or args.generate:
        step_generate(args.root, args.raw_dir, args.val_list, args)
    if args.all or args.from_scratch or args.verify:
        return step_verify(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
