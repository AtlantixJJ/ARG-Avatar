#!/usr/bin/env python3
"""Offline validation of an installed ARGAvatar release."""
from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The one released model, by project name. `checkpoints/<name>.pth` is the canonical
# path it ships at; the `expr/<name>/PT_<step>.pth` mirror that
# `manager.py::resolve_jobs` globs is a hardlink to the same inode, created by
# `download_weights.sh`. Verify against the canonical path so the gate does not depend
# on that side effect having run -- `verify_mirrors()` checks the mirrors separately.
MODEL_NAMES = (
    "ARGAvatar-Final",
)
MODELS = tuple(
    (f"configs/{name}.yaml", f"checkpoints/{name}.pth") for name in MODEL_NAMES
)
ENTRYPOINTS = (
    "evaluation.argavatar.eval_argavatar",
    "scripts.inference.pipeline",
    "scripts.inference.infer_argavatar",
    "scripts.inference.estimate_motion_mhr",
    "scripts.test.test_app_case",
    "scripts.test.test_app_video",
    "scripts.download_evaluation.prepare_neuman",
    "app.__main__",
    "xlib.model.arg_net_uv",
    "xlib.model.arg_net_uv_token_cascade",
)

# Every user-facing entrypoint must be named in the user docs, so a script cannot ship
# undocumented. Maintainer-only tooling under scripts/release/ is deliberately excluded.
DOCUMENTED_ENTRYPOINTS = (
    "app/__main__.py",
    "scripts/inference/infer_argavatar.py",
    "scripts/inference/estimate_motion_mhr.py",
    "scripts/download_weights.sh",
    "scripts/download_runtime_assets.py",
    "scripts/install_argavatar_env.sh",
    "scripts/download_evaluation/prepare_neuman.py",
    "scripts/download_evaluation/generate_neuman_masks.py",
    "scripts/download_evaluation/generate_neuman_poses.py",
    "evaluation/argavatar/eval_argavatar.py",
    "evaluation/scripts/collect_eval_results.py",
)
DOC_FILES = ("README.md", "evaluation/README.md")
# How the docs may name an entrypoint other than by its path.
DOC_ALIASES = {"app/__main__.py": "python -m app"}

# Developer tooling (release check, smoke/parity/browser tests) is documented for
# maintainers in CLAUDE.md rather than in the user README.
MAINTAINER_ENTRYPOINTS = (
    "scripts/verify_release.py",
    "scripts/test/test_app_case.py",
    "scripts/test/test_app_video.py",
    "scripts/test/check_artifact_parity.py",
    "scripts/test/check_eval_parity.py",
    "scripts/test/test_app_mobile.py",
    "scripts/test/test_app_streaming.py",
)
MAINTAINER_DOC_FILES = ("CLAUDE.md",)


def verify_imports() -> None:
    for module in ENTRYPOINTS:
        importlib.import_module(module)
        print(f"import OK: {module}")


def _iter_class_names(node, path: str):
    """Yield (yaml_path, class_name) for every `class_name:` string in a config tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "class_name" and isinstance(value, str):
                yield f"{path}.class_name", value
            else:
                yield from _iter_class_names(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _iter_class_names(value, f"{path}[{index}]")


def verify_configs() -> None:
    """Resolve every `class_name:` string in configs/**/*.yaml, not just model.class_name.

    Data class names are resolved by `construct_class_by_name` at runtime only, so a
    missing dataset module is invisible to both imports and the model gate.
    """
    import yaml

    from xlib.utils import get_obj_by_name, load_config

    for config_path, checkpoint_path in MODELS:
        load_config(str(ROOT / config_path))
        assert (ROOT / checkpoint_path).is_file(), checkpoint_path

    failures = []
    for config_file in sorted((ROOT / "configs").rglob("*.yaml")):
        rel = config_file.relative_to(ROOT)
        with open(config_file) as handle:
            tree = yaml.safe_load(handle)
        for yaml_path, class_name in _iter_class_names(tree, ""):
            try:
                cls = get_obj_by_name(class_name)
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                failures.append(f"{rel}:{yaml_path} -> {class_name}: {type(exc).__name__}: {exc}")
                print(f"config FAIL: {rel}:{yaml_path} -> {class_name}")
            else:
                print(f"config OK: {rel}:{yaml_path} -> {cls.__module__}.{cls.__name__}")
    if failures:
        print("\nunresolvable class_name entries:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        sys.exit(1)


def verify_checkpoints() -> None:
    import torch

    for _, checkpoint_path in MODELS:
        checkpoint = torch.load(ROOT / checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        assert isinstance(state_dict, dict) and state_dict, checkpoint_path
        print(f"checkpoint OK: {checkpoint_path} ({len(state_dict)} tensors)")


def verify_entrypoints() -> None:
    """Every shipped entrypoint exists on disk and is named in the documentation.

    Catches the two ways an entrypoint goes stale: documented but deleted, and shipped
    but never written up. Maintainer-only tooling under scripts/release/ is out of scope.
    """
    failures = []
    for entrypoints, doc_files in ((DOCUMENTED_ENTRYPOINTS, DOC_FILES),
                                   (MAINTAINER_ENTRYPOINTS, MAINTAINER_DOC_FILES)):
        docs = "\n".join((ROOT / name).read_text() for name in doc_files if (ROOT / name).is_file())
        for rel in entrypoints:
            if not (ROOT / rel).is_file():
                failures.append(f"{rel}: listed as an entrypoint but not on disk")
                print(f"entrypoint MISSING: {rel}")
            elif rel not in docs and DOC_ALIASES.get(rel, rel) not in docs:
                failures.append(f"{rel}: exists but is not mentioned in {' or '.join(doc_files)}")
                print(f"entrypoint UNDOCUMENTED: {rel}")
            else:
                print(f"entrypoint OK: {rel}")

    # The other direction: an executable script nobody listed.
    known = set(DOCUMENTED_ENTRYPOINTS) | set(MAINTAINER_ENTRYPOINTS)
    for pattern in ("scripts/**/*.py", "scripts/**/*.sh", "evaluation/**/run_*.sh"):
        for path in sorted(ROOT.glob(pattern)):
            rel = str(path.relative_to(ROOT))
            if rel in known or rel.startswith("scripts/release/"):
                continue
            # Only files with a __main__ / shebang entry are entrypoints.
            text = path.read_text(errors="ignore")
            if '__name__ == "__main__"' in text or text.startswith("#!"):
                failures.append(f"{rel}: looks like an entrypoint but is in no inventory")
                print(f"entrypoint UNLISTED: {rel}")

    if failures:
        print("\nentrypoint inventory problems:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        sys.exit(1)


def verify_mirrors() -> None:
    """Check the expr/<name>/PT_<step>.pth mirrors resolve_jobs globs for.

    `evaluation/argavatar/manager.py::resolve_jobs` resolves a bare model name by globbing
    `expr/<name>/PT_*.pth`, so the release sweep needs these to exist. They are hardlinks
    to `checkpoints/<name>.pth` created by `download_weights.sh`; a mirror that is a
    separate inode means the two can silently drift apart.
    """
    failures = []
    for name in MODEL_NAMES:
        canonical = ROOT / "checkpoints" / f"{name}.pth"
        mirrors = sorted((ROOT / "expr" / name).glob("PT_*.pth"))
        if not mirrors:
            failures.append(f"{name}: no expr/{name}/PT_*.pth (run scripts/download_weights.sh)")
            print(f"mirror MISSING: expr/{name}/PT_*.pth")
            continue
        if len(mirrors) > 1:
            failures.append(f"{name}: {len(mirrors)} checkpoints in expr/{name}, resolve_jobs picks by mtime")
            print(f"mirror AMBIGUOUS: expr/{name} holds {len(mirrors)} checkpoints")
            continue
        mirror = mirrors[0]
        same_inode = canonical.is_file() and mirror.stat().st_ino == canonical.stat().st_ino
        note = "hardlink" if same_inode else "SEPARATE COPY"
        print(f"mirror OK: {mirror.relative_to(ROOT)} ({note})")
    if failures:
        print("\nbroken checkpoint mirrors:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        sys.exit(1)


def verify_model_checkpoints() -> None:
    """Instantiate each released architecture and strictly verify its weights."""
    import torch

    from xlib.render.gaussian_splatting import set_gaussian_type
    from xlib.train.train_helper import resolve_sam3d_cfg, setup_model_sam3d
    from xlib.utils import load_config

    for config_path, checkpoint_path in MODELS:
        config = load_config(str(ROOT / config_path))
        config.model.load_checkpoint = ""
        set_gaussian_type(config.model.get("gaussian_type", "3d"))
        sam3d_cfg = resolve_sam3d_cfg(config, str(ROOT / config_path))
        model = setup_model_sam3d(config, sam3d_cfg, device=torch.device("cpu"))
        missing, unexpected = model.load_checkpoint(str(ROOT / checkpoint_path))
        assert not missing, f"{checkpoint_path}: missing keys: {missing[:10]}"
        assert not unexpected, f"{checkpoint_path}: unexpected keys: {unexpected[:10]}"
        print(f"model/checkpoint OK: {config_path} -> {checkpoint_path}")
        del model
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imports", action="store_true")
    parser.add_argument("--configs", action="store_true")
    parser.add_argument("--checkpoints", action="store_true")
    parser.add_argument("--models", action="store_true", help="instantiate each model and strictly load its checkpoint")
    parser.add_argument("--mirrors", action="store_true", help="check the expr/<name>/PT_*.pth checkpoint mirrors resolve_jobs needs")
    parser.add_argument("--entrypoints", action="store_true", help="check every shipped entrypoint exists and is documented")
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error("choose at least one verification gate")
    os.chdir(ROOT)
    if args.imports:
        verify_imports()
    if args.configs:
        verify_configs()
    if args.checkpoints:
        verify_checkpoints()
    if args.entrypoints:
        verify_entrypoints()
    if args.mirrors:
        verify_mirrors()
    if args.models:
        verify_model_checkpoints()


if __name__ == "__main__":
    main()
