#!/usr/bin/env python3
"""Maintainer tool: upload an ARGAvatar checkpoint to a Hugging Face model repo.

Users do not run this. `scripts/download_weights.sh` downloads from the model
repo in `$ARGAVATAR_WEIGHTS_HF_REPO` (default: atlantix/ARGAvatar).

    python scripts/release/publish_checkpoint.py --repo_id atlantix/ARGAvatar
    python scripts/release/publish_checkpoint.py --checkpoint /path/to/ARGAvatar-Final.pth \\
        --repo_id atlantix/ARGAvatar --private

Requires `hf auth login` (or `HF_TOKEN`) first. Does not create or touch a
model card -- only the checkpoint file is uploaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CHECKPOINT = Path("checkpoints/ARGAvatar-Final.pth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"Checkpoint file to upload (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--repo_id", required=True,
                        help="Target HF model repo, e.g. atlantix/ARGAvatar")
    parser.add_argument("--path_in_repo", default=None,
                        help="Destination filename in the repo (default: the checkpoint's own filename)")
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist yet")
    parser.add_argument("--commit_message", default=None)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"missing {args.checkpoint}")

    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"[repo ] {repo_url}")

    path_in_repo = args.path_in_repo or args.checkpoint.name
    size_mb = args.checkpoint.stat().st_size / 1e6
    print(f"[push ] {args.checkpoint} ({size_mb:.1f} MB) -> {args.repo_id}:{path_in_repo}")

    commit_info = api.upload_file(
        path_or_fileobj=str(args.checkpoint),
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message or f"Upload {path_in_repo}",
    )
    print(f"[done ] {commit_info.commit_url}")


if __name__ == "__main__":
    main()
