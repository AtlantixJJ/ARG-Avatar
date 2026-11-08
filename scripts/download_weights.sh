#!/usr/bin/env bash
# Fetch every third-party weight ARGAvatar needs. None of these are committed to
# the repository -- they are large and separately licensed, so they are pulled
# from upstream into the exact paths the configs and code expect.
#
# Run this BEFORE scripts/install_argavatar_env.sh, whose verification step
# loads the SAM-3D-Body backbone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_SAM2=0
SKIP_SAM3DBODY=0
SKIP_DINOV3=0

usage() {
  cat <<'EOF'
Usage: scripts/download_weights.sh [--skip-sam2] [--skip-sam3dbody] [--skip-dinov3]

Downloads, with the path each consumer reads:

  thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt   (~857 MB, inference only)
      read by xlib/model/sam_3d_body/tools/build_sam.py as
      "<segmentor_path>/checkpoints/sam2.1_hiera_large.pt"

  pretrained_models/sam-3d-body-dinov3/               (~2.7 GB, required)
      read by configs/_shared/sam3dbody_default.yaml (CHECKPOINT_PATH,
      MHR_MODEL_PATH)

  pretrained_models/models--facebook--dinov3-...      (config only, required)
  pretrained_models/dinov3-<revision>/                (pinned source, required)
      Fetched by scripts/download_runtime_assets.py for offline model construction.

  checkpoints/ARGAvatar-Final.pth                     (~260 MB, required)
      ARGAvatar's own checkpoint, from the ARGAVATAR_WEIGHTS_HF_REPO model
      repo (default: atlantix/ARGAvatar); mirrored to
      expr/ARGAvatar-Final/PT_80000.pth for manager.py::resolve_jobs.
      Set ARGAVATAR_WEIGHTS_URL instead to fetch from a plain base URL.

Both third-party HuggingFace repositories above are GATED. Accept their terms
in a browser and authenticate first:

  hf auth login          # or: export HF_TOKEN=...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sam2) SKIP_SAM2=1; shift ;;
    --skip-sam3dbody) SKIP_SAM3DBODY=1; shift ;;
    --skip-dinov3) SKIP_DINOV3=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

have_hf() { python -c "import huggingface_hub" >/dev/null 2>&1; }

# --- SAM 2 segmentation checkpoint (inference only) ------------------------
# Path is dictated by build_sam.py, which appends "checkpoints/" to the
# --segmentor_path value (default: thirdparty/sam2). Do not relocate it.
SAM2_DEST="thirdparty/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
if [[ "$SKIP_SAM2" == 0 ]]; then
  if [[ -f "$SAM2_DEST" ]]; then
    echo "[skip] $SAM2_DEST already present"
  else
    echo "[get ] $SAM2_DEST"
    mkdir -p "$(dirname "$SAM2_DEST")"
    curl -fL --progress-bar -o "$SAM2_DEST.part" "$SAM2_URL"
    mv "$SAM2_DEST.part" "$SAM2_DEST"
  fi
fi

# --- SAM 3D Body backbone (gated) ------------------------------------------
if [[ "$SKIP_SAM3DBODY" == 0 ]]; then
  if [[ -f "pretrained_models/sam-3d-body-dinov3/model.ckpt" ]]; then
    echo "[skip] pretrained_models/sam-3d-body-dinov3 already present"
  else
    have_hf || { echo "huggingface_hub is required; install the env first or pip install huggingface_hub" >&2; exit 1; }
    echo "[get ] pretrained_models/sam-3d-body-dinov3 (gated: accept terms at"
    echo "       https://huggingface.co/facebook/sam-3d-body-dinov3 )"
    python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/sam-3d-body-dinov3",
    local_dir="pretrained_models/sam-3d-body-dinov3",
)
PY
  fi
fi

# --- DINOv3 config (gated) and pinned source --------------------------------
if [[ "$SKIP_DINOV3" == 0 ]]; then
  have_hf || { echo "Install huggingface_hub in the Python used for downloads first" >&2; exit 1; }
  python scripts/download_runtime_assets.py
fi

# --- ARGAvatar's own checkpoint ---------------------------------------------
# ~260 MB across 2 files (the canonical copy and its expr/ mirror), gitignored
# like everything else here. Published to the HuggingFace model repo below by
# scripts/release/publish_checkpoint.py; ARGAVATAR_WEIGHTS_URL overrides it
# with a plain base URL (e.g. GitHub release assets) if set.
#
# The model lives at TWO paths, because two consumers disagree about layout:
#   checkpoints/<project_name>.pth    <- what a user passes to --checkpoint_path,
#                                        and what model.load_checkpoint names
#   expr/<project_name>/PT_<step>.pth <- manager.py::resolve_jobs, which globs
#                                        expr/<name>/PT_*.pth and takes the
#                                        newest by mtime
# Download fetches only the canonical checkpoints/ copy, then hardlinks the
# expr/ path to it -- so the pair costs one file on disk, as it does upstream.
ARGAVATAR_WEIGHTS_URL="${ARGAVATAR_WEIGHTS_URL:-}"
ARGAVATAR_WEIGHTS_HF_REPO="${ARGAVATAR_WEIGHTS_HF_REPO:-atlantix/ARGAvatar}"
# "<project_name>:<step>"
ARGAVATAR_MODELS=(
  "ARGAvatar-Final:80000"
)

missing=()
for entry in "${ARGAVATAR_MODELS[@]}"; do
  name="${entry%%:*}"
  [[ -f "checkpoints/$name.pth" ]] || missing+=("$name")
done

if [[ ${#missing[@]} -eq 0 ]]; then
  echo "[skip] ARGAvatar checkpoints already present"
elif [[ -n "$ARGAVATAR_WEIGHTS_URL" ]]; then
  for name in "${missing[@]}"; do
    dest="checkpoints/$name.pth"
    echo "[get ] $dest"
    mkdir -p "$(dirname "$dest")"
    curl -fL --progress-bar -o "$dest.part" "${ARGAVATAR_WEIGHTS_URL%/}/$dest"
    mv "$dest.part" "$dest"
  done
elif [[ -n "$ARGAVATAR_WEIGHTS_HF_REPO" ]]; then
  have_hf || { echo "huggingface_hub is required; install the env first or pip install huggingface_hub" >&2; exit 1; }
  for name in "${missing[@]}"; do
    dest="checkpoints/$name.pth"
    echo "[get ] $ARGAVATAR_WEIGHTS_HF_REPO:$name.pth"
    mkdir -p "$(dirname "$dest")"
    python - "$ARGAVATAR_WEIGHTS_HF_REPO" "$name.pth" "$dest" <<'PY'
import shutil
import sys

from huggingface_hub import hf_hub_download

repo_id, filename, dest = sys.argv[1:4]
path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
shutil.copyfile(path, dest)
PY
  done
else
  echo
  echo "ARGAvatar's own checkpoints are not present and no host is configured:" >&2
  printf '  missing: checkpoints/%s.pth\n' "${missing[@]}" >&2
  echo "Set ARGAVATAR_WEIGHTS_HF_REPO (a HuggingFace model repo id) or" >&2
  echo "ARGAVATAR_WEIGHTS_URL (a base URL), or copy them in by hand. Only the" >&2
  echo "third-party weights above were fetched." >&2
  exit 1
fi

# Mirror each checkpoint into expr/<project_name>/PT_<step>.pth for resolve_jobs.
# A hardlink, not a copy: the two paths are one file, as they are upstream.
for entry in "${ARGAVATAR_MODELS[@]}"; do
  name="${entry%%:*}"
  step="${entry##*:}"
  canonical="checkpoints/$name.pth"
  mirror="expr/$name/PT_$step.pth"
  [[ -f "$canonical" ]] || continue
  if [[ -f "$mirror" ]]; then
    echo "[skip] $mirror already present"
  else
    mkdir -p "$(dirname "$mirror")"
    ln "$canonical" "$mirror" 2>/dev/null || cp "$canonical" "$mirror"
    echo "[link] $mirror -> $canonical"
  fi
done

echo
echo "Done. Verify with:"
echo "  python scripts/verify_release.py --imports --configs --checkpoints --models"
