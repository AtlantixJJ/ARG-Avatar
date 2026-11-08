#!/usr/bin/env bash
# Create a clean ARGAvatar CUDA environment, install the release, and verify it.
set -euo pipefail

ENV_NAME="${ARGAVATAR_ENV_NAME:-argavatar}"
PYTHON_VERSION="${ARGAVATAR_PYTHON_VERSION:-3.12}"
VERIFY=1

usage() {
  cat <<'EOF'
Usage: scripts/install_argavatar_env.sh [--env NAME] [--no-verify]

Environment overrides: ARGAVATAR_ENV_NAME, ARGAVATAR_PYTHON_VERSION,
ARGAVATAR_TMPDIR, ARGAVATAR_PIP_CACHE_DIR. Package/CUDA versions live in requirements.txt.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="${2:?--env requires a name}"; shift 2 ;;
    --no-verify) VERIFY=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TMPDIR="${ARGAVATAR_TMPDIR:-$ROOT/.build-tmp}"
export PIP_CACHE_DIR="${ARGAVATAR_PIP_CACHE_DIR:-$ROOT/.pip-cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
# Do not inherit compilers, flags, or search paths from another Conda
# environment.  These can silently link an extension against a different env.
unset CC CXX CUDAHOSTCXX CC_FOR_BUILD CXX_FOR_BUILD CPP
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CPATH LIBRARY_PATH CMAKE_PREFIX_PATH
unset CONDA_BUILD_SYSROOT CONDA_TOOLCHAIN_BUILD CONDA_TOOLCHAIN_HOST
command -v conda >/dev/null || { echo "conda is required" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "CUDA toolkit (nvcc) is required to build extensions" >&2; exit 1; }

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

run() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }
# PyTorch and the build tools must exist before pip prepares the CUDA
# extensions' metadata, so install those pins from requirements.txt first.
mapfile -t BOOTSTRAP < <(grep -E '^(torch|torchvision|setuptools|wheel|ninja)([=<>!~ ]|$)' requirements.txt)
run python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 "${BOOTSTRAP[@]}"
run python -m pip install --no-build-isolation -r requirements.txt

if [[ "$VERIFY" == 1 ]]; then
  run python -m pip check
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 run python scripts/verify_release.py --imports --configs --checkpoints --models --entrypoints
fi
