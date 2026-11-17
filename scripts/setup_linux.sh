#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${VENV_DIR:-"$ROOT/.venv"}
TORCH_VERSION=${TORCH_VERSION:-2.9.1}
TORCH_BACKEND=${TORCH_BACKEND:-cu126}

if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
Usage: bash scripts/setup_linux.sh
Finds Python 3.11/3.12 or creates it with Conda, then installs training/test dependencies.
Environment options:
  PYTHON=python3.11       Explicit interpreter; otherwise auto-detect Python or Conda
  VENV_DIR=/path/to/venv  Default: <repository>/.venv
  TORCH_BACKEND=cu126     Official wheel channel; cpu, cu128, cu130 also supported
  TORCH_BACKEND=existing Keep PyTorch already installed inside VENV_DIR
  TORCH_VERSION=2.9.1     PyTorch version (must support the chosen channel)
  PIP_INDEX_URL=...       Optional mirror for non-PyTorch dependencies
HELP
    exit 0
fi

supported_python() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)' 2>/dev/null
}

if [[ -x "$VENV_DIR/bin/python" ]]; then
    if ! supported_python "$VENV_DIR/bin/python"; then
        echo "Existing $VENV_DIR needs Python 3.11/3.12. Select a new VENV_DIR; nothing was removed." >&2
        exit 1
    fi
elif [[ -n ${PYTHON:-} ]]; then
    if ! supported_python "$PYTHON"; then
        echo "PYTHON=$PYTHON is not Python 3.11/3.12. Unset PYTHON to auto-detect or use Conda." >&2
        exit 1
    fi
    "$PYTHON" -m venv "$VENV_DIR"
else
    for candidate in python3.11 python3.12 python3 python; do
        if supported_python "$candidate"; then
            PYTHON=$candidate
            break
        fi
    done
    if [[ -n ${PYTHON:-} ]]; then
        echo "Creating virtual environment with $PYTHON"
        "$PYTHON" -m venv "$VENV_DIR"
    elif command -v "${CONDA_EXE:-conda}" >/dev/null 2>&1; then
        echo "No Python 3.11/3.12 found; creating Python 3.11 with Conda at $VENV_DIR"
        "${CONDA_EXE:-conda}" create --yes --prefix "$VENV_DIR" \
            --override-channels --channel conda-forge python=3.11 pip
    else
        echo "Python 3.11/3.12 or Conda is required. Install one, then rerun this script." >&2
        exit 1
    fi
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip
if [[ "$TORCH_BACKEND" == existing ]]; then
    "$PY" -c 'import torch; print("Keeping PyTorch", torch.__version__)'
elif [[ "$TORCH_BACKEND" == cpu || "$TORCH_BACKEND" =~ ^cu[0-9]+$ ]]; then
    "$PY" -m pip install "torch==$TORCH_VERSION" \
        --index-url "https://download.pytorch.org/whl/$TORCH_BACKEND"
else
    echo "Invalid TORCH_BACKEND: $TORCH_BACKEND; use cpu, cu126, cu128, cu130 or existing." >&2
    exit 1
fi
"$PY" -m pip install -r "$ROOT/requirements-linux.txt"
# Runtime metadata permits the separately selected CUDA wheel; never pins local CPU torch.
"$PY" -m pip install --no-deps -e "$ROOT"
"$PY" -m pip check
"$PY" - "$TORCH_BACKEND" <<'PY'
import sys
import torch
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    value = torch.ones((16, 16), device="cuda")
    print("CUDA matmul:", (value @ value).mean().item())
elif sys.argv[1] not in ("cpu", "existing"):
    raise SystemExit("CUDA wheel installed but GPU unavailable. Check nvidia-smi/driver/container GPU access.")
PY
"$PY" -m pip freeze > "$VENV_DIR/server-environment.txt"
echo "Ready. Run: bash $ROOT/start_training.sh smoke"
