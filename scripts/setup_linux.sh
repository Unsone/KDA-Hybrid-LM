#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-"$ROOT/.venv"}
TORCH_VERSION=${TORCH_VERSION:-2.9.1}
TORCH_BACKEND=${TORCH_BACKEND:-cu126}

if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
Usage: bash scripts/setup_linux.sh
Creates a venv and installs training/test dependencies, without sudo.
Environment options:
  PYTHON=python3.11       Python 3.11 or 3.12 (default: python3)
  VENV_DIR=/path/to/venv  Default: <repository>/.venv
  TORCH_BACKEND=cu126     Official wheel channel; cpu, cu128, cu130 also supported
  TORCH_BACKEND=existing Keep PyTorch already installed inside VENV_DIR
  TORCH_VERSION=2.9.1     PyTorch version (must support the chosen channel)
  PIP_INDEX_URL=...       Optional mirror for non-PyTorch dependencies
HELP
    exit 0
fi

"$PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3, 11), (3, 12)), "Use Python 3.11/3.12: PYTHON=python3.11 bash scripts/setup_linux.sh"'
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
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
