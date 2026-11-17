#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY="${VENV_DIR:-$ROOT/.venv}/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "Training environment missing. Run: bash $ROOT/scripts/setup_linux.sh" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

case "${1:-}" in
    prepare)
        shift
        exec "$PY" "$ROOT/scripts/prepare_data.py" "$@"
        ;;
    smoke|compare|train)
        mode=$1
        shift
        exec "$PY" "$ROOT/scripts/train_lm.py" --mode "$mode" "$@"
        ;;
    *)
        exec "$PY" "$ROOT/scripts/train_lm.py" "$@"
        ;;
esac
