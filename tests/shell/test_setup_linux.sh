#!/usr/bin/env bash
# Exercise environment selection without downloading Python or installing packages.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$ROOT/outputs/logs"
WORK=$(mktemp -d "$ROOT/outputs/logs/setup-test.XXXXXX")
mkdir -p "$WORK/bin"
export MOCK_LOG="$WORK/commands.log"
export MOCK_PY="$WORK/good-python"
cat > "$MOCK_PY" <<'PY'
#!/usr/bin/env bash
echo "python $*" >> "$MOCK_LOG"
if [[ ${1:-} == -m && ${2:-} == venv ]]; then
    mkdir -p "$3/bin"
    cp "$MOCK_PY" "$3/bin/python"
    chmod +x "$3/bin/python"
fi
PY
chmod +x "$MOCK_PY"
for name in python3.11 python3.12 python3 python; do
    printf '#!/usr/bin/env bash\nexit 1\n' > "$WORK/bin/$name"
    chmod +x "$WORK/bin/$name"
done
cat > "$WORK/bin/conda" <<'CONDA'
#!/usr/bin/env bash
echo "conda $*" >> "$MOCK_LOG"
while [[ $# -gt 0 ]]; do
    if [[ $1 == --prefix ]]; then
        mkdir -p "$2/bin"
        cp "$MOCK_PY" "$2/bin/python"
        chmod +x "$2/bin/python"
        exit 0
    fi
    shift
done
exit 1
CONDA
chmod +x "$WORK/bin/conda"
export PATH="$WORK/bin:/usr/bin:/bin"
export CONDA_EXE="$WORK/bin/conda"
export TORCH_BACKEND=cpu
unset PYTHON

# Unsupported base Python falls back to Conda in the requested prefix.
export VENV_DIR="$WORK/conda-env"
bash "$ROOT/scripts/setup_linux.sh" > "$WORK/fallback.log"
grep -q 'conda create.*python=3.11' "$MOCK_LOG"
[[ -f "$VENV_DIR/server-environment.txt" ]]

# Rerunning uses the existing compatible prefix, without creating another environment.
before=$(grep -c '^conda create' "$MOCK_LOG")
bash "$ROOT/scripts/setup_linux.sh" > "$WORK/reuse.log"
[[ $(grep -c '^conda create' "$MOCK_LOG") == "$before" ]]

# Explicit invalid interpreter fails clearly before package installation.
export VENV_DIR="$WORK/invalid-env"
if PYTHON="$WORK/bin/python3" bash "$ROOT/scripts/setup_linux.sh" > "$WORK/invalid.log" 2>&1; then
    echo 'Expected invalid PYTHON to fail' >&2
    exit 1
fi
grep -q 'not Python 3.11/3.12' "$WORK/invalid.log"
[[ ! -e "$VENV_DIR" ]]

# Explicit compatible interpreter creates a venv instead of invoking Conda.
export VENV_DIR="$WORK/venv-env"
PYTHON="$MOCK_PY" bash "$ROOT/scripts/setup_linux.sh" > "$WORK/venv.log"
[[ $(grep -c '^conda create' "$MOCK_LOG") == "$before" ]]
[[ -f "$VENV_DIR/server-environment.txt" ]]
echo "Passed: Conda fallback, prefix reuse, invalid PYTHON, explicit venv. Logs: $WORK"
