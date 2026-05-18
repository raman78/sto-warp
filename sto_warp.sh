#!/bin/sh
# sto_warp.sh — Linux/macOS entry point for sto-warp.
#
# First run: creates a local .venv next to the script and installs the
# project in editable mode. Subsequent runs reuse the venv and just
# launch the `sto-warp` console script.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || { echo "[sto-warp] Cannot enter $SCRIPT_DIR"; exit 1; }

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_BIN="$VENV_DIR/bin"
STO_WARP="$VENV_BIN/sto-warp"

# Reuse existing venv if the console script is present.
if [ -x "$STO_WARP" ]; then
    exec "$STO_WARP" "$@"
fi

# ── First run — locate a system Python ≥3.14 ──────────────────────────────
PYTHON=""
for candidate in python3.15 python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ok=$("$candidate" -c "import sys; print(1 if sys.version_info >= (3,14) else 0)" 2>/dev/null)
        if [ "$ok" = "1" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[sto-warp] No Python 3.14+ found on PATH."
    echo "           Install it from https://www.python.org/downloads/ and re-run."
    exit 1
fi

echo "[sto-warp] First run — creating venv in $VENV_DIR (one-time, takes a few minutes)…"
"$PYTHON" -m venv "$VENV_DIR" || { echo "[sto-warp] venv creation failed"; exit 1; }

"$VENV_BIN/python" -m pip install --upgrade pip wheel >/dev/null || true
"$VENV_BIN/python" -m pip install -e "$SCRIPT_DIR" || {
    echo "[sto-warp] pip install failed — see output above"
    exit 1
}

if [ ! -x "$STO_WARP" ]; then
    echo "[sto-warp] install completed but $STO_WARP is missing — aborting."
    exit 1
fi

echo "[sto-warp] Setup done. Launching…"
exec "$STO_WARP" "$@"
