#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" "$ROOT_DIR/DPG_mujoco_RL/train_value.py" "$@"
