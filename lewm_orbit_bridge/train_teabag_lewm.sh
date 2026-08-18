#!/usr/bin/env bash
set -euo pipefail

ORBIT_DIR="${ORBIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${LEWM_PYTHON:-${ORBIT_DIR}/.venv-lewm/bin/python}"
CONFIG="${LEWM_CONFIG:-${ORBIT_DIR}/lewm_orbit_bridge/config/teabag.yaml}"
export PYTHONPATH="${ORBIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" -m lewm_orbit_bridge.train_lewm --config "${CONFIG}" "$@"
