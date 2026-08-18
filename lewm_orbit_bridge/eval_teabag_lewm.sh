#!/usr/bin/env bash
set -euo pipefail

ORBIT_DIR="${ORBIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${LEWM_PYTHON:-${ORBIT_DIR}/.venv-lewm/bin/python}"
CONFIG="${LEWM_CONFIG:-${ORBIT_DIR}/lewm_orbit_bridge/config/teabag.yaml}"
CHECKPOINT="${LEWM_CHECKPOINT:?Set LEWM_CHECKPOINT to the trained object or weights checkpoint}"
export PYTHONPATH="${ORBIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -ne 0 ]]; then
    echo "This wrapper takes no evaluator-specific arguments; invoke the Python evaluator directly for overrides."
    exit 2
fi
"${PYTHON}" -m lewm_orbit_bridge.evaluate_latent_prediction --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
"${PYTHON}" -m lewm_orbit_bridge.evaluate_goal_structure --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
