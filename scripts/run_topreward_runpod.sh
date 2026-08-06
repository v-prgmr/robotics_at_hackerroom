#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
ORBIT_DIR="${ORBIT_DIR:-${WORKSPACE_DIR}/orbit}"
VENV_DIR="${TOPREWARD_VENV:-${WORKSPACE_DIR}/venvs/orbit-topreward}"
INPUT_ROOT="${TOPREWARD_INPUT_ROOT:-${ORBIT_DIR}/dataset/topReward_smoketest}"
OUTPUT_DIR="${TOPREWARD_OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/topreward_smoketest}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "TOPReward environment not found: ${VENV_DIR}"
    echo "Run: bash ${ORBIT_DIR}/scripts/setup_topreward_runpod.sh"
    exit 1
fi

if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "Orbit dataset root not found: ${INPUT_ROOT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ORBIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"

MODE=(--resume)
for arg in "$@"; do
    if [[ "${arg}" == "--overwrite" ]]; then
        MODE=()
        break
    fi
done

exec "${VENV_DIR}/bin/python" -m topreward_orbit.score_orbit_dataset \
    --input-root "${INPUT_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    "${MODE[@]}" \
    "$@"
