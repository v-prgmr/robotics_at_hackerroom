#!/usr/bin/env bash
set -euo pipefail

ORBIT_DIR="${ORBIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LEWM_ROOT="${LEWM_ROOT:-${ORBIT_DIR}/external/le-wm}"
LEWM_SHA="${LEWM_SHA:-8edfeb336732b5f3ce7b8b210d0ba370a09e2cac}"
STABLEWM_ROOT="${STABLEWM_ROOT:-${ORBIT_DIR}/external/stable-worldmodel}"
STABLEWM_SHA="${STABLEWM_SHA:-9a66d7d020043c8efb507f45373e808714f0842d}"
VENV_DIR="${LEWM_VENV:-${ORBIT_DIR}/.venv-lewm}"

mkdir -p "$(dirname "${LEWM_ROOT}")"
if [[ ! -d "${LEWM_ROOT}/.git" ]]; then
    git clone https://github.com/lucas-maes/le-wm.git "${LEWM_ROOT}"
fi
git -C "${LEWM_ROOT}" fetch origin
git -C "${LEWM_ROOT}" checkout --detach "${LEWM_SHA}"

if [[ ! -d "${STABLEWM_ROOT}/.git" ]]; then
    git clone https://github.com/galilai-group/stable-worldmodel.git "${STABLEWM_ROOT}"
fi
git -C "${STABLEWM_ROOT}" fetch origin
git -C "${STABLEWM_ROOT}" checkout --detach "${STABLEWM_SHA}"

uv venv --python 3.10 "${VENV_DIR}"
uv pip install --python "${VENV_DIR}/bin/python" --upgrade pip
uv pip install --python "${VENV_DIR}/bin/python" -e "${STABLEWM_ROOT}[train,env]" -r "${ORBIT_DIR}/lewm_orbit_bridge/requirements-convert.txt"
"${VENV_DIR}/bin/python" -c 'import lancedb, stable_pretraining, stable_worldmodel, torch; print("LeWM environment OK", torch.__version__)'
printf '%s\n' "LeWM commit: $(git -C "${LEWM_ROOT}" rev-parse HEAD)"
printf '%s\n' "Stable World Model commit: $(git -C "${STABLEWM_ROOT}" rev-parse HEAD)"
