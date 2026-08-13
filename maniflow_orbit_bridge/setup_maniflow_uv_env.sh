#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
ORBIT_DIR="${ORBIT_DIR:-${WORKSPACE_DIR}/orbit}"
MANIFLOW_DIR="${MANIFLOW_DIR:-${WORKSPACE_DIR}/maniflow}"
UV_PROJECT_DIR="${UV_PROJECT_DIR:-${ORBIT_DIR}/maniflow_orbit_bridge/uv_training}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${WORKSPACE_DIR}/venvs/maniflow-training}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_DIR}/uv-cache}"
MANIFLOW_REVISION="${MANIFLOW_REVISION:-e4dc24d62a6c91825813308b9926e921b4bb9ef6}"
UV_INSTALL_DIR="${UV_INSTALL_DIR:-${WORKSPACE_DIR}/uv}"

export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR
mkdir -p "${WORKSPACE_DIR}" "$(dirname "${UV_PROJECT_ENVIRONMENT}")" "${UV_CACHE_DIR}"

if [[ ! -d "${ORBIT_DIR}" ]]; then
    echo "Orbit repo not found at ${ORBIT_DIR}"
    exit 1
fi
if [[ ! -f "${UV_PROJECT_DIR}/uv.lock" ]]; then
    echo "Locked ManiFlow uv project not found at ${UV_PROJECT_DIR}"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "${UV_INSTALL_DIR}/bin/uv" ]]; then
        export PATH="${UV_INSTALL_DIR}/bin:${PATH}"
    else
        echo "uv not found; installing it into ${UV_INSTALL_DIR}"
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh
        else
            echo "Neither curl nor wget is available to install uv."
            exit 1
        fi
        export PATH="${UV_INSTALL_DIR}/bin:${PATH}"
    fi
fi

if [[ ! -e "${MANIFLOW_DIR}" ]]; then
    echo "Cloning ManiFlow into ${MANIFLOW_DIR}"
    git clone https://github.com/allenai/maniflow.git "${MANIFLOW_DIR}"
elif [[ ! -d "${MANIFLOW_DIR}/.git" ]]; then
    echo "Existing ManiFlow path is not a git checkout: ${MANIFLOW_DIR}"
    exit 1
fi

echo "Checking out pinned ManiFlow revision ${MANIFLOW_REVISION}"
git -C "${MANIFLOW_DIR}" fetch origin "${MANIFLOW_REVISION}"
git -C "${MANIFLOW_DIR}" checkout --detach "${MANIFLOW_REVISION}"

echo "Syncing locked Python 3.10 CUDA environment at ${UV_PROJECT_ENVIRONMENT}"
uv python install 3.10
uv sync --project "${UV_PROJECT_DIR}" --frozen --python 3.10

ENV_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
touch "${MANIFLOW_DIR}/maniflow/__init__.py"
uv pip install --python "${ENV_PYTHON}" --no-deps -e "${MANIFLOW_DIR}"
"${ENV_PYTHON}" "${ORBIT_DIR}/maniflow_orbit_bridge/install_into_maniflow.py" \
    --maniflow-dir "${MANIFLOW_DIR}" \
    --overwrite

"${ENV_PYTHON}" - <<'PY'
import torch
import torchvision
import timm
import transformers
import zarr
import maniflow

assert torch.__version__.startswith("2.4.1")
assert torchvision.__version__.startswith("0.19.1")
print("ManiFlow uv environment OK")
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("timm", timm.__version__)
print("transformers", transformers.__version__)
print("zarr", zarr.__version__)
PY

echo "Setup complete. Activate with: source ${UV_PROJECT_ENVIRONMENT}/bin/activate"
