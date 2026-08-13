#!/usr/bin/env bash
set -euo pipefail

# Vast.ai RTX 4090 setup wrapper. The attached network volume is mounted at /workspace.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export ORBIT_DIR="${ORBIT_DIR:-${ORBIT_REPO_DIR}}"
export MANIFLOW_DIR="${MANIFLOW_DIR:-${WORKSPACE_DIR}/maniflow}"
export MANIFLOW_ENV_MANAGER="${MANIFLOW_ENV_MANAGER:-uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${WORKSPACE_DIR}/venvs/maniflow-training}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_DIR}/uv-cache}"
export MINICONDA_DIR="${MINICONDA_DIR:-${WORKSPACE_DIR}/miniconda3}"
export CONDA_ENV_DIR="${CONDA_ENV_DIR:-${WORKSPACE_DIR}/conda_envs/maniflow}"
export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/huggingface}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
SETUP_LAUNCHER="${SETUP_LAUNCHER:-${SCRIPT_DIR}/runpod_setup_maniflow_env.sh}"

mkdir -p "${WORKSPACE_DIR}" "${HF_HOME}"

echo "Setting up ManiFlow for Vast.ai RTX 4090"
echo "  persistent root: ${WORKSPACE_DIR}"
echo "  Orbit repo:      ${ORBIT_DIR}"
echo "  ManiFlow repo:   ${MANIFLOW_DIR}"
echo "  env manager:     ${MANIFLOW_ENV_MANAGER}"
echo "  uv env:          ${UV_PROJECT_ENVIRONMENT}"

bash "${SETUP_LAUNCHER}"

echo "Vast.ai setup complete. Start training with:"
echo "  bash ${SCRIPT_DIR}/vast_train_maniflow_orbit.sh"
