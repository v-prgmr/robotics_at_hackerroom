#!/usr/bin/env bash
set -euo pipefail

# RunPod setup for Orbit -> ManiFlow multi-camera language-conditioned 2D training.
#
# Recommended base image:
#   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# This script creates a separate Python 3.10 conda env for ManiFlow instead of
# using the image's Python 3.11 environment.
#
# Expected layout:
#   /workspace/orbit/      this repo
#   /workspace/maniflow/   AllenAI ManiFlow checkout, cloned if missing
#
# Usage:
#   cd /workspace/orbit
#   bash maniflow_orbit_bridge/runpod_setup_maniflow_env.sh

MANIFLOW_ENV_MANAGER="${MANIFLOW_ENV_MANAGER:-conda}"
if [[ "${MANIFLOW_ENV_MANAGER}" == "uv" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    exec bash "${SCRIPT_DIR}/setup_maniflow_uv_env.sh"
elif [[ "${MANIFLOW_ENV_MANAGER}" != "conda" ]]; then
    echo "MANIFLOW_ENV_MANAGER must be 'conda' or 'uv', got: ${MANIFLOW_ENV_MANAGER}"
    exit 1
fi

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
ORBIT_DIR="${ORBIT_DIR:-${WORKSPACE_DIR}/orbit}"
MANIFLOW_DIR="${MANIFLOW_DIR:-${WORKSPACE_DIR}/maniflow}"
CONDA_ENV="${CONDA_ENV:-maniflow}"
MANIFLOW_PYTHON_VERSION="${MANIFLOW_PYTHON_VERSION:-3.10}"
MINICONDA_DIR="${MINICONDA_DIR:-${WORKSPACE_DIR}/miniconda3}"
MINICONDA_URL="${MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${WORKSPACE_DIR}/conda_envs/${CONDA_ENV}}"

mkdir -p "${WORKSPACE_DIR}"

if ! command -v conda >/dev/null 2>&1; then
    if [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
        echo "conda not found on PATH; using existing Miniconda at ${MINICONDA_DIR}"
        export PATH="${MINICONDA_DIR}/bin:${PATH}"
    else
        echo "conda not found; installing Miniconda into ${MINICONDA_DIR}"
        MINICONDA_INSTALLER="${WORKSPACE_DIR}/miniconda.sh"
        if command -v curl >/dev/null 2>&1; then
            curl -L "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
        elif command -v wget >/dev/null 2>&1; then
            wget "${MINICONDA_URL}" -O "${MINICONDA_INSTALLER}"
        else
            echo "Neither curl nor wget is available to download Miniconda."
            exit 1
        fi
        bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_DIR}"
        rm -f "${MINICONDA_INSTALLER}"
        export PATH="${MINICONDA_DIR}/bin:${PATH}"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda install failed or conda is still not on PATH."
    exit 1
fi

# Some Conda package hooks read optional variables while switching environments.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda tos --help >/dev/null 2>&1; then
    echo "Accepting Anaconda Terms of Service for default channels"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
fi

if [[ ! -d "${ORBIT_DIR}" ]]; then
    echo "Orbit repo not found at ${ORBIT_DIR}"
    echo "Clone it first, for example:"
    echo "  git clone git@github.com:v-prgmr/robotics_at_hackerroom.git ${ORBIT_DIR}"
    exit 1
fi

if [[ ! -d "${MANIFLOW_DIR}" ]]; then
    echo "Cloning ManiFlow into ${MANIFLOW_DIR}"
    git clone https://github.com/allenai/maniflow.git "${MANIFLOW_DIR}"
fi

if [[ -d "${CONDA_ENV_DIR}" ]]; then
    if [[ ! -x "${CONDA_ENV_DIR}/bin/python" ]]; then
        echo "Existing Conda env is invalid because ${CONDA_ENV_DIR}/bin/python is missing."
        exit 1
    fi
    EXISTING_PYTHON_VERSION="$("${CONDA_ENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${EXISTING_PYTHON_VERSION}" != "${MANIFLOW_PYTHON_VERSION}" ]]; then
        echo "Existing Conda env uses Python ${EXISTING_PYTHON_VERSION}; ManiFlow requires ${MANIFLOW_PYTHON_VERSION}."
        echo "Remove ${CONDA_ENV_DIR} and rerun setup."
        exit 1
    fi
    echo "Conda env already exists at ${CONDA_ENV_DIR}; reusing it."
else
    echo "Creating persistent conda env at ${CONDA_ENV_DIR} with Python ${MANIFLOW_PYTHON_VERSION}"
    mkdir -p "$(dirname "${CONDA_ENV_DIR}")"
    conda create -y -p "${CONDA_ENV_DIR}" "python=${MANIFLOW_PYTHON_VERSION}" pip
fi

ENV_PYTHON="${CONDA_ENV_DIR}/bin/python"
ACTIVE_PYTHON_VERSION="$("${ENV_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${ACTIVE_PYTHON_VERSION}" != "${MANIFLOW_PYTHON_VERSION}" ]]; then
    echo "Expected Python ${MANIFLOW_PYTHON_VERSION} at ${ENV_PYTHON}, got ${ACTIVE_PYTHON_VERSION}."
    exit 1
fi

"${ENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

# Match the RunPod CUDA 12.4 image while using Python 3.10 in this env.
# The MKL/OpenMP pins avoid PyTorch import failures like:
#   libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
conda install -y -p "${CONDA_ENV_DIR}" -c pytorch -c nvidia -c defaults \
    "python=${MANIFLOW_PYTHON_VERSION}" \
    pytorch==2.4.1 \
    torchvision \
    torchaudio \
    pytorch-cuda=12.4 \
    "mkl<2024.1" \
    "intel-openmp<2024.1"
set -u

# Minimal dependency set for Orbit's 2D image ManiFlow path. This intentionally
# skips PyTorch3D, flash-attn, MuJoCo, RoboTwin, DexArt, and pointcloud deps.
"${ENV_PYTHON}" -m pip install \
    "numpy==1.24.4" \
    "scipy==1.10.1" \
    "scikit-learn==1.3.2" \
    "pandas" \
    "pyarrow==15.0.2" \
    "h5py==3.13.0" \
    "opencv-python==4.5.5.64" \
    "zarr==2.12.0" \
    "numcodecs<0.16" \
    "numba==0.61.2" \
    "hydra-core==1.2.0" \
    "hydra-colorlog" \
    "omegaconf" \
    "dill==0.3.5.1" \
    "wandb" \
    "tqdm==4.66.5" \
    "termcolor" \
    "einops==0.8.1" \
    "timm" \
    "diffusers==0.27.2" \
    "accelerate==0.34.2" \
    "peft>=0.15,<0.19" \
    "transformers==4.46.1" \
    "huggingface_hub==0.25.0" \
    "safetensors==0.4.5" \
    "regex==2024.9.11" \
    "sentencepiece==0.2.0" \
    "ftfy"

# Upstream ManiFlow may not include this package marker, which makes
# `pip install -e` succeed while `import maniflow` still fails.
touch "${MANIFLOW_DIR}/maniflow/__init__.py"
"${ENV_PYTHON}" -m pip install -e "${MANIFLOW_DIR}"

"${ENV_PYTHON}" "${ORBIT_DIR}/maniflow_orbit_bridge/install_into_maniflow.py" \
    --maniflow-dir "${MANIFLOW_DIR}" \
    --overwrite

"${ENV_PYTHON}" - <<'PY'
import cv2
import h5py
import hydra
import numpy
import pandas
import pyarrow
import torch
import torchvision
import zarr
import maniflow
import timm
import transformers

print("ManiFlow Orbit env OK")
print("python ok")
print("torch", torch.__version__)
print("torch cuda available", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
print("h5py", h5py.__version__)
print("pyarrow", pyarrow.__version__)
print("zarr", zarr.__version__)
print("transformers", transformers.__version__)
PY

echo "Setup complete. Activate with: conda activate ${CONDA_ENV_DIR}"
echo "Place your converted dataset under: ${WORKSPACE_DIR}/dataset"
echo "Then run: bash ${ORBIT_DIR}/maniflow_orbit_bridge/runpod_train_maniflow_orbit.sh"
