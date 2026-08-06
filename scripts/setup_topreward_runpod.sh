#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
ORBIT_DIR="${ORBIT_DIR:-${WORKSPACE_DIR}/orbit}"
VENV_DIR="${TOPREWARD_VENV:-${WORKSPACE_DIR}/venvs/orbit-topreward}"

python3 -m venv --system-site-packages "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
    "accelerate>=1.7,<2" \
    "numpy>=1.26,<3" \
    "opencv-python-headless>=4.9,<5" \
    "pandas>=2.1,<3" \
    "pillow>=10,<13" \
    "pyarrow>=15,<23" \
    "qwen-vl-utils>=0.0.14" \
    "transformers>=4.57,<5"

"${VENV_DIR}/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"
"${VENV_DIR}/bin/python" -c "from transformers import Qwen3VLForConditionalGeneration; print('Qwen3-VL dependencies ready')"

printf 'TOPReward environment ready: %s\n' "${VENV_DIR}"
printf 'Orbit repository: %s\n' "${ORBIT_DIR}"
