#!/usr/bin/env bash
set -euo pipefail

# RunPod launcher for Orbit -> ManiFlow multi-camera language-conditioned 2D training.
#
# Expected layout on the pod:
#   /workspace/orbit/                      this repo
#   /workspace/maniflow/                   AllenAI ManiFlow checkout
#   /workspace/dataset/<dataset>.zarr      converted Orbit ManiFlow zarr dataset
#   /workspace/outputs/train/<run_name>/   training outputs/checkpoints
#
# Override defaults with environment variables, for example:
#   DATASET_NAME=teabags_kitting_50_v2_maniflow.zarr \
#   HF_DATASET_REPO_ID=v-prgmr/teabags-kitting-50-v2-maniflow \
#   RUN_NAME=maniflow_teabags_v2 \
#   BATCH_SIZE=8 \
#   LOGGING_MODE=offline \
#   bash maniflow_orbit_bridge/runpod_train_maniflow_orbit.sh

is_true() {
    case "${1:-}" in
        true|True|TRUE|1|yes|Yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
ORBIT_DIR="${ORBIT_DIR:-${WORKSPACE_DIR}/orbit}"
MANIFLOW_DIR="${MANIFLOW_DIR:-${WORKSPACE_DIR}/maniflow}"

DATASET_NAME="${DATASET_NAME:-teabags_kitting_50_v2_maniflow.zarr}"
DATASET_ZARR="${DATASET_ZARR:-${WORKSPACE_DIR}/dataset/${DATASET_NAME}}"

HF_DATASET_REPO_ID="${HF_DATASET_REPO_ID:-}"
HF_DATASET_REPO_TYPE="${HF_DATASET_REPO_TYPE:-dataset}"
HF_DATASET_REVISION="${HF_DATASET_REVISION:-main}"
HF_DATASET_PATH_IN_REPO="${HF_DATASET_PATH_IN_REPO:-${DATASET_NAME}}"
HF_DATASET_TOKEN="${HF_DATASET_TOKEN:-${HF_TOKEN:-}}"
HF_DATASET_FORCE_DOWNLOAD="${HF_DATASET_FORCE_DOWNLOAD:-false}"

RUN_NAME="${RUN_NAME:-maniflow_teabags_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/train/${RUN_NAME}}"

CONDA_ENV="${CONDA_ENV:-maniflow}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${WORKSPACE_DIR}/conda_envs/${CONDA_ENV}}"
MINICONDA_DIR="${MINICONDA_DIR:-${WORKSPACE_DIR}/miniconda3}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-501}"
DEBUG="${DEBUG:-False}"
LOGGING_MODE="${LOGGING_MODE:-offline}"

PUSH_TO_HF="${PUSH_TO_HF:-false}"
PUSH_TO_HF_ON_SUCCESS_ONLY="${PUSH_TO_HF_ON_SUCCESS_ONLY:-true}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_TOKEN="${HF_TOKEN:-}"
HF_PRIVATE="${HF_PRIVATE:-false}"
HF_REMOTE_PREFIX="${HF_REMOTE_PREFIX:-runs/${RUN_NAME}}"
HF_UPLOAD_ALL_CHECKPOINTS="${HF_UPLOAD_ALL_CHECKPOINTS:-false}"
HF_UPLOAD_WANDB="${HF_UPLOAD_WANDB:-false}"

STOP_POD_ON_EXIT="${STOP_POD_ON_EXIT:-false}"
STOP_POD_ON_SUCCESS_ONLY="${STOP_POD_ON_SUCCESS_ONLY:-true}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-${POD_ID:-}}"

mkdir -p "${WORKSPACE_DIR}/dataset" "${WORKSPACE_DIR}/outputs/train"

if [[ ! -d "${ORBIT_DIR}" ]]; then
    echo "Orbit repo not found at ${ORBIT_DIR}"
    echo "Clone it first, for example: git clone git@github.com:v-prgmr/robotics_at_hackerroom.git ${ORBIT_DIR}"
    exit 1
fi

if [[ ! -d "${MANIFLOW_DIR}" ]]; then
    echo "ManiFlow checkout not found at ${MANIFLOW_DIR}; cloning it."
    git clone https://github.com/allenai/maniflow.git "${MANIFLOW_DIR}"
fi

if ! command -v conda >/dev/null 2>&1 && [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
    export PATH="${MINICONDA_DIR}/bin:${PATH}"
fi

if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if [[ -d "${CONDA_ENV_DIR}" ]]; then
        conda activate "${CONDA_ENV_DIR}"
    elif conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
        conda activate "${CONDA_ENV}"
    else
        echo "Conda env not found at ${CONDA_ENV_DIR}. Create/install the ManiFlow env before training."
        exit 1
    fi
else
    echo "conda not found; continuing with current Python environment."
fi

if [[ -n "${HF_DATASET_REPO_ID}" ]]; then
    if [[ ! -d "${DATASET_ZARR}" ]] || is_true "${HF_DATASET_FORCE_DOWNLOAD}"; then
        echo "Downloading ManiFlow dataset from Hugging Face Hub"
        echo "  repo:          ${HF_DATASET_REPO_ID}"
        echo "  path in repo:  ${HF_DATASET_PATH_IN_REPO}"
        echo "  destination:   ${DATASET_ZARR}"
        HF_DATASET_REPO_ID="${HF_DATASET_REPO_ID}" \
        HF_DATASET_REPO_TYPE="${HF_DATASET_REPO_TYPE}" \
        HF_DATASET_REVISION="${HF_DATASET_REVISION}" \
        HF_DATASET_PATH_IN_REPO="${HF_DATASET_PATH_IN_REPO}" \
        HF_DATASET_TOKEN="${HF_DATASET_TOKEN}" \
        DATASET_NAME="${DATASET_NAME}" \
        DATASET_ZARR="${DATASET_ZARR}" \
        python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


repo_id = os.environ["HF_DATASET_REPO_ID"]
repo_type = os.environ.get("HF_DATASET_REPO_TYPE", "dataset")
revision = os.environ.get("HF_DATASET_REVISION") or None
path_in_repo = os.environ.get("HF_DATASET_PATH_IN_REPO", "").strip("/")
token = os.environ.get("HF_DATASET_TOKEN") or None
dataset_zarr = Path(os.environ["DATASET_ZARR"])

dataset_zarr.parent.mkdir(parents=True, exist_ok=True)

if path_in_repo in {"", "."}:
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
        local_dir=str(dataset_zarr),
    )
else:
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
        local_dir=str(dataset_zarr.parent),
        allow_patterns=[f"{path_in_repo}/**"],
    )
    downloaded_path = dataset_zarr.parent / path_in_repo
    if downloaded_path.resolve() != dataset_zarr.resolve():
        if dataset_zarr.exists():
            shutil.rmtree(dataset_zarr)
        dataset_zarr.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded_path), str(dataset_zarr))

if not dataset_zarr.is_dir():
    raise FileNotFoundError(f"Downloaded dataset zarr not found: {dataset_zarr}")

print(f"Dataset ready: {dataset_zarr}")
PY
    else
        echo "Dataset already exists at ${DATASET_ZARR}; skipping HF dataset download."
    fi
fi

if [[ ! -d "${DATASET_ZARR}" ]]; then
    echo "Dataset zarr not found: ${DATASET_ZARR}"
    echo "Upload/copy the converted .zarr into ${WORKSPACE_DIR}/dataset, or set HF_DATASET_REPO_ID to download it from Hugging Face Hub."
    exit 1
fi

python "${ORBIT_DIR}/maniflow_orbit_bridge/install_into_maniflow.py" \
    --maniflow-dir "${MANIFLOW_DIR}" \
    --overwrite

cd "${MANIFLOW_DIR}/maniflow/workspace"

echo "Starting Orbit ManiFlow training"
echo "  dataset: ${DATASET_ZARR}"
echo "  output:  ${OUTPUT_DIR}"
echo "  device:  ${GPU_DEVICE}"
echo "  debug:   ${DEBUG}"

set +e
python train_maniflow_orbit_workspace.py \
    --config-name=maniflow_image_orbit.yaml \
    robotwin_task=orbit_so100_image \
    robotwin_task.dataset.zarr_path="${DATASET_ZARR}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    exp_name="${RUN_NAME}" \
    training.debug="${DEBUG}" \
    training.device="${GPU_DEVICE}" \
    training.num_epochs="${NUM_EPOCHS}" \
    dataloader.batch_size="${BATCH_SIZE}" \
    val_dataloader.batch_size="${BATCH_SIZE}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    val_dataloader.num_workers="${NUM_WORKERS}" \
    logging.mode="${LOGGING_MODE}" \
    "$@"
TRAIN_EXIT_CODE=$?
set -e

echo "Training exited with code ${TRAIN_EXIT_CODE}"

FINAL_EXIT_CODE="${TRAIN_EXIT_CODE}"

if is_true "${PUSH_TO_HF}"; then
    if [[ -z "${HF_REPO_ID}" ]]; then
        echo "PUSH_TO_HF=true but HF_REPO_ID is empty"
        FINAL_EXIT_CODE=1
    elif is_true "${PUSH_TO_HF_ON_SUCCESS_ONLY}" && [[ "${TRAIN_EXIT_CODE}" -ne 0 ]]; then
        echo "Skipping Hugging Face upload because PUSH_TO_HF_ON_SUCCESS_ONLY=true and training exit code is ${TRAIN_EXIT_CODE}"
    else
        echo "Uploading ManiFlow artifacts to Hugging Face Hub: ${HF_REPO_ID}"
        set +e
        OUTPUT_DIR="${OUTPUT_DIR}" \
        RUN_NAME="${RUN_NAME}" \
        HF_REPO_ID="${HF_REPO_ID}" \
        HF_REPO_TYPE="${HF_REPO_TYPE}" \
        HF_TOKEN="${HF_TOKEN}" \
        HF_PRIVATE="${HF_PRIVATE}" \
        HF_REMOTE_PREFIX="${HF_REMOTE_PREFIX}" \
        HF_UPLOAD_ALL_CHECKPOINTS="${HF_UPLOAD_ALL_CHECKPOINTS}" \
        HF_UPLOAD_WANDB="${HF_UPLOAD_WANDB}" \
        python - <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def is_true(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "y"}


output_dir = Path(os.environ["OUTPUT_DIR"])
repo_id = os.environ["HF_REPO_ID"]
repo_type = os.environ.get("HF_REPO_TYPE", "model")
token = os.environ.get("HF_TOKEN") or None
private = is_true(os.environ.get("HF_PRIVATE"))
remote_prefix = os.environ.get("HF_REMOTE_PREFIX") or f"runs/{os.environ['RUN_NAME']}"
upload_all_checkpoints = is_true(os.environ.get("HF_UPLOAD_ALL_CHECKPOINTS"))
upload_wandb = is_true(os.environ.get("HF_UPLOAD_WANDB"))

if not output_dir.exists():
    raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

api = HfApi(token=token)
create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True, token=token)


def upload_file(path: Path, path_in_repo: str) -> None:
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return
    print(f"Uploading file {path} -> {path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )


def upload_folder(path: Path, path_in_repo: str) -> None:
    if not path.exists():
        print(f"Skipping missing folder: {path}")
        return
    print(f"Uploading folder {path} -> {path_in_repo}")
    api.upload_folder(
        folder_path=str(path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )


upload_folder(output_dir / ".hydra", f"{remote_prefix}/.hydra")

for log_path in output_dir.glob("*.log"):
    upload_file(log_path, f"{remote_prefix}/{log_path.name}")

if upload_all_checkpoints:
    upload_folder(output_dir / "checkpoints", f"{remote_prefix}/checkpoints")
else:
    upload_file(output_dir / "checkpoints" / "latest.ckpt", f"{remote_prefix}/checkpoints/latest.ckpt")

if upload_wandb:
    upload_folder(output_dir / "wandb", f"{remote_prefix}/wandb")

print("Hugging Face upload complete")
PY
        HF_EXIT_CODE=$?
        set -e
        if [[ "${HF_EXIT_CODE}" -ne 0 ]]; then
            echo "Hugging Face upload failed with code ${HF_EXIT_CODE}"
            FINAL_EXIT_CODE=1
        fi
    fi
fi

if is_true "${STOP_POD_ON_EXIT}"; then
    if is_true "${STOP_POD_ON_SUCCESS_ONLY}" && [[ "${FINAL_EXIT_CODE}" -ne 0 ]]; then
        echo "Not stopping pod because STOP_POD_ON_SUCCESS_ONLY=true and final exit code is ${FINAL_EXIT_CODE}"
    elif [[ -z "${RUNPOD_API_KEY}" || -z "${RUNPOD_POD_ID}" ]]; then
        echo "STOP_POD_ON_EXIT=true but RUNPOD_API_KEY or RUNPOD_POD_ID is empty"
        FINAL_EXIT_CODE=1
    else
        echo "Stopping RunPod pod ${RUNPOD_POD_ID}"
        set +e
        RUNPOD_API_KEY="${RUNPOD_API_KEY}" RUNPOD_POD_ID="${RUNPOD_POD_ID}" python - <<'PY'
import json
import os
import urllib.error
import urllib.request

api_key = os.environ["RUNPOD_API_KEY"]
pod_id = os.environ["RUNPOD_POD_ID"]
payload = {
    "query": "mutation StopPod($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }",
    "variables": {"input": {"podId": pod_id}},
}
request = urllib.request.Request(
    f"https://api.runpod.io/graphql?api_key={api_key}",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"RunPod stop failed: HTTP {exc.code}: {body}") from exc

print(body)
data = json.loads(body)
if data.get("errors"):
    raise RuntimeError(f"RunPod stop returned errors: {data['errors']}")
PY
        STOP_EXIT_CODE=$?
        set -e
        if [[ "${STOP_EXIT_CODE}" -ne 0 ]]; then
            echo "RunPod stop failed with code ${STOP_EXIT_CODE}"
            FINAL_EXIT_CODE=1
        fi
    fi
fi

exit "${FINAL_EXIT_CODE}"
