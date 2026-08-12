#!/usr/bin/env bash
set -euo pipefail

# Train the Phase 1A frozen ManiFlow progress probe.
#
# Required:
#   SOURCE_CHECKPOINT=/path/to/trained_maniflow.ckpt
#   DATASET_ZARR=/path/to/progress-enabled-maniflow.zarr
#
# Additional command-line arguments are passed to Hydra unchanged.

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
ORBIT_DIR="${ORBIT_DIR:-${WORKSPACE_DIR}/orbit}"
MANIFLOW_DIR="${MANIFLOW_DIR:-${WORKSPACE_DIR}/maniflow}"
CONDA_ENV="${CONDA_ENV:-maniflow}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${WORKSPACE_DIR}/conda_envs/${CONDA_ENV}}"
MINICONDA_DIR="${MINICONDA_DIR:-${WORKSPACE_DIR}/miniconda3}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-}"
SOURCE_STATE_KEY="${SOURCE_STATE_KEY:-ema_model}"
DATASET_ZARR="${DATASET_ZARR:-}"

RUN_NAME="${RUN_NAME:-maniflow_progress_phase1a}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/train/${RUN_NAME}}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
NUM_GRAD_STEPS="${NUM_GRAD_STEPS:-}"
GRADIENT_ACCUMULATE_EVERY="${GRADIENT_ACCUMULATE_EVERY:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-3}"
PROGRESS_HIDDEN_DIM="${PROGRESS_HIDDEN_DIM:-512}"
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_EVERY="${VAL_EVERY:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-}"
LOGGING_MODE="${LOGGING_MODE:-online}"

if [[ -z "${SOURCE_CHECKPOINT}" ]]; then
    echo "SOURCE_CHECKPOINT is required and must point to a trained ManiFlow checkpoint."
    exit 1
fi
if [[ -z "${DATASET_ZARR}" ]]; then
    echo "DATASET_ZARR is required and must point to a progress-enabled ManiFlow zarr."
    exit 1
fi
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
    echo "Source checkpoint not found: ${SOURCE_CHECKPOINT}"
    exit 1
fi
if [[ ! -d "${DATASET_ZARR}" ]]; then
    echo "Dataset zarr not found: ${DATASET_ZARR}"
    exit 1
fi
if [[ ! -d "${ORBIT_DIR}" ]]; then
    echo "Orbit repository not found: ${ORBIT_DIR}"
    exit 1
fi
if [[ ! -d "${MANIFLOW_DIR}/maniflow" ]]; then
    echo "ManiFlow checkout not found: ${MANIFLOW_DIR}"
    exit 1
fi
for integer_setting in \
    "BATCH_SIZE=${BATCH_SIZE}" \
    "VAL_BATCH_SIZE=${VAL_BATCH_SIZE}" \
    "GRADIENT_ACCUMULATE_EVERY=${GRADIENT_ACCUMULATE_EVERY}"; do
    setting_name="${integer_setting%%=*}"
    setting_value="${integer_setting#*=}"
    if [[ ! "${setting_value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${setting_name} must be a positive integer, got: ${setting_value}"
        exit 1
    fi
done

SOURCE_CHECKPOINT="$(readlink -f "${SOURCE_CHECKPOINT}")"
DATASET_ZARR="$(readlink -f "${DATASET_ZARR}")"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && readlink -f "${OUTPUT_DIR}")"

if ! command -v conda >/dev/null 2>&1 && [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
    export PATH="${MINICONDA_DIR}/bin:${PATH}"
fi

if command -v conda >/dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if [[ -d "${CONDA_ENV_DIR}" ]]; then
        conda activate "${CONDA_ENV_DIR}"
    elif conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
        conda activate "${CONDA_ENV}"
    else
        echo "Conda environment not found at ${CONDA_ENV_DIR} and no named '${CONDA_ENV}' environment exists."
        exit 1
    fi
    set -u
else
    echo "conda not found; using current Python environment."
fi

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
SOURCE_STATE_KEY="${SOURCE_STATE_KEY}" \
DATASET_ZARR="${DATASET_ZARR}" \
VAL_RATIO="${VAL_RATIO}" \
python - <<'PY'
import os
from pathlib import Path

try:
    import dill
    import hydra
    import torch
    import wandb
    import zarr
    from omegaconf import OmegaConf
except ImportError as exc:
    raise RuntimeError(f"The active Python environment is missing a progress-training dependency: {exc}") from exc

checkpoint = Path(os.environ["SOURCE_CHECKPOINT"])
payload = torch.load(checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
state_key = os.environ["SOURCE_STATE_KEY"]
state_dicts = payload.get("state_dicts", {})
if state_key not in state_dicts:
    raise KeyError(
        f"Checkpoint {checkpoint} has no state_dicts.{state_key}; available keys: {sorted(state_dicts)}"
    )

root = zarr.open_group(os.environ["DATASET_ZARR"], mode="r")
required = (
    "data/episode_progress",
    "data/progress_valid",
    "data/source_episode_index",
    "data/source_frame_index",
    "meta/episode_ends",
)
missing = [key for key in required if key not in root]
if missing:
    raise KeyError(f"Progress dataset is missing required zarr arrays: {missing}")

episode_ends = root["meta/episode_ends"][:]
progress_valid = root["data/progress_valid"][:]
starts = [0, *episode_ends[:-1]]
supervised = sum(bool(progress_valid[start:end].any()) for start, end in zip(starts, episode_ends, strict=True))
if float(os.environ["VAL_RATIO"]) > 0 and supervised < 2:
    raise ValueError(
        f"Episode-held-out validation requires at least two supervised episodes; found {supervised}"
    )
print(f"Validated checkpoint state '{state_key}' and {supervised} supervised dataset episodes.")
PY

python "${ORBIT_DIR}/maniflow_orbit_bridge/install_into_maniflow.py" \
    --maniflow-dir "${MANIFLOW_DIR}" \
    --overwrite

HYDRA_OVERRIDES=(
    "source_checkpoint=${SOURCE_CHECKPOINT}"
    "source_state_key=${SOURCE_STATE_KEY}"
    "progress_dataset.zarr_path=${DATASET_ZARR}"
    "hydra.run.dir=${OUTPUT_DIR}"
    "training.device=${GPU_DEVICE}"
    "training.num_epochs=${NUM_EPOCHS}"
    "training.gradient_accumulate_every=${GRADIENT_ACCUMULATE_EVERY}"
    "training.split_seed=${SPLIT_SEED}"
    "training.val_ratio=${VAL_RATIO}"
    "training.val_every=${VAL_EVERY}"
    "training.checkpoint_every=${CHECKPOINT_EVERY}"
    "dataloader.batch_size=${BATCH_SIZE}"
    "val_dataloader.batch_size=${VAL_BATCH_SIZE}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "val_dataloader.num_workers=${NUM_WORKERS}"
    "optimizer.lr=${LEARNING_RATE}"
    "optimizer.weight_decay=${WEIGHT_DECAY}"
    "policy.progress_hidden_dim=${PROGRESS_HIDDEN_DIM}"
    "logging.mode=${LOGGING_MODE}"
)

if [[ -n "${NUM_GRAD_STEPS}" ]]; then
    if [[ ! "${NUM_GRAD_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_GRAD_STEPS must be a positive integer, got: ${NUM_GRAD_STEPS}"
        exit 1
    fi
    HYDRA_OVERRIDES+=("training.num_grad_steps=${NUM_GRAD_STEPS}")
fi
if [[ -n "${MAX_TRAIN_STEPS}" ]]; then
    HYDRA_OVERRIDES+=("training.max_train_steps=${MAX_TRAIN_STEPS}")
fi
if [[ -n "${MAX_VAL_STEPS}" ]]; then
    HYDRA_OVERRIDES+=("training.max_val_steps=${MAX_VAL_STEPS}")
fi

echo "Starting Phase 1A ManiFlow progress training"
echo "  checkpoint: ${SOURCE_CHECKPOINT} [${SOURCE_STATE_KEY}]"
echo "  dataset:    ${DATASET_ZARR}"
echo "  output:     ${OUTPUT_DIR}"
echo "  device:     ${GPU_DEVICE}"
echo "  batches:    train=${BATCH_SIZE}, val=${VAL_BATCH_SIZE}, accumulation=${GRADIENT_ACCUMULATE_EVERY}"
echo "  effective:  $((BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY)) samples (before any short final batch)"
if [[ -n "${NUM_GRAD_STEPS}" ]]; then
    echo "  run length: ${NUM_GRAD_STEPS} optimizer steps"
else
    echo "  run length: ${NUM_EPOCHS} epochs"
fi

cd "${MANIFLOW_DIR}/maniflow/workspace"
python train_maniflow_progress_workspace.py \
    --config-name=maniflow_progress_orbit.yaml \
    "${HYDRA_OVERRIDES[@]}" \
    "$@"
