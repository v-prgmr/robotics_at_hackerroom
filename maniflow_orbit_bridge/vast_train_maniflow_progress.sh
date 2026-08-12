#!/usr/bin/env bash
set -uo pipefail

# Vast.ai wrapper for Phase 1A frozen ManiFlow progress-head training.
# The shared progress launcher performs environment, checkpoint, dataset, and
# dependency validation. This wrapper adds Vast persistent paths and optional
# instance destruction after training finishes.

is_true() {
    case "${1:-}" in
        true|True|TRUE|1|yes|Yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
export ORBIT_DIR="${ORBIT_DIR:-${ORBIT_REPO_DIR}}"
export MANIFLOW_DIR="${MANIFLOW_DIR:-${WORKSPACE_DIR}/maniflow}"
export MINICONDA_DIR="${MINICONDA_DIR:-${WORKSPACE_DIR}/miniconda3}"
export CONDA_ENV="${CONDA_ENV:-maniflow}"
export CONDA_ENV_DIR="${CONDA_ENV_DIR:-${WORKSPACE_DIR}/conda_envs/${CONDA_ENV}}"
export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/huggingface}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

export RUN_NAME="${RUN_NAME:-maniflow_progress_phase1a}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/train/${RUN_NAME}}"

VAST_DESTROY_ON_EXIT="${VAST_DESTROY_ON_EXIT:-false}"
VAST_DESTROY_ON_SUCCESS_ONLY="${VAST_DESTROY_ON_SUCCESS_ONLY:-true}"
VAST_API_KEY="${VAST_API_KEY:-}"
VAST_INSTANCE_ID="${VAST_INSTANCE_ID:-}"
VAST_API_BASE_URL="${VAST_API_BASE_URL:-https://console.vast.ai}"
VAST_DESTROY_HELPER="${VAST_DESTROY_HELPER:-}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${SCRIPT_DIR}/runpod_train_maniflow_progress.sh}"

if is_true "${VAST_DESTROY_ON_EXIT}"; then
    if [[ -z "${VAST_API_KEY}" || -z "${VAST_INSTANCE_ID}" ]]; then
        echo "VAST_DESTROY_ON_EXIT=true requires VAST_API_KEY and VAST_INSTANCE_ID."
        exit 1
    fi
    if [[ ! "${VAST_INSTANCE_ID}" =~ ^[0-9]+$ ]]; then
        echo "VAST_INSTANCE_ID must be an integer, got: ${VAST_INSTANCE_ID}"
        exit 1
    fi
fi

echo "Starting Vast.ai Phase 1A ManiFlow progress workflow"
echo "  persistent root: ${WORKSPACE_DIR}"
echo "  Orbit repo:      ${ORBIT_DIR}"
echo "  ManiFlow repo:   ${MANIFLOW_DIR}"
echo "  output:          ${OUTPUT_DIR}"
echo "  destroy on exit: ${VAST_DESTROY_ON_EXIT}"
echo "  success only:    ${VAST_DESTROY_ON_SUCCESS_ONLY}"

set +e
bash "${TRAIN_LAUNCHER}" "$@"
TRAIN_EXIT_CODE=$?
set -e

FINAL_EXIT_CODE="${TRAIN_EXIT_CODE}"
should_destroy=false
if is_true "${VAST_DESTROY_ON_EXIT}"; then
    if is_true "${VAST_DESTROY_ON_SUCCESS_ONLY}" && [[ "${TRAIN_EXIT_CODE}" -ne 0 ]]; then
        echo "Not destroying Vast instance because training exited with code ${TRAIN_EXIT_CODE}."
    else
        should_destroy=true
    fi
fi

if is_true "${should_destroy}"; then
    echo "Destroying Vast.ai instance ${VAST_INSTANCE_ID}"
    set +e
    if [[ -n "${VAST_DESTROY_HELPER}" ]]; then
        "${VAST_DESTROY_HELPER}" "${VAST_INSTANCE_ID}" "${VAST_API_KEY}"
        DESTROY_EXIT_CODE=$?
    else
        VAST_API_KEY="${VAST_API_KEY}" \
        VAST_INSTANCE_ID="${VAST_INSTANCE_ID}" \
        VAST_API_BASE_URL="${VAST_API_BASE_URL}" \
        python - <<'PY'
import json
import os
import urllib.error
import urllib.request

api_key = os.environ["VAST_API_KEY"]
instance_id = int(os.environ["VAST_INSTANCE_ID"])
base_url = os.environ["VAST_API_BASE_URL"].rstrip("/")
request = urllib.request.Request(
    f"{base_url}/api/v0/instances/{instance_id}",
    headers={
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "orbit-vast-progress-launcher/1.0",
    },
    method="DELETE",
)
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"Vast destroy failed with HTTP {exc.code}: {body}") from exc

payload = json.loads(body)
if payload.get("success") is not True:
    raise RuntimeError(f"Vast destroy returned an unsuccessful response: {payload}")
print(f"Vast destroy requested successfully: {payload}")
PY
        DESTROY_EXIT_CODE=$?
    fi
    set -e
    if [[ "${DESTROY_EXIT_CODE}" -ne 0 ]]; then
        echo "Vast.ai instance destruction failed with code ${DESTROY_EXIT_CODE}."
        FINAL_EXIT_CODE=1
    fi
fi

exit "${FINAL_EXIT_CODE}"
