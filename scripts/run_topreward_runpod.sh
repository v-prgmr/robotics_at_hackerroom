#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
ORBIT_DIR="${ORBIT_DIR:-${WORKSPACE_DIR}/orbit}"
VENV_DIR="${TOPREWARD_VENV:-${WORKSPACE_DIR}/venvs/orbit-topreward}"
INPUT_ROOT="${TOPREWARD_INPUT_ROOT:-${ORBIT_DIR}/dataset/topReward_smoketest}"
OUTPUT_DIR="${TOPREWARD_OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/topreward_smoketest}"
CACHE_DIR="${TOPREWARD_CACHE_DIR:-${WORKSPACE_DIR}/.cache}"
STOP_POD_ON_EXIT="${STOP_POD_ON_EXIT:-false}"
STOP_POD_ON_SUCCESS_ONLY="${STOP_POD_ON_SUCCESS_ONLY:-true}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-${POD_ID:-}}"

is_true() {
    case "${1:-}" in
        true|True|TRUE|1|yes|Yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

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
mkdir -p "${WORKSPACE_DIR}/tmp" "${CACHE_DIR}/huggingface" "${CACHE_DIR}/torch"
export PYTHONPATH="${ORBIT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="false"
export TMPDIR="${WORKSPACE_DIR}/tmp"
export XDG_CACHE_HOME="${CACHE_DIR}"
export HF_HOME="${CACHE_DIR}/huggingface"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/huggingface/hub"
export TORCH_HOME="${CACHE_DIR}/torch"

MODE=(--resume)
for arg in "$@"; do
    if [[ "${arg}" == "--overwrite" ]]; then
        MODE=()
        break
    fi
done

set +e
"${VENV_DIR}/bin/python" -m topreward_orbit.score_orbit_dataset \
    --input-root "${INPUT_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    "${MODE[@]}" \
    "$@"
FINAL_EXIT_CODE=$?
set -e

if is_true "${STOP_POD_ON_EXIT}"; then
    if is_true "${STOP_POD_ON_SUCCESS_ONLY}" && [[ "${FINAL_EXIT_CODE}" -ne 0 ]]; then
        echo "Not stopping pod because scoring exited with code ${FINAL_EXIT_CODE}"
    elif [[ -z "${RUNPOD_API_KEY}" || -z "${RUNPOD_POD_ID}" ]]; then
        echo "STOP_POD_ON_EXIT=true but RUNPOD_API_KEY or RUNPOD_POD_ID is empty"
        FINAL_EXIT_CODE=1
    else
        echo "Stopping RunPod pod ${RUNPOD_POD_ID}"
        set +e
        RUNPOD_API_KEY="${RUNPOD_API_KEY}" RUNPOD_POD_ID="${RUNPOD_POD_ID}" \
            "${VENV_DIR}/bin/python" - <<'PY'
import json
import os
import urllib.error
import urllib.parse
import urllib.request

api_key = os.environ["RUNPOD_API_KEY"]
pod_id = os.environ["RUNPOD_POD_ID"]
payload = json.dumps({
    "query": "mutation StopPod($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }",
    "variables": {"input": {"podId": pod_id}},
}).encode("utf-8")

base_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "curl/8.0",
}
attempts = [
    (
        "bearer-token",
        "https://api.runpod.io/graphql",
        {**base_headers, "Authorization": f"Bearer {api_key}"},
    ),
    (
        "query-api-key",
        f"https://api.runpod.io/graphql?api_key={urllib.parse.quote(api_key)}",
        base_headers,
    ),
]

failures = []
for label, url, headers in attempts:
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        failures.append(f"{label}: HTTP {exc.code}: {body}")
        continue

    data = json.loads(body)
    if data.get("errors"):
        failures.append(f"{label}: GraphQL errors: {data['errors']}")
        continue
    pod_stop = data.get("data", {}).get("podStop")
    if not pod_stop:
        failures.append(f"{label}: missing podStop response: {data}")
        continue
    print(f"RunPod stop requested: {pod_stop}")
    break
else:
    raise RuntimeError("RunPod stop failed; attempts: " + " | ".join(failures))
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
