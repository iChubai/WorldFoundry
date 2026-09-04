#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 CONFIG [OUTPUT_DIR]" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

CONFIG=$1
OUTPUT_DIR=${2:-tmp/embodied_dlc/run}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${WORLDFOUNDRY_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
PYTHON_BIN=${WF_EMBODIED_PYTHON:-python}
CONDA_ENV=${WF_EMBODIED_CONDA_ENV:-}
BOOTSTRAP=${WF_EMBODIED_BOOTSTRAP:-0}
BOOTSTRAP_PACKAGES=${WF_EMBODIED_BOOTSTRAP_PACKAGES:-pyyaml msgpack packaging tqdm websockets}
SERVER_URL=${WF_EMBODIED_SERVER_URL:-}
SERVE_CONFIG=${WF_EMBODIED_SERVE_CONFIG:-}
SERVE_HOST=${WF_EMBODIED_SERVE_HOST:-0.0.0.0}
SERVE_PORT=${WF_EMBODIED_SERVE_PORT:-8000}
READY_TIMEOUT=${WF_EMBODIED_SERVE_READY_TIMEOUT:-1800}
PLAN_ONLY=${WF_EMBODIED_PLAN_ONLY:-0}
NO_SAVE=${WF_EMBODIED_NO_SAVE:-0}
EVAL_ID=${WF_EMBODIED_EVAL_ID:-}
MERGE_TIMEOUT=${WF_EMBODIED_MERGE_TIMEOUT:-14400}
MERGE_POLL_SECONDS=${WF_EMBODIED_MERGE_POLL_SECONDS:-2}
EXPECTED_NUM_SHARDS=${WF_EMBODIED_EXPECTED_NUM_SHARDS:-}

EXPLICIT_SHARDING=0
if [[ -n "${WF_EMBODIED_SHARD_ID:-}" || -n "${WF_EMBODIED_NUM_SHARDS:-}" ]]; then
  EXPLICIT_SHARDING=1
fi
SHARD_ID=${WF_EMBODIED_SHARD_ID:-${RANK:-}}
NUM_SHARDS=${WF_EMBODIED_NUM_SHARDS:-${WORLD_SIZE:-}}
if [[ -n "${EXPECTED_NUM_SHARDS}" && ! "${EXPECTED_NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WF_EMBODIED_EXPECTED_NUM_SHARDS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${EXPECTED_NUM_SHARDS}" ]] && (( EXPECTED_NUM_SHARDS > 1 )) && [[ -z "${SHARD_ID}" ]]; then
  echo "DLC configured ${EXPECTED_NUM_SHARDS} workers but did not provide RANK/WORLD_SIZE" >&2
  exit 2
fi
if [[ -n "${SHARD_ID}" || -n "${NUM_SHARDS}" ]]; then
  if [[ -z "${SHARD_ID}" || -z "${NUM_SHARDS}" ]]; then
    echo "shard id and shard count must be provided together (WF_EMBODIED_SHARD_* or RANK/WORLD_SIZE)" >&2
    exit 2
  fi
  if [[ ! "${SHARD_ID}" =~ ^[0-9]+$ || ! "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "shard id and shard count must be non-negative/positive integers" >&2
    exit 2
  fi
  if (( SHARD_ID >= NUM_SHARDS )); then
    echo "shard id ${SHARD_ID} must be smaller than shard count ${NUM_SHARDS}" >&2
    exit 2
  fi
  if [[ -n "${EXPECTED_NUM_SHARDS}" ]] && (( NUM_SHARDS != EXPECTED_NUM_SHARDS )); then
    echo "DLC WORLD_SIZE ${NUM_SHARDS} does not match configured workers ${EXPECTED_NUM_SHARDS}" >&2
    exit 2
  fi
fi

COORDINATED_SHARDS=0
if [[ -n "${NUM_SHARDS}" ]] && (( NUM_SHARDS > 1 )); then
  COORDINATED_SHARDS=1
elif [[ "${EXPLICIT_SHARDING}" == "0" ]]; then
  # Preserve the historical single-worker root output layout.
  SHARD_ID=
  NUM_SHARDS=
fi

STATUS_DIR=
if [[ "${COORDINATED_SHARDS}" == "1" ]]; then
  if [[ -z "${EVAL_ID}" ]]; then
    echo "WF_EMBODIED_EVAL_ID is required for coordinated multi-worker evaluation" >&2
    exit 2
  fi
  if [[ ! "${MERGE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WF_EMBODIED_MERGE_TIMEOUT must be a positive integer" >&2
    exit 2
  fi
  if [[ ! "${MERGE_POLL_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "WF_EMBODIED_MERGE_POLL_SECONDS must be a non-negative number" >&2
    exit 2
  fi
  STATUS_TOKEN=$(printf '%s' "${EVAL_ID}" | tr -c 'A-Za-z0-9._-' '_')
  STATUS_DIR="${OUTPUT_DIR}/.dlc-status-${STATUS_TOKEN}"
fi

cd "${REPO_ROOT}"
export WORLDFOUNDRY_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

if [[ -n "${CONDA_ENV}" && -n "$(command -v conda || true)" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

if [[ "${BOOTSTRAP}" == "1" ]]; then
  read -r -a BOOTSTRAP_PACKAGE_ARGS <<< "${BOOTSTRAP_PACKAGES}"
  if [[ ${#BOOTSTRAP_PACKAGE_ARGS[@]} -gt 0 ]]; then
    "${PYTHON_BIN}" -m pip install --no-cache-dir "${BOOTSTRAP_PACKAGE_ARGS[@]}"
  fi
fi

echo "WORLDFOUNDRY_REPO_ROOT=${WORLDFOUNDRY_REPO_ROOT}"
echo "CONFIG=${CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "BOOTSTRAP=${BOOTSTRAP}"
echo "MASTER_ADDR=${MASTER_ADDR:-}"
echo "MASTER_PORT=${MASTER_PORT:-}"
echo "WORLD_SIZE=${WORLD_SIZE:-}"
echo "RANK=${RANK:-}"
echo "WF_EMBODIED_SHARD_ID=${SHARD_ID}"
echo "WF_EMBODIED_NUM_SHARDS=${NUM_SHARDS}"
echo "WF_EMBODIED_EVAL_ID=${EVAL_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
fi

SERVER_PID=
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ -n "${SERVE_CONFIG}" ]]; then
  "${PYTHON_BIN}" -m worldfoundry.evaluation.tasks.embodied.model_server.serve \
    --config "${SERVE_CONFIG}" \
    --host "${SERVE_HOST}" \
    --port "${SERVE_PORT}" &
  SERVER_PID=$!
  SERVER_URL=${SERVER_URL:-ws://127.0.0.1:${SERVE_PORT}}
  "${PYTHON_BIN}" - "${SERVE_PORT}" "${READY_TIMEOUT}" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])
deadline = time.time() + int(sys.argv[2])
while time.time() < deadline:
    with socket.socket() as sock:
        sock.settimeout(2)
        try:
            sock.connect(("127.0.0.1", port))
            raise SystemExit(0)
        except OSError:
            time.sleep(2)
raise SystemExit(f"server did not open port {port}")
PY
fi

mkdir -p "${OUTPUT_DIR}"
if [[ -n "${STATUS_DIR}" ]]; then
  mkdir -p "${STATUS_DIR}"
fi

if [[ "${COORDINATED_SHARDS}" != "1" || "${SHARD_ID}" == "0" ]]; then
  "${PYTHON_BIN}" - "${CONFIG}" "${OUTPUT_DIR}" "${SERVER_URL}" <<'PY'
import json
import sys
from pathlib import Path

from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config
from worldfoundry.evaluation.tasks.embodied.materialize_rollouts import materialize_embodied_rollout_requests

config = load_canonical_embodied_config(sys.argv[1], output_dir=sys.argv[2], server_url=sys.argv[3] or None)
output_dir = Path(sys.argv[2]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
benchmarks = []
request_count = 0
for bench_cfg in config.get("benchmarks") or ():
    requests = materialize_embodied_rollout_requests(bench_cfg)
    request_count += len(requests)
    benchmarks.append(
        {
            "id": str(bench_cfg.get("id") or bench_cfg.get("benchmark_id") or "benchmark"),
            "benchmark_id": str(bench_cfg.get("benchmark_id") or bench_cfg.get("id") or "libero"),
            "request_count": len(requests),
        }
    )
payload = {
    "schema_version": "worldfoundry-embodied-eval-plan",
    "config_path": str(Path(sys.argv[1]).resolve()),
    "output_dir": str(output_dir),
    "model_id": str((config.get("model") or {}).get("id") or config.get("model_id") or "openvla"),
    "server_url": sys.argv[3] or None,
    "benchmark_count": len(benchmarks),
    "request_count": request_count,
    "benchmarks": benchmarks,
}
(output_dir / "embodied_plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
fi

if [[ "${PLAN_ONLY}" == "1" ]]; then
  exit 0
fi

if "${PYTHON_BIN}" - "${CONFIG}" "${OUTPUT_DIR}" "${SERVER_URL}" "${SHARD_ID}" "${NUM_SHARDS}" "${EVAL_ID}" "${NO_SAVE}" <<'PY'
import json
import sys

from worldfoundry.evaluation.tasks.embodied.orchestrator import run_embodied_eval_config


def _optional_int(value: str):
    return int(value) if value else None


result = run_embodied_eval_config(
    sys.argv[1],
    output_dir=sys.argv[2],
    server_url=sys.argv[3] or None,
    shard_id=_optional_int(sys.argv[4]),
    num_shards=_optional_int(sys.argv[5]),
    eval_id=sys.argv[6] or None,
    no_docker=True,
    no_save=sys.argv[7] == "1",
)
print(json.dumps(result.to_dict(), sort_keys=True))
raise SystemExit(int(result.evaluate_result.exit_code))
PY
then
  EVAL_RC=0
else
  EVAL_RC=$?
fi

if [[ "${COORDINATED_SHARDS}" != "1" || "${NO_SAVE}" == "1" ]]; then
  exit "${EVAL_RC}"
fi

STATUS_FILE="${STATUS_DIR}/rank${SHARD_ID}.status"
STATUS_TMP="${STATUS_FILE}.tmp.${BASHPID}"
printf '%s\n' "${EVAL_RC}" > "${STATUS_TMP}"
mv "${STATUS_TMP}" "${STATUS_FILE}"

if [[ "${EVAL_RC}" != "0" ]]; then
  exit "${EVAL_RC}"
fi
if [[ "${SHARD_ID}" != "0" ]]; then
  exit 0
fi

DEADLINE=$(( $(date +%s) + MERGE_TIMEOUT ))
while true; do
  ALL_FINISHED=1
  for ((rank = 0; rank < NUM_SHARDS; rank++)); do
    RANK_STATUS_FILE="${STATUS_DIR}/rank${rank}.status"
    if [[ ! -f "${RANK_STATUS_FILE}" ]]; then
      ALL_FINISHED=0
      continue
    fi
    read -r RANK_RC < "${RANK_STATUS_FILE}"
    if [[ ! "${RANK_RC}" =~ ^[0-9]+$ ]]; then
      echo "invalid DLC shard status in ${RANK_STATUS_FILE}: ${RANK_RC}" >&2
      exit 1
    fi
    if [[ "${RANK_RC}" != "0" ]]; then
      echo "DLC embodied shard ${rank}/${NUM_SHARDS} failed with exit code ${RANK_RC}" >&2
      exit "${RANK_RC}"
    fi
  done
  if [[ "${ALL_FINISHED}" == "1" ]]; then
    break
  fi
  if (( $(date +%s) >= DEADLINE )); then
    echo "timed out waiting ${MERGE_TIMEOUT}s for ${NUM_SHARDS} DLC embodied shards in ${STATUS_DIR}" >&2
    exit 1
  fi
  sleep "${MERGE_POLL_SECONDS}"
done

MERGE_ARGS=(
  -m worldfoundry.cli.main embodied merge
  --config "${CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --eval-id "${EVAL_ID}"
)
"${PYTHON_BIN}" "${MERGE_ARGS[@]}"
