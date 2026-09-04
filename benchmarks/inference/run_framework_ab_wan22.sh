#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: benchmarks/inference/run_framework_ab_wan22.sh

Run the order-balanced Wan2.2 WorldFoundry/reference framework benchmark.
The launcher takes no positional arguments; configure it with WAN22_AB_*
environment variables. Important controls include:

  WAN22_AB_GPUS                 One certifying physical GPU index (default: 0)
  WAN22_AB_ROUNDS               Number of paired AB/BA rounds (default: 6)
  WAN22_AB_REFERENCE_FRAMEWORK  fastvideo or lightx2v (default: fastvideo)
  WAN22_AB_OUTPUT_DIR           Persistent result directory
  WAN22_AB_WF_PYTHON            WorldFoundry interpreter override
                                (default: python3 from the active PATH)
  WAN22_AB_HEIGHT/WIDTH         Output resolution (defaults: 704/1280)
  WAN22_AB_FRAMES/STEPS         Video frames and denoise steps (defaults: 121/50)
  WAN22_AB_GUIDANCE             Classifier-free guidance scale (default: 5)
  WAN22_AB_VAE_WEIGHT_DTYPE     Shared resident VAE dtype, bf16/fp16 (default: bf16)
  WAN22_AB_VAE_CHANNELS_LAST_3D Shared Conv3d layout switch, 0/1 (default: 1)
  WAN22_AB_WF_INPLACE_RESIDUAL  Safe no-grad in-place Wan residuals, 0/1
                                (default: 0)
  WAN22_AB_WF_FUSE_QKV          Install merged QKV weights, 0/1 (default: 0)
  WAN22_AB_WF_QKV_STRATEGY      auto, packed, or split (default: auto)
  WAN22_AB_WF_QKV_SPLIT_THRESHOLD
                                Auto split cutoff in token rows (default: 8192)
  WAN22_AB_WF_ROPE_PRECISION    fp32 or fp64 (default: fp32)
  WAN22_AB_WF_RMS_NORM_PRECISION
                                input or fp32 (default: input)
  WAN22_AB_WF_APPROXIMATE_ATTENTION
                                none, sta, vsa, vmoba, fastvideo_sla,
                                fastvideo_sagesla, dynamic_sparse, sparge, nbhd,
                                lightx2v_sla_mask, flexblock, lightx2v_spas_sage,
                                draft_attn, radial_attn, rainfusion_attn,
                                svg_attn, svg2_attn, or lightx2v_svg_mask
  WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE_JSON
                                Complete inline VMoBA profile JSON
  WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE
                                Path to the same JSON profile (exclusive with inline)
  WAN22_AB_FV_ATTENTION_BACKEND Explicit FastVideo backend; learned-SLA lanes
                                require SLA_ATTN or SAGE_SLA_ATTN
  WAN22_AB_LX_ATTENTION_MODE    Legacy LightX2V dense self/cross mode
                                (default: flash_attn2)
  WAN22_AB_LX_SELF_ATTENTION_MODE
                                Optional explicit LightX2V self-attention mode
  WAN22_AB_LX_CROSS_ATTENTION_MODE
                                Explicit dense cross mode required by sparse self
  WAN22_AB_LX_SELF_ATTENTION_SETTINGS_JSON
                                Upstream LightX2V self-attention settings object
  WAN22_AB_SPARGE_SOURCE        SpargeAttn checkout used by Sage2-backed
                                LightX2V sparse providers
  WAN22_AB_GPU_ISOLATION_PROOF compute-exclusive (release) or bounded-polling
  WAN22_AB_ALLOW_UNPROVEN_PID_BINDING
                                Set to 1 only for diagnostic runs on hosts that
                                hide container/host PID mapping
  WAN22_AB_FAIL_ON_CONTAMINATION
                                Set to 0 only with the diagnostic option above

Use the Python entry point for detailed per-sample and summarize options:
  python -m benchmarks.inference.framework_ab_wan22 --help
EOF
}

if (($# > 0)); then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unexpected positional argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
WF_PYTHON="${WAN22_AB_WF_PYTHON:-}"
if [[ -z "${WF_PYTHON}" ]]; then
  WF_PYTHON="$(command -v python3 || true)"
fi
# Retain the historical arena environment as a last-resort path for hosts
# without python3 on PATH.  A compatibility preflight below rejects stale
# environments before any GPU work starts.
if [[ -z "${WF_PYTHON}" ]]; then
  WF_PYTHON="${ARENA_ROOT}/envs/worldfm/bin/python"
fi
FV_PYTHON="${ARENA_ROOT}/envs/fastvideo-bench/bin/python"
LX_PYTHON="${WAN22_AB_LX_PYTHON:-${ARENA_ROOT}/envs/lightx2v-bench/bin/python}"
FV_SOURCE="${REPO_ROOT}/tmp/refs/FastVideo"
LX_SOURCE="${REPO_ROOT}/tmp/refs/LightX2V"
SPARGE_SOURCE="${WAN22_AB_SPARGE_SOURCE:-${REPO_ROOT}/tmp/refs/SpargeAttn}"
CHECKPOINT="${WAN22_AB_CHECKPOINT:-${ARENA_ROOT}/ckpts/Wan2.2-TI2V-5B-Diffusers}"
WF_CHECKPOINT="${WAN22_AB_WF_CHECKPOINT:-${ARENA_ROOT}/ckpts/Wan-AI--Wan2.2-TI2V-5B}"
FV_CHECKPOINT="${WAN22_AB_FV_CHECKPOINT:-${CHECKPOINT}}"
LX_CHECKPOINT="${WAN22_AB_LX_CHECKPOINT:-${ARENA_ROOT}/ckpts/Wan-AI--Wan2.2-TI2V-5B}"
OUTPUT_DIR="${WAN22_AB_OUTPUT_DIR:-${REPO_ROOT}/benchmarks/results/20260830/framework_ab}"
# One round is one paired WorldFoundry/reference sample.  Six rounds give
# three AB and three BA pairs on the single certifying GPU.
ROUNDS="${WAN22_AB_ROUNDS:-6}"
MIN_SAMPLES="${WAN22_AB_MIN_SAMPLES:-6}"
TAG="${WAN22_AB_TAG:-release}"
REFERENCE_FRAMEWORK="${WAN22_AB_REFERENCE_FRAMEWORK:-fastvideo}"
WF_ATTENTION_BACKEND="${WAN22_AB_WF_ATTENTION_BACKEND:-flash}"
WF_VAE_DECODE_AUTOCAST="${WAN22_AB_WF_VAE_DECODE_AUTOCAST:-bf16}"
VAE_WEIGHT_DTYPE="${WAN22_AB_VAE_WEIGHT_DTYPE:-bf16}"
VAE_CHANNELS_LAST_3D="${WAN22_AB_VAE_CHANNELS_LAST_3D:-1}"
WF_DIT_WEIGHT_DTYPE="${WAN22_AB_WF_DIT_WEIGHT_DTYPE:-bf16}"
WF_QUANTIZATION="${WAN22_AB_WF_QUANTIZATION:-none}"
WF_QUANTIZATION_CONFIG_JSON="${WAN22_AB_WF_QUANTIZATION_CONFIG_JSON:-}"
WF_FEATURE_CACHE="${WAN22_AB_WF_FEATURE_CACHE:-none}"
WF_FEATURE_CACHE_OPTIONS_JSON="${WAN22_AB_WF_FEATURE_CACHE_OPTIONS_JSON:-}"
WF_APPROXIMATE_ATTENTION="${WAN22_AB_WF_APPROXIMATE_ATTENTION:-none}"
WF_APPROXIMATE_ATTENTION_PROFILE_JSON="${WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE_JSON:-}"
WF_APPROXIMATE_ATTENTION_PROFILE="${WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE:-}"
FV_ATTENTION_BACKEND="${WAN22_AB_FV_ATTENTION_BACKEND:-}"
LX_ATTENTION_MODE="${WAN22_AB_LX_ATTENTION_MODE:-flash_attn2}"
LX_SELF_ATTENTION_MODE="${WAN22_AB_LX_SELF_ATTENTION_MODE:-}"
LX_CROSS_ATTENTION_MODE="${WAN22_AB_LX_CROSS_ATTENTION_MODE:-}"
LX_SELF_ATTENTION_SETTINGS_JSON="${WAN22_AB_LX_SELF_ATTENTION_SETTINGS_JSON:-}"
WF_COMPILE="${WAN22_AB_WF_COMPILE:-0}"
WF_FUSED_ROPE="${WAN22_AB_WF_FUSED_ROPE:-0}"
WF_CUDA_GRAPH="${WAN22_AB_WF_CUDA_GRAPH:-0}"
WF_FUSE_QKV="${WAN22_AB_WF_FUSE_QKV:-0}"
WF_QKV_STRATEGY="${WAN22_AB_WF_QKV_STRATEGY:-auto}"
WF_QKV_SPLIT_THRESHOLD="${WAN22_AB_WF_QKV_SPLIT_THRESHOLD:-8192}"
WF_INPLACE_RESIDUAL="${WAN22_AB_WF_INPLACE_RESIDUAL:-0}"
WF_STATIC_CROSS_KV="${WAN22_AB_WF_STATIC_CROSS_KV:-1}"
WF_ROPE_PRECISION="${WAN22_AB_WF_ROPE_PRECISION:-fp32}"
WF_RMS_NORM_PRECISION="${WAN22_AB_WF_RMS_NORM_PRECISION:-input}"
GPU_LIST_RAW="${WAN22_AB_GPUS:-0}"
PARALLEL_GPUS="${WAN22_AB_PARALLEL_GPUS:-0}"
CROSS_FRAMEWORK_PIXEL_QUALITY="${WAN22_AB_CROSS_FRAMEWORK_PIXEL_QUALITY:-0}"
SAMPLE_TIMEOUT_SECONDS="${WAN22_AB_SAMPLE_TIMEOUT_SECONDS:-2400}"
GPU_ISOLATION_PROOF="${WAN22_AB_GPU_ISOLATION_PROOF:-compute-exclusive}"
ALLOW_UNPROVEN_PID_BINDING="${WAN22_AB_ALLOW_UNPROVEN_PID_BINDING:-0}"
FAIL_ON_CONTAMINATION="${WAN22_AB_FAIL_ON_CONTAMINATION:-1}"
HEIGHT="${WAN22_AB_HEIGHT:-704}"
WIDTH="${WAN22_AB_WIDTH:-1280}"
FRAMES="${WAN22_AB_FRAMES:-121}"
STEPS="${WAN22_AB_STEPS:-50}"
GUIDANCE="${WAN22_AB_GUIDANCE:-5}"
FPS="${WAN22_AB_FPS:-24}"
SEED="${WAN22_AB_SEED:-42}"

if [[ -z "${WF_FEATURE_CACHE_OPTIONS_JSON}" ]]; then
  WF_FEATURE_CACHE_OPTIONS_JSON='{}'
fi
if [[ -z "${WF_QUANTIZATION_CONFIG_JSON}" ]]; then
  WF_QUANTIZATION_CONFIG_JSON='{}'
fi
if [[ -z "${LX_SELF_ATTENTION_SETTINGS_JSON}" ]]; then
  LX_SELF_ATTENTION_SETTINGS_JSON='{}'
fi
if [[ -n "${WF_APPROXIMATE_ATTENTION_PROFILE_JSON}" && -n "${WF_APPROXIMATE_ATTENTION_PROFILE}" ]]; then
  echo "set only one of WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE_JSON and WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE" >&2
  exit 2
fi
if [[ "${WF_APPROXIMATE_ATTENTION}" == "vmoba" && -z "${WF_APPROXIMATE_ATTENTION_PROFILE_JSON}" && -z "${WF_APPROXIMATE_ATTENTION_PROFILE}" ]]; then
  echo "VMoBA requires WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE_JSON or WAN22_AB_WF_APPROXIMATE_ATTENTION_PROFILE" >&2
  exit 2
fi
if [[ -z "${FV_ATTENTION_BACKEND}" ]]; then
  case "${WF_APPROXIMATE_ATTENTION}" in
    fastvideo_sla)
      FV_ATTENTION_BACKEND="SLA_ATTN"
      ;;
    fastvideo_sagesla)
      FV_ATTENTION_BACKEND="SAGE_SLA_ATTN"
      ;;
    *)
      FV_ATTENTION_BACKEND="FLASH_ATTN"
      ;;
  esac
fi
case "${WF_APPROXIMATE_ATTENTION}" in
  fastvideo_sla)
    if [[ "${FV_ATTENTION_BACKEND}" != "SLA_ATTN" ]]; then
      echo "fastvideo_sla requires WAN22_AB_FV_ATTENTION_BACKEND=SLA_ATTN" >&2
      exit 2
    fi
    ;;
  fastvideo_sagesla)
    if [[ "${FV_ATTENTION_BACKEND}" != "SAGE_SLA_ATTN" ]]; then
      echo "fastvideo_sagesla requires WAN22_AB_FV_ATTENTION_BACKEND=SAGE_SLA_ATTN" >&2
      exit 2
    fi
    ;;
  *)
    if [[ "${FV_ATTENTION_BACKEND}" == "SLA_ATTN" || "${FV_ATTENTION_BACKEND}" == "SAGE_SLA_ATTN" ]]; then
      echo "${FV_ATTENTION_BACKEND} requires matching WAN22_AB_WF_APPROXIMATE_ATTENTION" >&2
      exit 2
    fi
    ;;
esac

read -r -a GPUS <<<"${GPU_LIST_RAW}"
if ((${#GPUS[@]} == 0)); then
  echo "WAN22_AB_GPUS must contain at least one physical GPU index" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "WAN22_AB_GPUS contains a non-numeric GPU index: ${gpu}" >&2
    exit 2
  fi
done
if ((${#GPUS[@]} != 1)); then
  echo "Framework A/B v4 certification requires exactly one physical GPU for all repeats; set WAN22_AB_GPUS to one index" >&2
  exit 2
fi
if [[ ! "${ROUNDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WAN22_AB_ROUNDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MIN_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WAN22_AB_MIN_SAMPLES must be a positive integer" >&2
  exit 2
fi
if ((ROUNDS < MIN_SAMPLES)); then
  echo "WAN22_AB_ROUNDS=${ROUNDS} is below WAN22_AB_MIN_SAMPLES=${MIN_SAMPLES}" >&2
  exit 2
fi
for flag in \
  "${PARALLEL_GPUS}" \
  "${WF_COMPILE}" \
  "${WF_FUSED_ROPE}" \
  "${WF_CUDA_GRAPH}" \
  "${WF_FUSE_QKV}" \
  "${WF_INPLACE_RESIDUAL}" \
  "${WF_STATIC_CROSS_KV}"; do
  if [[ "${flag}" != "0" && "${flag}" != "1" ]]; then
    echo "WorldFoundry A/B boolean environment controls must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "${VAE_CHANNELS_LAST_3D}" != "0" && "${VAE_CHANNELS_LAST_3D}" != "1" ]]; then
  echo "WAN22_AB_VAE_CHANNELS_LAST_3D must be 0 or 1" >&2
  exit 2
fi
if [[ "${VAE_WEIGHT_DTYPE}" != "bf16" && "${VAE_WEIGHT_DTYPE}" != "fp16" ]]; then
  echo "WAN22_AB_VAE_WEIGHT_DTYPE must be bf16 or fp16 for shared LightX2V parity" >&2
  exit 2
fi
if [[ "${WF_QKV_STRATEGY}" != "auto" && "${WF_QKV_STRATEGY}" != "packed" && "${WF_QKV_STRATEGY}" != "split" ]]; then
  echo "WAN22_AB_WF_QKV_STRATEGY must be auto, packed, or split" >&2
  exit 2
fi
if [[ ! "${WF_QKV_SPLIT_THRESHOLD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WAN22_AB_WF_QKV_SPLIT_THRESHOLD must be a positive integer" >&2
  exit 2
fi
if [[ "${WF_ROPE_PRECISION}" != "fp32" && "${WF_ROPE_PRECISION}" != "fp64" ]]; then
  echo "WAN22_AB_WF_ROPE_PRECISION must be fp32 or fp64" >&2
  exit 2
fi
if [[ "${WF_RMS_NORM_PRECISION}" != "input" && "${WF_RMS_NORM_PRECISION}" != "fp32" ]]; then
  echo "WAN22_AB_WF_RMS_NORM_PRECISION must be input or fp32" >&2
  exit 2
fi
if [[ "${WF_FUSED_ROPE}" == "1" && "${WF_ROPE_PRECISION}" != "fp32" ]]; then
  echo "WAN22_AB_WF_FUSED_ROPE=1 requires WAN22_AB_WF_ROPE_PRECISION=fp32" >&2
  exit 2
fi
if [[ "${WF_FUSED_ROPE}" == "1" && "${WF_RMS_NORM_PRECISION}" != "fp32" ]]; then
  echo "WAN22_AB_WF_FUSED_ROPE=1 requires WAN22_AB_WF_RMS_NORM_PRECISION=fp32" >&2
  exit 2
fi
for flag in "${ALLOW_UNPROVEN_PID_BINDING}" "${FAIL_ON_CONTAMINATION}"; do
  if [[ "${flag}" != "0" && "${flag}" != "1" ]]; then
    echo "Framework A/B isolation boolean controls must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "${GPU_ISOLATION_PROOF}" != "compute-exclusive" && "${GPU_ISOLATION_PROOF}" != "bounded-polling" ]]; then
  echo "WAN22_AB_GPU_ISOLATION_PROOF must be compute-exclusive or bounded-polling" >&2
  exit 2
fi
if [[ "${ALLOW_UNPROVEN_PID_BINDING}" == "1" && "${FAIL_ON_CONTAMINATION}" != "0" ]]; then
  echo "WAN22_AB_ALLOW_UNPROVEN_PID_BINDING=1 requires WAN22_AB_FAIL_ON_CONTAMINATION=0" >&2
  exit 2
fi
if [[ "${PARALLEL_GPUS}" != "0" ]]; then
  echo "Framework A/B v4 certification is globally serial; WAN22_AB_PARALLEL_GPUS must be 0" >&2
  exit 2
fi

if [[ "${REFERENCE_FRAMEWORK}" != "fastvideo" && "${REFERENCE_FRAMEWORK}" != "lightx2v" ]]; then
  echo "WAN22_AB_REFERENCE_FRAMEWORK must be fastvideo or lightx2v" >&2
  exit 2
fi
if [[ "${REFERENCE_FRAMEWORK}" != "fastvideo" && ( "${WF_APPROXIMATE_ATTENTION}" == "fastvideo_sla" || "${WF_APPROXIMATE_ATTENTION}" == "fastvideo_sagesla" ) ]]; then
  echo "FastVideo learned-SLA comparison requires WAN22_AB_REFERENCE_FRAMEWORK=fastvideo" >&2
  exit 2
fi
if [[ "${REFERENCE_FRAMEWORK}" == "lightx2v" && ! -x "${LX_PYTHON}" ]]; then
  echo "LightX2V Python is not executable: ${LX_PYTHON}; set WAN22_AB_LX_PYTHON" >&2
  exit 2
fi
if [[ ! -x "${WF_PYTHON}" ]]; then
  echo "WorldFoundry Python is not executable: ${WF_PYTHON}; set WAN22_AB_WF_PYTHON" >&2
  exit 2
fi
if ! "${WF_PYTHON}" - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
import re
import sys


def major_minor(raw: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", raw)
    if match is None:
        raise ValueError(raw)
    return int(match.group(1)), int(match.group(2))


try:
    transformers_version = version("transformers")
    transformers_root = distribution("transformers").locate_file("transformers")
    compatible = (
        (4, 57) <= major_minor(transformers_version) < (5, 0)
        and Path(transformers_root, "masking_utils.py").is_file()
    )
except (PackageNotFoundError, ValueError):
    compatible = False
sys.exit(0 if compatible else 1)
PY
then
  echo "WorldFoundry Python ${WF_PYTHON} requires transformers>=4.57,<5 with transformers.masking_utils; set WAN22_AB_WF_PYTHON to a compatible interpreter" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
pids=()

cleanup_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup_children EXIT INT TERM

wait_all() {
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  pids=()
  return "${status}"
}

run_sample() {
  local framework="$1"
  local gpu="$2"
  local pair_id="$3"
  local order="$4"
  local run_id="${pair_id}-${framework}"
  local python_bin="${WF_PYTHON}"
  local python_path="${REPO_ROOT}"
  if [[ "${framework}" == "fastvideo" ]]; then
    python_bin="${FV_PYTHON}"
    python_path="${FV_SOURCE}:${FV_SOURCE}/fastvideo-kernel/python:${REPO_ROOT}"
  elif [[ "${framework}" == "lightx2v" ]]; then
    python_bin="${LX_PYTHON}"
    # Pin the same SpargeAttn checkout for the reference lane as for
    # WorldFoundry. Editable environment state must not decide which Sage2
    # binary enters a supposedly paired comparison.
    python_path="${SPARGE_SOURCE}:${LX_SOURCE}:${REPO_ROOT}"
  elif [[ "${WF_APPROXIMATE_ATTENTION}" == "fastvideo_sla" || "${WF_APPROXIMATE_ATTENTION}" == "fastvideo_sagesla" ]]; then
    # WorldFoundry's learned-SLA adapter imports the exact pinned upstream
    # provider, while the kernel package lives in FastVideo's nested Python
    # root. Both paths are required; source fingerprinting alone is not an
    # executable provider capability proof.
    python_path="${FV_SOURCE}:${FV_SOURCE}/fastvideo-kernel/python:${REPO_ROOT}"
  else
    case "${WF_APPROXIMATE_ATTENTION}" in
      dynamic_sparse|sparge|nbhd|lightx2v_sla_mask|flexblock|lightx2v_spas_sage|draft_attn|radial_attn|rainfusion_attn|svg_attn|svg2_attn|lightx2v_svg_mask)
        # These lanes deliberately delegate their mask/operator contract to
        # the pinned LightX2V checkout. Sage2-backed variants additionally
        # import the compiled SpargeAttn extension from its checkout. Keep
        # both roots explicit so the WorldFoundry process cannot accidentally
        # resolve a stale ambient installation.
        python_path="${SPARGE_SOURCE}:${LX_SOURCE}:${REPO_ROOT}"
        ;;
    esac
  fi
  local -a worldfoundry_args=(
    --worldfoundry-quantization "${WF_QUANTIZATION}"
    --worldfoundry-quantization-config-json "${WF_QUANTIZATION_CONFIG_JSON}"
    --worldfoundry-feature-cache "${WF_FEATURE_CACHE}"
    --worldfoundry-feature-cache-options-json "${WF_FEATURE_CACHE_OPTIONS_JSON}"
    --worldfoundry-approximate-attention "${WF_APPROXIMATE_ATTENTION}"
    --worldfoundry-qkv-strategy "${WF_QKV_STRATEGY}"
    --worldfoundry-qkv-split-threshold "${WF_QKV_SPLIT_THRESHOLD}"
    --worldfoundry-rope-precision "${WF_ROPE_PRECISION}"
    --worldfoundry-rms-norm-precision "${WF_RMS_NORM_PRECISION}"
  )
  local -a lightx2v_args=(
    --lightx2v-attention-mode "${LX_ATTENTION_MODE}"
    --lightx2v-self-attention-settings-json
    "${LX_SELF_ATTENTION_SETTINGS_JSON}"
  )
  local -a isolation_args=(
    --gpu-isolation-proof "${GPU_ISOLATION_PROOF}"
  )
  if [[ "${ALLOW_UNPROVEN_PID_BINDING}" == "1" ]]; then
    isolation_args+=(--allow-unproven-pid-binding)
  fi
  if [[ "${FAIL_ON_CONTAMINATION}" == "0" ]]; then
    isolation_args+=(--no-fail-on-contamination)
  fi
  if [[ -n "${LX_SELF_ATTENTION_MODE}" ]]; then
    lightx2v_args+=(
      --lightx2v-self-attention-mode "${LX_SELF_ATTENTION_MODE}"
    )
  fi
  if [[ -n "${LX_CROSS_ATTENTION_MODE}" ]]; then
    lightx2v_args+=(
      --lightx2v-cross-attention-mode "${LX_CROSS_ATTENTION_MODE}"
    )
  fi
  if [[ -n "${WF_APPROXIMATE_ATTENTION_PROFILE_JSON}" ]]; then
    worldfoundry_args+=(
      --worldfoundry-approximate-attention-profile-json
      "${WF_APPROXIMATE_ATTENTION_PROFILE_JSON}"
    )
  fi
  if [[ -n "${WF_APPROXIMATE_ATTENTION_PROFILE}" ]]; then
    worldfoundry_args+=(
      --worldfoundry-approximate-attention-profile
      "${WF_APPROXIMATE_ATTENTION_PROFILE}"
    )
  fi
  if [[ "${WF_COMPILE}" == "1" ]]; then
    worldfoundry_args+=(--worldfoundry-compile)
  fi
  if [[ "${WF_FUSED_ROPE}" == "1" ]]; then
    worldfoundry_args+=(--worldfoundry-fused-rope)
  fi
  if [[ "${WF_CUDA_GRAPH}" == "1" ]]; then
    worldfoundry_args+=(--worldfoundry-cuda-graph)
  fi
  if [[ "${WF_FUSE_QKV}" == "1" ]]; then
    worldfoundry_args+=(--worldfoundry-fuse-qkv)
  else
    worldfoundry_args+=(--no-worldfoundry-fuse-qkv)
  fi
  if [[ "${WF_INPLACE_RESIDUAL}" == "1" ]]; then
    worldfoundry_args+=(--worldfoundry-inplace-residual)
  else
    worldfoundry_args+=(--no-worldfoundry-inplace-residual)
  fi
  if [[ "${WF_STATIC_CROSS_KV}" == "0" ]]; then
    worldfoundry_args+=(--no-worldfoundry-static-cross-kv)
  fi
  if [[ "${VAE_CHANNELS_LAST_3D}" == "1" ]]; then
    worldfoundry_args+=(--worldfoundry-vae-channels-last-3d)
  else
    worldfoundry_args+=(--no-worldfoundry-vae-channels-last-3d)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${python_path}" \
    FASTVIDEO_ATTENTION_BACKEND="${FV_ATTENTION_BACKEND}" \
    DTYPE="${VAE_WEIGHT_DTYPE^^}" \
    USE_CHANNELS_LAST_3D="${VAE_CHANNELS_LAST_3D}" \
    OMP_NUM_THREADS=8 NUMEXPR_MAX_THREADS=64 NUMEXPR_NUM_THREADS=8 \
    timeout --signal=TERM --kill-after=60s "${SAMPLE_TIMEOUT_SECONDS}s" \
    "${python_bin}" -m benchmarks.inference.framework_ab_wan22 run \
      --framework "${framework}" \
      --checkpoint "${CHECKPOINT}" \
      --worldfoundry-checkpoint "${WF_CHECKPOINT}" \
      --fastvideo-checkpoint "${FV_CHECKPOINT}" \
      --lightx2v-checkpoint "${LX_CHECKPOINT}" \
      --output-dir "${OUTPUT_DIR}" \
      --run-id "${run_id}" \
      --pair-id "${pair_id}" \
      --order "${order}" \
      --tag "${TAG}" \
      --physical-gpu "${gpu}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --frames "${FRAMES}" \
      --steps "${STEPS}" \
      --guidance "${GUIDANCE}" \
      --fps "${FPS}" \
      --seed "${SEED}" \
      "${isolation_args[@]}" \
      --worldfoundry-attention-backend "${WF_ATTENTION_BACKEND}" \
      --worldfoundry-vae-decode-autocast "${WF_VAE_DECODE_AUTOCAST}" \
      --worldfoundry-vae-weight-dtype "${VAE_WEIGHT_DTYPE}" \
      --worldfoundry-dit-weight-dtype "${WF_DIT_WEIGHT_DTYPE}" \
      "${worldfoundry_args[@]}" \
      --fastvideo-source "${FV_SOURCE}" \
      --fastvideo-attention-backend "${FV_ATTENTION_BACKEND}" \
      --lightx2v-source "${LX_SOURCE}" \
      "${lightx2v_args[@]}" \
      >"${OUTPUT_DIR}/logs/${run_id}.log" 2>&1
}

# Each round runs both frameworks on the same physical GPU, then alternates
# order.  All samples are globally serial so CPU, storage, and board-power
# effects cannot leak across simultaneous lanes.  The default six rounds are
# three AB plus three BA pairs.
for ((round = 0; round < ROUNDS; round++)); do
  pids=()
  for gpu_index in "${!GPUS[@]}"; do
    gpu="${GPUS[$gpu_index]}"
    if (((gpu_index + round) % 2 == 0)); then
      framework="worldfoundry"
      order="AB"
    else
      framework="${REFERENCE_FRAMEWORK}"
      order="BA"
    fi
    pair_id="r${round}-gpu${gpu}"
    if [[ "${PARALLEL_GPUS}" == "1" ]]; then
      run_sample "${framework}" "${gpu}" "${pair_id}" "${order}" &
      pids+=("$!")
    else
      run_sample "${framework}" "${gpu}" "${pair_id}" "${order}"
    fi
  done
  if [[ "${PARALLEL_GPUS}" == "1" ]]; then
    wait_all
  fi

  pids=()
  for gpu_index in "${!GPUS[@]}"; do
    gpu="${GPUS[$gpu_index]}"
    if (((gpu_index + round) % 2 == 0)); then
      framework="${REFERENCE_FRAMEWORK}"
      order="AB"
    else
      framework="worldfoundry"
      order="BA"
    fi
    pair_id="r${round}-gpu${gpu}"
    if [[ "${PARALLEL_GPUS}" == "1" ]]; then
      run_sample "${framework}" "${gpu}" "${pair_id}" "${order}" &
      pids+=("$!")
    else
      run_sample "${framework}" "${gpu}" "${pair_id}" "${order}"
    fi
  done
  if [[ "${PARALLEL_GPUS}" == "1" ]]; then
    wait_all
  fi
done

summary_args=(
  --output-dir "${OUTPUT_DIR}"
  --tag "${TAG}"
  --reference-framework "${REFERENCE_FRAMEWORK}"
  --min-samples "${MIN_SAMPLES}"
  --performance-gate
)
# Same-seed samples from independent diffusion implementations are not paired
# pixels: small scheduler differences cause valid trajectories to diverge.
# Keep full decode/shape validation mandatory, and make cross-framework
# PSNR/SSIM/LPIPS an explicit diagnostic rather than a misleading default gate.
if [[ "${CROSS_FRAMEWORK_PIXEL_QUALITY}" == "1" ]]; then
  summary_args+=(--quality)
fi
if [[ "${GPU_ISOLATION_PROOF}" == "bounded-polling" ]]; then
  # Bounded polling is intentionally diagnostic-only.  The summarizer keeps
  # such records out unless the caller explicitly acknowledges that status.
  summary_args+=(--allow-unaudited-generation-isolation)
fi
PYTHONPATH="${REPO_ROOT}" "${WF_PYTHON}" -m benchmarks.inference.framework_ab_wan22 summarize \
  "${summary_args[@]}"
