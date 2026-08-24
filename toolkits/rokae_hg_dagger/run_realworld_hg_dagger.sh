#!/usr/bin/env bash
# Complete ROKAE pi0.5 HG-DAgger workflow.
#
# The script runs on the RLinf/training machine. The ROKAE gateway normally
# runs separately on the robot computer; use GATEWAY_CMD only when both are on
# the same host. No command in this script writes to the remote checkpoint
# server: SSH/SCP are read-only operations.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RLINF_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
# Read-only OpenPI tree that produced the step-3000 checkpoint.
OPENPI_ROOT=${OPENPI_ROOT:-"/vepfs-1/users/piaoweiyi/Projects/openpi"}

REMOTE_HOST=${REMOTE_HOST:-root@115.191.63.139}
REMOTE_PORT=${REMOTE_PORT:-11557}
REMOTE_CHECKPOINT=${REMOTE_CHECKPOINT:-"/vepfs-1/runs/schaeffler3d/schaeffler3d_cr_orange_round_front_left_pi05_rtc_v1_20260811_001/checkpoints/schaeffler3d_cr_orange_round_front_left_pi05_rtc_v1_20260811_001/schaeffler3d_cr_orange_round_front_left_pi05_rtc_v1_20260811_001/3000"}

RUN_NAME=${RUN_NAME:-"schaeffler3d_cr_orange_round_front_left_pi05_rtc_hgdagger_v1_20260824_001"}
RUN_ROOT=${RUN_ROOT:-"/vepfs-1/runs/schaeffler3d/${RUN_NAME}"}
WORK_ROOT=${WORK_ROOT:-"${RUN_ROOT}"}
JAX_CHECKPOINT=${JAX_CHECKPOINT:-"${RUN_ROOT}/jax_step3000"}
TORCH_CHECKPOINT=${TORCH_CHECKPOINT:-"${RUN_ROOT}/pi05_step3000_torch"}
CONFIG_NAME=${CONFIG_NAME:-pi05_rokae_joint_horizon50}
RLINF_CONFIG=${RLINF_CONFIG:-rokae_hg_dagger_pi05}
ROBOT_PC_IP=${ROBOT_PC_IP:-}
GATEWAY_ADDRESS=${GATEWAY_ADDRESS:-}
DATA_PATH=${DATA_PATH:-"${RUN_ROOT}/online_lerobot"}
LOG_PATH=${LOG_PATH:-"${RUN_ROOT}"}

# OPENPI_PYTHON must point to the uv/virtualenv interpreter that has openpi,
# orbax, flax and tyro. RLINF_PYTHON must have RLinf, OpenPI, Ray and embodied
# deps because actor/rollout workers import OpenPI at runtime.
OPENPI_PYTHON=${OPENPI_PYTHON:-"$(command -v python)"}
RLINF_PYTHON=${RLINF_PYTHON:-"${OPENPI_PYTHON}"}
GATEWAY_CMD=${GATEWAY_CMD:-}
START_LOCAL_GATEWAY=${START_LOCAL_GATEWAY:-0}
RAY_ADDRESS=${RAY_ADDRESS:-auto}
RAY_HEAD=${RAY_HEAD:-0}
INSTALL_DEPS=${INSTALL_DEPS:-0}
SKIP_COPY=${SKIP_COPY:-0}
SKIP_CONVERT=${SKIP_CONVERT:-0}
SKIP_RAY=${SKIP_RAY:-0}

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%F %T')] $*"; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

usage() {
  cat <<'EOF'
Usage: run_realworld_hg_dagger.sh [phase]

Phases:
  all       copy checkpoint, convert, validate, start gateway (optional), train
  prepare   copy checkpoint, convert and validate only
  train     validate an existing torch checkpoint and start RLinf training
  gateway   start the gateway using GATEWAY_CMD or gateway_example.yaml

Required for `all`/`prepare`: OPENPI_PYTHON (or an activated OpenPI uv env).
Required for `train`: RLINF_PYTHON with RLinf, OpenPI and Ray installed.

Important environment variables:
  ROBOT_PC_IP       Robot computer IP, used to override env.train/eval gateway address.
  GATEWAY_ADDRESS   Full ZeroMQ address, e.g. tcp://192.168.1.20:5560.
  OPENPI_PYTHON     Python executable from the OpenPI environment.
  RLINF_PYTHON      Python executable from the RLinf/Ray environment.
  GATEWAY_CMD       Optional command for a gateway already configured on this host.
  START_LOCAL_GATEWAY=1  Use GATEWAY_PYTHON and GATEWAY_CONFIG on this host.
  GATEWAY_PYTHON    Python executable for a local gateway (default: RLINF_PYTHON).
  GATEWAY_CONFIG    Local gateway YAML (default: gateway_example.yaml).
  REMOTE_CHECKPOINT Remote step-3000 directory (read-only SCP source).
  RUN_NAME          Run directory name under /vepfs-1/runs/schaeffler3d.
  RUN_ROOT          Complete output root for conversion, logs, data and checkpoints.
  JAX_CHECKPOINT    Local directory containing the copied `params/` and `assets/`.
  TORCH_CHECKPOINT  Local converted HuggingFace-style checkpoint directory.
  INSTALL_DEPS=1    Install pytest into RLINF_PYTHON and print missing runtime deps.

Examples:
  OPENPI_PYTHON=/path/to/openpi/.venv/bin/python \
  RLINF_PYTHON=/path/to/rlinf/.venv/bin/python \
  ROBOT_PC_IP=192.168.1.20 bash toolkits/rokae_hg_dagger/run_realworld_hg_dagger.sh all

  SKIP_COPY=1 SKIP_CONVERT=1 ROBOT_PC_IP=192.168.1.20 \
  bash toolkits/rokae_hg_dagger/run_realworld_hg_dagger.sh train
EOF
}

phase=${1:-all}
case "${phase}" in
  all|prepare|train|gateway|help|-h|--help) ;;
  *) usage; die "unknown phase: ${phase}" ;;
esac
[[ ${phase} == help || ${phase} == -h || ${phase} == --help ]] && { usage; exit 0; }

need_cmd ssh
need_cmd scp
need_cmd sed
need_cmd awk

if [[ ${INSTALL_DEPS} == 1 ]]; then
  log "Installing pytest in ${RLINF_PYTHON}"
  "${RLINF_PYTHON}" -m pip install --upgrade pytest
fi

if [[ -z ${GATEWAY_ADDRESS} && -n ${ROBOT_PC_IP} ]]; then
  GATEWAY_ADDRESS="tcp://${ROBOT_PC_IP}:5560"
fi

validate_torch_checkpoint() {
  [[ -f ${TORCH_CHECKPOINT}/model.safetensors ]] || die "missing ${TORCH_CHECKPOINT}/model.safetensors"
  [[ -f ${TORCH_CHECKPOINT}/config.json ]] || die "missing ${TORCH_CHECKPOINT}/config.json"
  "${RLINF_PYTHON}" - "${TORCH_CHECKPOINT}/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
expected = {"action_dim": 32, "action_horizon": 50, "discrete_state_input": False, "max_token_len": 180}
bad = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
if bad:
    raise SystemExit(f"checkpoint config mismatch: {bad}")
print("checkpoint config: ok")
PY
}

copy_checkpoint() {
  [[ ${SKIP_COPY} == 1 ]] && { log "Skipping SCP (SKIP_COPY=1)"; return; }
  mkdir -p "${WORK_ROOT}"
  [[ -e ${JAX_CHECKPOINT} ]] && die "refusing to overwrite existing ${JAX_CHECKPOINT}; set SKIP_COPY=1 or choose another JAX_CHECKPOINT"
  log "Checking read-only SSH access to ${REMOTE_HOST}:${REMOTE_PORT}"
  ssh -p "${REMOTE_PORT}" -o BatchMode=yes "${REMOTE_HOST}" "test -d '${REMOTE_CHECKPOINT}/params' && test -d '${REMOTE_CHECKPOINT}/assets'"
  log "Copying remote step-3000 checkpoint to ${JAX_CHECKPOINT}"
  mkdir -p "${JAX_CHECKPOINT}"
  scp -P "${REMOTE_PORT}" -r \
    "${REMOTE_HOST}:${REMOTE_CHECKPOINT}/params" \
    "${REMOTE_HOST}:${REMOTE_CHECKPOINT}/assets" \
    "${JAX_CHECKPOINT}/"
  [[ -d ${JAX_CHECKPOINT}/params ]] || die "SCP result has no params/: ${JAX_CHECKPOINT}"
  [[ -d ${JAX_CHECKPOINT}/assets ]] || die "SCP result has no assets/: ${JAX_CHECKPOINT}"
}

convert_checkpoint() {
  [[ ${SKIP_CONVERT} == 1 ]] && { log "Skipping conversion (SKIP_CONVERT=1)"; return; }
  [[ -n ${OPENPI_PYTHON} ]] || die "OPENPI_PYTHON is required for conversion"
  [[ -x ${OPENPI_PYTHON} ]] || die "OPENPI_PYTHON is not executable: ${OPENPI_PYTHON}"
  [[ -d ${OPENPI_ROOT}/src/openpi ]] || die "read-only OpenPI source not found: ${OPENPI_ROOT}"
  [[ -d ${JAX_CHECKPOINT}/params ]] || die "missing JAX checkpoint params/: ${JAX_CHECKPOINT}"
  [[ ! -e ${TORCH_CHECKPOINT} ]] || die "refusing to overwrite ${TORCH_CHECKPOINT}; choose a new path"
  log "Converting JAX checkpoint with ${OPENPI_PYTHON}"
  (cd "${RLINF_ROOT}" && PYTHONPATH="${OPENPI_ROOT}/src:${RLINF_ROOT}:${PYTHONPATH:-}" "${OPENPI_PYTHON}" -m rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python \
    --checkpoint-dir "${JAX_CHECKPOINT}" \
    --config-name "${CONFIG_NAME}" \
    --output-path "${TORCH_CHECKPOINT}" \
    --precision bfloat16)
}

start_gateway() {
  if [[ -z ${GATEWAY_CMD} && ${START_LOCAL_GATEWAY} == 1 ]]; then
    local gateway_python=${GATEWAY_PYTHON:-${RLINF_PYTHON}}
    local gateway_config=${GATEWAY_CONFIG:-${SCRIPT_DIR}/gateway_example.yaml}
    GATEWAY_CMD="${gateway_python} ${SCRIPT_DIR}/gateway.py --config_path=${gateway_config}"
  fi
  [[ -n ${GATEWAY_CMD} ]] || {
    log "GATEWAY_CMD is empty; gateway must be started separately on the robot computer."
    return 0
  }
  log "Starting gateway: ${GATEWAY_CMD}"
  bash -lc "${GATEWAY_CMD}" &
  GATEWAY_PID=$!
  trap 'kill "${GATEWAY_PID}" 2>/dev/null || true' EXIT INT TERM
  sleep "${GATEWAY_STARTUP_S:-2}"
}

start_ray() {
  [[ ${SKIP_RAY} == 1 ]] && { log "Skipping Ray startup (SKIP_RAY=1)"; return; }
  "${RLINF_PYTHON}" -c 'import ray' >/dev/null 2>&1 || die "${RLINF_PYTHON} cannot import ray"
  if [[ ${RAY_HEAD} == 1 ]]; then
    log "Starting Ray head"
    "${RLINF_PYTHON}" -m ray start --head --port="${RAY_PORT:-6379}" --disable-usage-stats
  elif [[ ${RAY_ADDRESS} != auto ]]; then
    log "Joining Ray at ${RAY_ADDRESS}"
    "${RLINF_PYTHON}" -m ray start --address="${RAY_ADDRESS}" --disable-usage-stats
  else
    "${RLINF_PYTHON}" -m ray status >/dev/null 2>&1 || die "Ray is not running; set RAY_HEAD=1 or RAY_ADDRESS=<head>:6379"
  fi
}

start_training() {
  [[ -n ${ROBOT_PC_IP} || -n ${GATEWAY_ADDRESS} ]] || die "set ROBOT_PC_IP or GATEWAY_ADDRESS"
  validate_torch_checkpoint
  start_ray
  local address_override=""
  local norm_stats_path
  norm_stats_path=$(find "${TORCH_CHECKPOINT}/assets" -type f -name norm_stats.json -print -quit 2>/dev/null || true)
  [[ -n ${norm_stats_path} ]] || die "no norm_stats.json found under ${TORCH_CHECKPOINT}/assets"
  [[ -n ${GATEWAY_ADDRESS} ]] && address_override="env.train.gateway.address=${GATEWAY_ADDRESS} env.eval.gateway.address=${GATEWAY_ADDRESS}"
  log "Starting RLinf HG-DAgger training"
  cd "${RLINF_ROOT}"
  PYTHONPATH="${OPENPI_ROOT}/src:${RLINF_ROOT}:${PYTHONPATH:-}" \
    "${RLINF_PYTHON}" examples/embodiment/train_async.py \
      --config-path examples/embodiment/config \
      --config-name "${RLINF_CONFIG}" \
      runner.logger.log_path="${LOG_PATH}" \
      algorithm.dagger.online_lerobot.data_path="${DATA_PATH}" \
      actor.model.model_path="${TORCH_CHECKPOINT}" \
      rollout.model.model_path="${TORCH_CHECKPOINT}" \
      actor.model.openpi_data.norm_stats_path="${norm_stats_path}" \
      ${address_override}
}

case "${phase}" in
  prepare)
    copy_checkpoint
    convert_checkpoint
    validate_torch_checkpoint
    ;;
  gateway)
    start_gateway
    wait
    ;;
  train)
    start_training
    ;;
  all)
    copy_checkpoint
    convert_checkpoint
    validate_torch_checkpoint
    start_gateway
    start_training
    ;;
esac
