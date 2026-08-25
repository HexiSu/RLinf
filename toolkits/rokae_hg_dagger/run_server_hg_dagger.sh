#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.

# Single-entry non-interactive local-rollout server job.
set -Eeuo pipefail

PHASE=${1:-train}
ROBOT_PC_IP=${ROBOT_PC_IP:-}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUN_SCRIPT=${SCRIPT_DIR}/run_realworld_hg_dagger.sh

export OPENPI_ROOT=${OPENPI_ROOT:-/vepfs-1/users/piaoweiyi/Projects/openpi}
export OPENPI_PYTHON=${OPENPI_PYTHON:-/root/miniconda3/bin/python}
export RLINF_PYTHON=${RLINF_PYTHON:-/root/miniconda3/bin/python}
export RUN_NAME=${RUN_NAME:-schaeffler3d_cr_orange_round_front_left_pi05_rtc_hgdagger_v1_20260824_001}
export ROBOT_PC_IP
export RUN_ROOT=${RUN_ROOT:-/vepfs-1/runs/schaeffler3d/${RUN_NAME}}
export RLINF_ROOT=${RLINF_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
export EPISODE_ROOT=${EPISODE_ROOT:-${RUN_ROOT}/robot_episodes}
export DATA_PATH=${DATA_PATH:-${RUN_ROOT}/online_lerobot}
export POLICY_PUBLISH_ROOT=${POLICY_PUBLISH_ROOT:-${RUN_ROOT}/published}

case "${PHASE}" in
  prepare)
    # Run once before submitting non-interactive training jobs.
    bash "${RUN_SCRIPT}" prepare
    "${RLINF_PYTHON}" "${SCRIPT_DIR}/seed_global_step0.py" \
      --source "${RUN_ROOT}/pi05_step3000_torch" \
      --output "${RUN_ROOT}/rokae_hg_dagger_pi05_step3000/checkpoints/global_step_0"
    ;;
  train)
    # All services are children of this one non-interactive job and are cleaned up
    # when the training process exits.
    export SKIP_COPY=1
    export SKIP_CONVERT=1
    export LOCAL_ROLLOUT=1
    export SEED_RESUME_DIR=${SEED_RESUME_DIR:-${RUN_ROOT}/rokae_hg_dagger_pi05_step3000/checkpoints/global_step_0}
    export RAY_HEAD=${RAY_HEAD:-1}
    mkdir -p "${EPISODE_ROOT}" "${DATA_PATH}" "${POLICY_PUBLISH_ROOT}"
    if [[ ! -e ${POLICY_PUBLISH_ROOT}/latest ]]; then
      ln -s "${RUN_ROOT}/pi05_step3000_torch" "${POLICY_PUBLISH_ROOT}/latest"
    fi
    pids=()
    cleanup() {
      trap - EXIT INT TERM
      for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done
      wait || true
    }
    trap cleanup EXIT INT TERM
    "${RLINF_PYTHON}" "${SCRIPT_DIR}/trajectory_receiver.py" \
      --root "${EPISODE_ROOT}" --host "${TRAJECTORY_HOST:-0.0.0.0}" \
      --port "${TRAJECTORY_PORT:-8766}" >"${RUN_ROOT}/trajectory_receiver.log" 2>&1 &
    pids+=("$!")
    "${RLINF_PYTHON}" "${SCRIPT_DIR}/policy_sync_server.py" \
      --latest "${POLICY_PUBLISH_ROOT}/latest" --host "${POLICY_HOST:-0.0.0.0}" \
      --port "${POLICY_PORT:-8765}" >"${RUN_ROOT}/policy_sync_server.log" 2>&1 &
    pids+=("$!")
    "${RLINF_PYTHON}" "${SCRIPT_DIR}/convert_npz_episodes.py" \
      --input "${EPISODE_ROOT}" --output "${DATA_PATH}" --fps 30 --watch \
      >"${RUN_ROOT}/episode_converter.log" 2>&1 &
    pids+=("$!")
    PYTHON_BIN="${RLINF_PYTHON}" "${SCRIPT_DIR}/watch_publish_policy.sh" \
      "${RUN_ROOT}/rokae_hg_dagger_pi05_step3000/checkpoints" \
      "${POLICY_PUBLISH_ROOT}" "${POLICY_WATCH_INTERVAL_S:-10}" \
      "${RUN_ROOT}/pi05_step3000_torch" \
      >"${RUN_ROOT}/policy_publish.log" 2>&1 &
    pids+=("$!")
    sleep "${SERVICE_STARTUP_S:-2}"
    if [[ ${LOCAL_TRAIN_ONLY:-1} == 1 ]]; then
      cd "${RLINF_ROOT}"
      PYTHONPATH="${OPENPI_ROOT}/src:${RLINF_ROOT}:${PYTHONPATH:-}" \
        "${RLINF_PYTHON}" examples/embodiment/train_local_dagger.py \
          --config-path examples/embodiment/config \
          --config-name rokae_hg_dagger_pi05 \
          runner.logger.log_path="${RUN_ROOT}" \
          algorithm.dagger.online_lerobot.data_path="${DATA_PATH}" \
          actor.model.model_path="${RUN_ROOT}/pi05_step3000_torch" \
          rollout.model.model_path="${RUN_ROOT}/pi05_step3000_torch" \
          actor.model.openpi_data.norm_stats_path="${RUN_ROOT}/pi05_step3000_torch/assets/schaeffler3d_cr_orange_round_front_left_robot1_cr_spacemouse_v1_20260811/2.1/norm_stats.json" \
          runner.resume_dir="${RESUME_DIR:-${SEED_RESUME_DIR}}"
    else
      bash "${RUN_SCRIPT}" train
    fi
    ;;
  *)
    echo "usage: $0 {prepare|train}" >&2
    exit 2
    ;;
esac
