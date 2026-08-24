#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.

# Non-interactive server entry point. The robot gateway is started separately.
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

case "${PHASE}" in
  prepare)
    # Run once before submitting non-interactive training jobs.
    exec bash "${RUN_SCRIPT}" prepare
    ;;
  train)
    : "${ROBOT_PC_IP:?set ROBOT_PC_IP to the robot computer address}"
    # All files already created by prepare; no SSH/SCP or conversion is done.
    export SKIP_COPY=1
    export SKIP_CONVERT=1
    export RAY_HEAD=${RAY_HEAD:-1}
    exec bash "${RUN_SCRIPT}" train
    ;;
  *)
    echo "usage: ROBOT_PC_IP=<robot-ip> $0 {prepare|train}" >&2
    exit 2
    ;;
esac
