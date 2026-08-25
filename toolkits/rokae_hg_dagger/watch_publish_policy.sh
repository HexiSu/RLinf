#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
set -Eeuo pipefail

CHECKPOINT_ROOT=${1:?checkpoint root required}
PUBLISH_ROOT=${2:?publish root required}
INTERVAL_S=${3:-10}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_CHECKPOINT=${4:?base inference checkpoint required}
declare -A published=()

while true; do
  shopt -s nullglob
  for actor in "${CHECKPOINT_ROOT}"/global_step_*/actor; do
    [[ -d ${actor} ]] || continue
    step=$(basename "$(dirname "${actor}")")
    [[ ${published[${step}]:-0} == 1 ]] && continue
    complete=0
    [[ -f ${actor}/model_state_dict/full_weights.pt ]] && complete=1
    [[ -f ${actor}/actor/model_state_dict/full_weights.pt ]] && complete=1
    [[ -f ${actor}/model.safetensors ]] && complete=1
    [[ ${complete} == 1 ]] || continue
    # A checkpoint can still be flushed after its weight file appears. Require
    # a stable size across two polls before publishing it.
    size1=$(du -sb "${actor}" | awk '{print $1}')
    sleep 2
    size2=$(du -sb "${actor}" | awk '{print $1}')
    [[ ${size1} == ${size2} ]] || continue
    staging=$(mktemp -d "${PUBLISH_ROOT}/.publish.XXXXXX")
    version_dir="${staging}/${step}"
    "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/export_actor_inference.py" \
      --actor "${actor}" --base "${BASE_CHECKPOINT}" --output "${version_dir}"
    final_dir="${PUBLISH_ROOT}/${step}"
    mv "${version_dir}" "${final_dir}"
    rmdir "${staging}"
    bash "${SCRIPT_DIR}/publish_policy_checkpoint.sh" "${final_dir}" "${PUBLISH_ROOT}"
    published[${step}]=1
  done
  sleep "${INTERVAL_S}"
done
