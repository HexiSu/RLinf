#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
set -Eeuo pipefail

CHECKPOINT=${1:?usage: $0 <checkpoint-dir> [publish-root]}
PUBLISH_ROOT=${2:-${POLICY_PUBLISH_ROOT:-$(dirname "${CHECKPOINT}")/published}}
CHECKPOINT=$(readlink -f -- "${CHECKPOINT}")
PUBLISH_ROOT=$(readlink -m -- "${PUBLISH_ROOT}")

[[ -d ${CHECKPOINT} ]] || { echo "checkpoint directory not found: ${CHECKPOINT}" >&2; exit 1; }
if [[ ! -f ${CHECKPOINT}/model.safetensors && ! -f ${CHECKPOINT}/model_state_dict/full_weights.pt && ! -f ${CHECKPOINT}/actor/model_state_dict/full_weights.pt ]]; then
  echo "checkpoint has no supported inference weights: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${PUBLISH_ROOT}"
version=$(basename "${CHECKPOINT}")
published=${PUBLISH_ROOT}/${version}
if [[ ! -e ${published} ]]; then
  ln -s "${CHECKPOINT}" "${published}"
fi
ln -sfn "${version}" "${PUBLISH_ROOT}/latest"
echo "Published policy ${version} via ${PUBLISH_ROOT}/latest"
