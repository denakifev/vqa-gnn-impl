#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
GQA_DIR="${GQA_DIR:-${REPO_ROOT}/data/gqa}"
VARIANT="${VARIANT:-full}"
MINI_N="${MINI_N:-500}"

if [[ "${VARIANT}" == "mini" ]]; then
    DEFAULT_OUTPUT_DIR="${REPO_ROOT}/kaggle_staging/gqa-gnn-data-mini"
else
    DEFAULT_OUTPUT_DIR="${REPO_ROOT}/kaggle_staging/gqa-gnn-data"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_gqa_data.py" stage \
    --data-dir "${GQA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --variant "${VARIANT}" \
    --mini-n "${MINI_N}" \
    "$@"
