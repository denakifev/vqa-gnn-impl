#!/usr/bin/env bash
# Short real-data GQA smoke test for VQA-GNN on Kaggle.
#
# Expects the Kaggle Dataset mounted at /kaggle/input/gqa-gnn-data
# or at GQA_DATA_DIR.
#
# USAGE:
#   bash scripts/run_kaggle_smoke_test.sh
#
# Offline text encoder example:
#   OFFLINE=1 \
#   TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
#   bash scripts/run_kaggle_smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GQA_DATA_DIR="${GQA_DATA_DIR:-/kaggle/input/gqa-gnn-data}"
SAVE_DIR="${SAVE_DIR:-/kaggle/working/saved}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
N_EPOCHS="${N_EPOCHS:-1}"
EPOCH_LEN="${EPOCH_LEN:-50}"
VALIDATE_SAMPLE_SIZE="${VALIDATE_SAMPLE_SIZE:-50}"
FREEZE_TEXT_ENCODER="${FREEZE_TEXT_ENCODER:-true}"
D_KG="${D_KG:-600}"
MAX_KG_NODES="${MAX_KG_NODES:-100}"
NUM_ANSWERS="${NUM_ANSWERS:-1842}"
RUN_NAME="${RUN_NAME:-smoke_gqa}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-}"

export COMET_API_KEY="${COMET_API_KEY:-offline_placeholder}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

if [[ "${OFFLINE:-0}" == "1" ]]; then
    export TRANSFORMERS_OFFLINE=1
fi

echo "========================================================"
echo "VQA-GNN GQA Kaggle smoke test"
echo "  Data dir:   ${GQA_DATA_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epoch len:  ${EPOCH_LEN}"
echo "  Epochs:     ${N_EPOCHS}"
echo "  Save dir:   ${SAVE_DIR}"
echo "========================================================"

mkdir -p "${SAVE_DIR}"

echo ""
echo "[1/3] Checking required GQA files..."
required_paths=(
    "${GQA_DATA_DIR}/gqa_answer_vocab.json"
    "${GQA_DATA_DIR}/gqa_relation_vocab.json"
    "${GQA_DATA_DIR}/questions/train_balanced_questions.json"
    "${GQA_DATA_DIR}/questions/val_balanced_questions.json"
    "${GQA_DATA_DIR}/knowledge_graphs/train_graphs.h5"
    "${GQA_DATA_DIR}/knowledge_graphs/val_graphs.h5"
    "${GQA_DATA_DIR}/visual_features/train_features.h5"
    "${GQA_DATA_DIR}/visual_features/val_features.h5"
)

for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] Required path not found: ${path}" >&2
        echo "        Restore or mount the private GQA Kaggle dataset first." >&2
        exit 1
    fi
    echo "  [OK] ${path}"
done

echo ""
echo "[2/3] Validating GQA data structure..."
validate_cmd=(
    "${PYTHON_BIN}" scripts/validate_gqa_data.py
    --data-dir "${GQA_DATA_DIR}"
    --split train val
    --answer-vocab "${GQA_DATA_DIR}/gqa_answer_vocab.json"
    --relation-vocab "${GQA_DATA_DIR}/gqa_relation_vocab.json"
    --num-visual-nodes 100
    --feature-dim 2048
    --d-kg "${D_KG}"
    --num-answers "${NUM_ANSWERS}"
    --max-kg-nodes "${MAX_KG_NODES}"
    --sample-size "${VALIDATE_SAMPLE_SIZE}"
)

if [[ -n "${TEXT_ENCODER_PATH}" ]]; then
    validate_cmd+=("--text-encoder" "${TEXT_ENCODER_PATH}")
else
    validate_cmd+=("--skip-runtime-check")
fi

"${validate_cmd[@]}"

echo ""
echo "[3/3] Launching GQA smoke-train: ${RUN_NAME}"

cmd=(
    "${PYTHON_BIN}" train.py
    --config-name baseline_gqa
    datasets=gqa
    "datasets.train.data_dir=${GQA_DATA_DIR}"
    "datasets.train.answer_vocab_path=${GQA_DATA_DIR}/gqa_answer_vocab.json"
    "datasets.train.d_kg=${D_KG}"
    "datasets.val.data_dir=${GQA_DATA_DIR}"
    "datasets.val.answer_vocab_path=${GQA_DATA_DIR}/gqa_answer_vocab.json"
    "datasets.val.d_kg=${D_KG}"
    "model.d_kg=${D_KG}"
    "dataloader.batch_size=${BATCH_SIZE}"
    "dataloader.num_workers=${NUM_WORKERS}"
    "trainer.save_dir=${SAVE_DIR}"
    "trainer.n_epochs=${N_EPOCHS}"
    "trainer.epoch_len=${EPOCH_LEN}"
    trainer.override=True
    "model.freeze_text_encoder=${FREEZE_TEXT_ENCODER}"
    writer.mode=offline
    "writer.run_name=${RUN_NAME}"
)

if [[ -n "${TEXT_ENCODER_PATH}" ]]; then
    cmd+=(
        "model.text_encoder_name=${TEXT_ENCODER_PATH}"
        "datasets.train.text_encoder_name=${TEXT_ENCODER_PATH}"
        "datasets.val.text_encoder_name=${TEXT_ENCODER_PATH}"
    )
fi

printf 'Command: '
printf '%q ' "${cmd[@]}"
printf '\n\n'

exec "${cmd[@]}"
