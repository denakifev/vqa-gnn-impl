#!/usr/bin/env bash
set -euo pipefail

# Short real-data smoke test for VQA-GNN on Kaggle.
#
# Expected runtime layout:
#   /kaggle/working/vqa_runtime/
#     answer_vocab.json
#     questions/{train,val}_questions.json
#     annotations/{train,val}_annotations.json
#     knowledge_graphs/{train,val}_graphs.h5
#     visual_features/{train,val}_features.h5
#
# Typical usage:
#   bash scripts/run_kaggle_smoke_test.sh
#
# With local/offline text encoder:
#   OFFLINE=1 \
#   TEXT_ENCODER_PATH=/kaggle/input/hf-roberta-large \
#   bash scripts/run_kaggle_smoke_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/kaggle/working/vqa_runtime}"
SAVE_DIR="${SAVE_DIR:-/kaggle/working/saved}"
RUN_NAME="${RUN_NAME:-smoke_real_vqa}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
N_EPOCHS="${N_EPOCHS:-1}"
EPOCH_LEN="${EPOCH_LEN:-50}"
VALIDATE_SAMPLE_SIZE="${VALIDATE_SAMPLE_SIZE:-50}"
FREEZE_TEXT_ENCODER="${FREEZE_TEXT_ENCODER:-true}"
D_KG="${D_KG:-300}"
NUM_ANSWERS="${NUM_ANSWERS:-3129}"
MAX_KG_NODES="${MAX_KG_NODES:-30}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-}"

export COMET_API_KEY="${COMET_API_KEY:-offline_placeholder}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

if [[ "${OFFLINE:-0}" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
fi

required_paths=(
  "${DATA_ROOT}/answer_vocab.json"
  "${DATA_ROOT}/questions/train_questions.json"
  "${DATA_ROOT}/questions/val_questions.json"
  "${DATA_ROOT}/annotations/train_annotations.json"
  "${DATA_ROOT}/annotations/val_annotations.json"
  "${DATA_ROOT}/knowledge_graphs/train_graphs.h5"
  "${DATA_ROOT}/knowledge_graphs/val_graphs.h5"
  "${DATA_ROOT}/visual_features/train_features.h5"
  "${DATA_ROOT}/visual_features/val_features.h5"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${SAVE_DIR}"

echo "[1/2] Validating runtime data root: ${DATA_ROOT}"
"${PYTHON_BIN}" scripts/validate_data.py \
  --data-dir "${DATA_ROOT}" \
  --split train val \
  --answer-vocab "${DATA_ROOT}/answer_vocab.json" \
  --num-visual-nodes 36 \
  --feature-dim 2048 \
  --d-kg "${D_KG}" \
  --num-answers "${NUM_ANSWERS}" \
  --max-kg-nodes "${MAX_KG_NODES}" \
  --sample-size "${VALIDATE_SAMPLE_SIZE}"

echo "[2/2] Launching smoke-train run: ${RUN_NAME}"

cmd=(
  "${PYTHON_BIN}" train.py
  --config-name baseline_vqa
  datasets=vqa
  "datasets.train.data_dir=${DATA_ROOT}"
  "datasets.train.answer_vocab_path=${DATA_ROOT}/answer_vocab.json"
  "datasets.val.data_dir=${DATA_ROOT}"
  "datasets.val.answer_vocab_path=${DATA_ROOT}/answer_vocab.json"
  "model.d_kg=${D_KG}"
  "datasets.train.d_kg=${D_KG}"
  "datasets.val.d_kg=${D_KG}"
  "dataloader.batch_size=${BATCH_SIZE}"
  "dataloader.num_workers=${NUM_WORKERS}"
  "trainer.save_dir=${SAVE_DIR}"
  "trainer.n_epochs=${N_EPOCHS}"
  "trainer.epoch_len=${EPOCH_LEN}"
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
printf '\n'

exec "${cmd[@]}"
