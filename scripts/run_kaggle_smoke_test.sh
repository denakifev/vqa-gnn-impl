#!/usr/bin/env bash
# run_kaggle_smoke_test.sh — Short real-data smoke test for VQA-GNN on Kaggle.
#
# Supports two modes: GQA and VCR.
# Select via TASK env var (default: gqa).
#
# GQA mode:
#   Expects Kaggle Dataset mounted at /kaggle/input/gqa-gnn-data (or GQA_DATA_DIR).
#   Runs: python train.py --config-name baseline_gqa datasets=gqa ...
#
# VCR mode (Q->A):
#   Expects Kaggle Dataset mounted at /kaggle/input/vcr-gnn-data (or VCR_DATA_DIR).
#   Runs: python train.py --config-name baseline_vcr_qa datasets=vcr_qa ...
#
# USAGE:
#   # GQA smoke test (default):
#   TASK=gqa bash scripts/run_kaggle_smoke_test.sh
#
#   # VCR Q->A smoke test:
#   TASK=vcr bash scripts/run_kaggle_smoke_test.sh
#
#   # With offline text encoder (no HuggingFace download):
#   TASK=gqa \
#   OFFLINE=1 \
#   TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
#   bash scripts/run_kaggle_smoke_test.sh
#
# EXPECTED DATA LAYOUT:
#
#   GQA (/kaggle/input/gqa-gnn-data or GQA_DATA_DIR):
#     gqa_answer_vocab.json
#     questions/train_balanced_questions.json
#     questions/val_balanced_questions.json
#     knowledge_graphs/train_graphs.h5
#     knowledge_graphs/val_graphs.h5
#     visual_features/train_features.h5
#     visual_features/val_features.h5
#
#   VCR (/kaggle/input/vcr-gnn-data or VCR_DATA_DIR):
#     train.jsonl
#     val.jsonl
#     knowledge_graphs/train_graphs.h5
#     knowledge_graphs/val_graphs.h5
#     visual_features/train_features.h5
#     visual_features/val_features.h5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Common settings
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_DIR="${SAVE_DIR:-/kaggle/working/saved}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
N_EPOCHS="${N_EPOCHS:-1}"
EPOCH_LEN="${EPOCH_LEN:-50}"
VALIDATE_SAMPLE_SIZE="${VALIDATE_SAMPLE_SIZE:-50}"
FREEZE_TEXT_ENCODER="${FREEZE_TEXT_ENCODER:-true}"
D_KG="${D_KG:-}"
MAX_KG_NODES="${MAX_KG_NODES:-}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-}"
TASK="${TASK:-gqa}"   # "gqa" or "vcr"

if [[ -z "${D_KG}" ]]; then
    if [[ "${TASK}" == "gqa" ]]; then
        D_KG=600
    else
        D_KG=300
    fi
fi

if [[ -z "${MAX_KG_NODES}" ]]; then
    if [[ "${TASK}" == "gqa" ]]; then
        MAX_KG_NODES=100
    else
        MAX_KG_NODES=30
    fi
fi

export COMET_API_KEY="${COMET_API_KEY:-offline_placeholder}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

if [[ "${OFFLINE:-0}" == "1" ]]; then
    export TRANSFORMERS_OFFLINE=1
fi

echo "========================================================"
echo "VQA-GNN Kaggle smoke test"
echo "  Task:       ${TASK}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epoch len:  ${EPOCH_LEN}"
echo "  Epochs:     ${N_EPOCHS}"
echo "  Save dir:   ${SAVE_DIR}"
echo "========================================================"

mkdir -p "${SAVE_DIR}"

# ---------------------------------------------------------------------------
# GQA smoke test
# ---------------------------------------------------------------------------
if [[ "${TASK}" == "gqa" ]]; then
    GQA_DATA_DIR="${GQA_DATA_DIR:-/kaggle/input/gqa-gnn-data}"
    NUM_ANSWERS="${NUM_ANSWERS:-1842}"
    RUN_NAME="${RUN_NAME:-smoke_gqa}"

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
            echo "        Prepare with: bash scripts/stage_gqa_for_kaggle.sh" >&2
            exit 1
        fi
        echo "  [OK] ${path}"
    done

    echo ""
    echo "[2/3] Validating GQA data structure..."
    validate_cmd=(
        "${PYTHON_BIN}" scripts/validate_gqa_data.py
        --data-dir "${GQA_DATA_DIR}" \
        --split train val \
        --answer-vocab "${GQA_DATA_DIR}/gqa_answer_vocab.json" \
        --relation-vocab "${GQA_DATA_DIR}/gqa_relation_vocab.json" \
        --num-visual-nodes 100 \
        --feature-dim 2048 \
        --d-kg "${D_KG}" \
        --num-answers "${NUM_ANSWERS}" \
        --max-kg-nodes "${MAX_KG_NODES}" \
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

# ---------------------------------------------------------------------------
# VCR Q->A smoke test
# ---------------------------------------------------------------------------
elif [[ "${TASK}" == "vcr" ]]; then
    VCR_DATA_DIR="${VCR_DATA_DIR:-/kaggle/input/vcr-gnn-data}"
    VCR_TASK_MODE="${VCR_TASK_MODE:-qa}"   # "qa" or "qar"
    RUN_NAME="${RUN_NAME:-smoke_vcr_${VCR_TASK_MODE}}"

    if [[ "${VCR_TASK_MODE}" == "qa" ]]; then
        CONFIG_NAME="baseline_vcr_qa"
        DATASETS_KEY="vcr_qa"
    elif [[ "${VCR_TASK_MODE}" == "qar" ]]; then
        CONFIG_NAME="baseline_vcr_qar"
        DATASETS_KEY="vcr_qar"
    else
        echo "[ERROR] VCR_TASK_MODE must be 'qa' or 'qar', got: ${VCR_TASK_MODE}" >&2
        exit 1
    fi

    echo ""
    echo "[1/3] Checking required VCR files..."
    required_paths=(
        "${VCR_DATA_DIR}/train.jsonl"
        "${VCR_DATA_DIR}/val.jsonl"
        "${VCR_DATA_DIR}/knowledge_graphs/train_graphs.h5"
        "${VCR_DATA_DIR}/knowledge_graphs/val_graphs.h5"
        "${VCR_DATA_DIR}/visual_features/train_features.h5"
        "${VCR_DATA_DIR}/visual_features/val_features.h5"
    )

    for path in "${required_paths[@]}"; do
        if [[ ! -e "${path}" ]]; then
            echo "[ERROR] Required path not found: ${path}" >&2
            echo "        Prepare with: bash scripts/stage_vcr_for_kaggle.sh" >&2
            exit 1
        fi
        echo "  [OK] ${path}"
    done

    echo ""
    echo "[2/3] Validating VCR data structure..."
    "${PYTHON_BIN}" scripts/validate_vcr_data.py \
        --data-dir "${VCR_DATA_DIR}" \
        --split train val \
        --num-visual-nodes 36 \
        --feature-dim 2048 \
        --d-kg "${D_KG}" \
        --max-kg-nodes "${MAX_KG_NODES}" \
        --sample-size "${VALIDATE_SAMPLE_SIZE}"

    echo ""
    echo "[3/3] Launching VCR ${VCR_TASK_MODE} smoke-train: ${RUN_NAME}"

    cmd=(
        "${PYTHON_BIN}" train.py
        --config-name "${CONFIG_NAME}"
        "datasets=${DATASETS_KEY}"
        "datasets.train.data_dir=${VCR_DATA_DIR}"
        "datasets.train.d_kg=${D_KG}"
        "datasets.val.data_dir=${VCR_DATA_DIR}"
        "datasets.val.d_kg=${D_KG}"
        "model.d_kg=${D_KG}"
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
    printf '\n\n'

    exec "${cmd[@]}"

else
    echo "[ERROR] TASK must be 'gqa' or 'vcr', got: ${TASK}" >&2
    exit 1
fi
