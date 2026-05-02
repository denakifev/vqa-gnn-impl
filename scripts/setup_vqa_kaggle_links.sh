#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Create a VQA-2 runtime layout under a target directory using symlinks.

Defaults are tailored for the author's Kaggle datasets:
  --dataset-dir      /kaggle/input/datasets/daakifev/vqa-gnn-data
  --visual-features  /kaggle/input/datasets/daakifev/vqa-gnn-visual-features/all_features.h5
  --target-dir       /kaggle/working/data/vqa

Usage:
  bash scripts/setup_vqa_kaggle_links.sh

  bash scripts/setup_vqa_kaggle_links.sh \
    --dataset-dir /kaggle/input/datasets/daakifev/vqa-gnn-data \
    --visual-features /kaggle/input/datasets/daakifev/vqa-gnn-visual-features/all_features.h5 \
    --target-dir /kaggle/working/data/vqa
EOF
}

DATASET_DIR="/kaggle/input/datasets/daakifev/vqa-gnn-data"
VISUAL_FEATURES_H5="/kaggle/input/datasets/daakifev/vqa-gnn-visual-features/all_features.h5"
TARGET_DIR="/kaggle/working/data/vqa"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --visual-features)
            VISUAL_FEATURES_H5="$2"
            shift 2
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

for required_path in \
    "${DATASET_DIR}/answer_vocab.json" \
    "${DATASET_DIR}/annotations/train_annotations.json" \
    "${DATASET_DIR}/annotations/val_annotations.json" \
    "${DATASET_DIR}/questions/train_questions.json" \
    "${DATASET_DIR}/questions/val_questions.json" \
    "${DATASET_DIR}/knowledge_graphs/train_graphs.h5" \
    "${DATASET_DIR}/knowledge_graphs/val_graphs.h5" \
    "${VISUAL_FEATURES_H5}"
do
    if [[ ! -e "${required_path}" ]]; then
        echo "[ERROR] Required path not found: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${TARGET_DIR}/annotations"
mkdir -p "${TARGET_DIR}/questions"
mkdir -p "${TARGET_DIR}/visual_features"
mkdir -p "${TARGET_DIR}/knowledge_graphs"

ln -sf "${DATASET_DIR}/answer_vocab.json" \
    "${TARGET_DIR}/answer_vocab.json"

ln -sf "${DATASET_DIR}/annotations/train_annotations.json" \
    "${TARGET_DIR}/annotations/train_annotations.json"
ln -sf "${DATASET_DIR}/annotations/val_annotations.json" \
    "${TARGET_DIR}/annotations/val_annotations.json"

ln -sf "${DATASET_DIR}/questions/train_questions.json" \
    "${TARGET_DIR}/questions/train_questions.json"
ln -sf "${DATASET_DIR}/questions/val_questions.json" \
    "${TARGET_DIR}/questions/val_questions.json"

ln -sf "${DATASET_DIR}/knowledge_graphs/train_graphs.h5" \
    "${TARGET_DIR}/knowledge_graphs/train_graphs.h5"
ln -sf "${DATASET_DIR}/knowledge_graphs/val_graphs.h5" \
    "${TARGET_DIR}/knowledge_graphs/val_graphs.h5"

ln -sf "${VISUAL_FEATURES_H5}" \
    "${TARGET_DIR}/visual_features/train_features.h5"
ln -sf "${VISUAL_FEATURES_H5}" \
    "${TARGET_DIR}/visual_features/val_features.h5"

echo "VQA-2 runtime links created under: ${TARGET_DIR}"
echo "Dataset root: ${DATASET_DIR}"
echo "Visual features HDF5: ${VISUAL_FEATURES_H5}"
