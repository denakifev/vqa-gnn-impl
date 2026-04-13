#!/usr/bin/env bash
# stage_vcr_for_kaggle.sh — Prepare VCR data for Kaggle Dataset upload.
#
# Creates a Kaggle-ready staging directory from locally prepared VCR artifacts.
# Supports two variants:
#   mini  — first N annotations per split (for sanity / smoke runs on Kaggle)
#   full  — full VCR train/val split
#
# ============================================================
# LICENSE AND DISTRIBUTION NOTICE
# ============================================================
# VCR (Visual Commonsense Reasoning) data is subject to the VCR Terms of Use:
#   https://visualcommonsense.com/
# You MUST agree to the VCR license before downloading or using the data.
# The full VCR dataset MUST NOT be uploaded as a public Kaggle Dataset.
# Private Kaggle Datasets are acceptable for personal research use.
#
# ConceptNet Numberbatch (used for KG graphs) is CC BY-SA 4.0.
# ============================================================
#
# PREREQUISITES (all must be prepared locally first):
#   data/vcr/train.jsonl
#   data/vcr/val.jsonl
#   data/vcr/visual_features/train_features.h5
#   data/vcr/visual_features/val_features.h5
#   data/vcr/knowledge_graphs/train_graphs.h5    (built by prepare_vcr_data.py)
#   data/vcr/knowledge_graphs/val_graphs.h5
#
# USAGE:
#   # Stage full dataset (keep private on Kaggle):
#   bash scripts/stage_vcr_for_kaggle.sh
#
#   # Stage mini subset (first 200 annotations per split):
#   VARIANT=mini MINI_N=200 bash scripts/stage_vcr_for_kaggle.sh
#
#   # Custom paths:
#   VCR_DIR=data/vcr STAGING_DIR=kaggle_staging/vcr-gnn-data-mini \
#     VARIANT=mini MINI_N=200 bash scripts/stage_vcr_for_kaggle.sh
#
# OUTPUT STRUCTURE (matches VCRDataset expected layout):
#   ${STAGING_DIR}/
#   ├── dataset-metadata.json           ← Kaggle Dataset metadata (PRIVATE)
#   ├── train.jsonl
#   ├── val.jsonl
#   ├── visual_features/
#   │   ├── train_features.h5
#   │   └── val_features.h5
#   └── knowledge_graphs/
#       ├── train_graphs.h5
#       └── val_graphs.h5
#
# After staging (PRIVATE ONLY — do not make VCR data public):
#   kaggle datasets create -p ${STAGING_DIR}   # creates private dataset by default
#   kaggle datasets version -p ${STAGING_DIR} -m "description"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration (can be overridden via environment)
# ---------------------------------------------------------------------------
VCR_DIR="${VCR_DIR:-${REPO_ROOT}/data/vcr}"
VARIANT="${VARIANT:-full}"          # "mini" or "full"
MINI_N="${MINI_N:-200}"             # annotations per split for mini variant
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${VARIANT}" == "mini" ]]; then
    STAGING_DIR="${STAGING_DIR:-${REPO_ROOT}/kaggle_staging/vcr-gnn-data-mini}"
    DATASET_ID="${DATASET_ID:-vcr-gnn-data-mini}"
    DATASET_TITLE="${DATASET_TITLE:-VCR VQA-GNN data (mini ${MINI_N} per split) — PRIVATE}"
else
    STAGING_DIR="${STAGING_DIR:-${REPO_ROOT}/kaggle_staging/vcr-gnn-data}"
    DATASET_ID="${DATASET_ID:-vcr-gnn-data}"
    DATASET_TITLE="${DATASET_TITLE:-VCR VQA-GNN data (full) — PRIVATE}"
fi

echo "========================================================"
echo "VCR Kaggle staging"
echo "  Variant:     ${VARIANT}"
if [[ "${VARIANT}" == "mini" ]]; then
    echo "  Mini N:      ${MINI_N} per split"
fi
echo "  Source:      ${VCR_DIR}"
echo "  Destination: ${STAGING_DIR}"
echo ""
echo "  LICENSE: VCR data must remain PRIVATE on Kaggle."
echo "           Do NOT set visibility to public."
echo "========================================================"

# ---------------------------------------------------------------------------
# Verify source files
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Checking source files..."

required_files=(
    "${VCR_DIR}/train.jsonl"
    "${VCR_DIR}/val.jsonl"
    "${VCR_DIR}/visual_features/train_features.h5"
    "${VCR_DIR}/visual_features/val_features.h5"
    "${VCR_DIR}/knowledge_graphs/train_graphs.h5"
    "${VCR_DIR}/knowledge_graphs/val_graphs.h5"
)

all_ok=1
for f in "${required_files[@]}"; do
    if [[ ! -f "${f}" ]]; then
        echo "  [MISSING] ${f}"
        all_ok=0
    else
        echo "  [OK]      ${f}"
    fi
done

if [[ "${all_ok}" == "0" ]]; then
    echo ""
    echo "[ERROR] Some source files are missing. Prepare them first:" >&2
    echo "  1. Download VCR from https://visualcommonsense.com (license required)" >&2
    echo "  2. python scripts/prepare_vcr_data.py --jsonl data/vcr/train.jsonl ..." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Create staging directory structure
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Creating staging directory structure..."

mkdir -p "${STAGING_DIR}/visual_features"
mkdir -p "${STAGING_DIR}/knowledge_graphs"

echo "  Created: ${STAGING_DIR}/"

# ---------------------------------------------------------------------------
# Copy or subset JSONL files
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Processing JSONL annotation files..."

if [[ "${VARIANT}" == "mini" ]]; then
    echo "  Building mini JSONL subsets (first ${MINI_N} lines per split)..."
    for split in train val; do
        src="${VCR_DIR}/${split}.jsonl"
        dst="${STAGING_DIR}/${split}.jsonl"
        head -n "${MINI_N}" "${src}" > "${dst}"
        actual=$(wc -l < "${dst}")
        total=$(wc -l < "${src}")
        echo "  [OK] ${split}.jsonl: ${actual}/${total} annotations -> ${dst}"
    done
else
    cp "${VCR_DIR}/train.jsonl" "${STAGING_DIR}/train.jsonl"
    cp "${VCR_DIR}/val.jsonl" "${STAGING_DIR}/val.jsonl"
    echo "  [OK] Copied full JSONL files."
fi

# ---------------------------------------------------------------------------
# Handle HDF5 files
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Processing HDF5 files..."

if [[ "${VARIANT}" == "mini" ]]; then
    echo "  Subsetting HDF5 files to match mini JSONL..."

    "${PYTHON_BIN}" - <<PYEOF
import json, sys
from pathlib import Path
try:
    import h5py
except ImportError as e:
    print(f"[ERROR] {e}. Install h5py: pip install h5py", file=sys.stderr)
    sys.exit(1)

for split in ("train", "val"):
    jsonl_path = Path("${STAGING_DIR}") / f"{split}.jsonl"
    annot_ids = set()
    img_fns = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            annot_ids.add(item["annot_id"])
            img_fns.add(item["img_fn"])

    # Visual features (keyed by img_fn)
    src_vis = Path("${VCR_DIR}") / "visual_features" / f"{split}_features.h5"
    dst_vis = Path("${STAGING_DIR}") / "visual_features" / f"{split}_features.h5"
    n_vis = 0
    with h5py.File(src_vis, "r") as src, h5py.File(dst_vis, "w") as dst:
        for img_fn in img_fns:
            if img_fn in src:
                dst.create_dataset(img_fn, data=src[img_fn][()], compression="gzip")
                n_vis += 1
    print(f"  [OK] {split} visual: {n_vis}/{len(img_fns)} images -> {dst_vis}")

    # KG graphs (keyed by annot_id)
    src_kg = Path("${VCR_DIR}") / "knowledge_graphs" / f"{split}_graphs.h5"
    dst_kg = Path("${STAGING_DIR}") / "knowledge_graphs" / f"{split}_graphs.h5"
    n_kg = 0
    with h5py.File(src_kg, "r") as src, h5py.File(dst_kg, "w") as dst:
        for annot_id in annot_ids:
            if annot_id in src:
                grp = dst.create_group(annot_id)
                for k in ("node_features", "adj_matrix", "node_types"):
                    grp.create_dataset(k, data=src[annot_id][k][()], compression="gzip")
                n_kg += 1
    print(f"  [OK] {split} KG: {n_kg}/{len(annot_ids)} graphs -> {dst_kg}")
PYEOF

else
    echo "  Copying full HDF5 files (may take a while)..."
    cp "${VCR_DIR}/visual_features/train_features.h5" \
       "${STAGING_DIR}/visual_features/train_features.h5"
    echo "  [OK] Copied train visual features."
    cp "${VCR_DIR}/visual_features/val_features.h5" \
       "${STAGING_DIR}/visual_features/val_features.h5"
    echo "  [OK] Copied val visual features."
    cp "${VCR_DIR}/knowledge_graphs/train_graphs.h5" \
       "${STAGING_DIR}/knowledge_graphs/train_graphs.h5"
    echo "  [OK] Copied train KG graphs."
    cp "${VCR_DIR}/knowledge_graphs/val_graphs.h5" \
       "${STAGING_DIR}/knowledge_graphs/val_graphs.h5"
    echo "  [OK] Copied val KG graphs."
fi

# ---------------------------------------------------------------------------
# Write dataset-metadata.json
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Writing Kaggle dataset metadata..."

cat > "${STAGING_DIR}/dataset-metadata.json" <<JSON
{
  "title": "${DATASET_TITLE}",
  "id": "KAGGLE_USERNAME/${DATASET_ID}",
  "licenses": [
    {
      "name": "other"
    }
  ]
}
JSON

# Write a prominent license note
cat > "${STAGING_DIR}/LICENSE_NOTE.txt" <<TXT
VCR Data License Notice
=======================

This dataset contains preprocessed VCR (Visual Commonsense Reasoning) data.

VCR is subject to the VCR Terms of Use:
  https://visualcommonsense.com/

THIS DATASET MUST REMAIN PRIVATE.
Do not make this Kaggle Dataset publicly visible.

KG subgraph data is derived from ConceptNet Numberbatch (CC BY-SA 4.0):
  https://github.com/commonsense/conceptnet-numberbatch
TXT

echo "  [OK] dataset-metadata.json written."
echo "  [OK] LICENSE_NOTE.txt written."
echo "  ACTION REQUIRED: replace KAGGLE_USERNAME with your Kaggle username."

# ---------------------------------------------------------------------------
# Validate staged data
# ---------------------------------------------------------------------------
echo ""
echo "Validating staged data..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_vcr_data.py" \
    --data-dir "${STAGING_DIR}" \
    --split train val \
    --d-kg 300 \
    --sample-size 20

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "Staging complete: ${STAGING_DIR}"
echo ""
echo "IMPORTANT: Keep this Kaggle Dataset PRIVATE (VCR license)."
echo ""
echo "Kaggle upload commands (private):"
echo "  # First time:"
echo "  kaggle datasets create -p ${STAGING_DIR}"
echo ""
echo "  # Update existing:"
echo "  kaggle datasets version -p ${STAGING_DIR} -m 'VCR VQA-GNN ${VARIANT} data'"
echo ""
echo "Kaggle training command after attach:"
KAGGLE_DATA_PATH="/kaggle/input/${DATASET_ID}"
echo "  # Q->A:"
echo "  python train.py --config-name baseline_vcr_qa \\"
echo "    datasets=vcr_qa \\"
echo "    datasets.train.data_dir=${KAGGLE_DATA_PATH} \\"
echo "    datasets.val.data_dir=${KAGGLE_DATA_PATH} \\"
echo "    trainer.save_dir=/kaggle/working/saved"
echo ""
echo "  # QA->R:"
echo "  python train.py --config-name baseline_vcr_qar \\"
echo "    datasets=vcr_qar \\"
echo "    datasets.train.data_dir=${KAGGLE_DATA_PATH} \\"
echo "    datasets.val.data_dir=${KAGGLE_DATA_PATH} \\"
echo "    trainer.save_dir=/kaggle/working/saved"
echo "========================================================"
