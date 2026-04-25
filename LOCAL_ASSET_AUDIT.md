# Local Asset Audit

Audit date: 2026-04-21  
GQA strict update: 2026-04-25  
Repository: `/Users/srs15/Documents/CourseWorkR&D/paper_model_replication`  
Scope: initial local filesystem/repository audit, then 2026-04-25 strict GQA artifact acquisition/preprocessing update. No model training and no architecture fixes were performed.

Status vocabulary in this report:

- `ready locally`: file/code/artifact is physically present and usable for the checked local purpose.
- `missing locally`: expected file/artifact was not found on disk.
- `present but mismatched`: file exists, but does not satisfy the current strict contract.
- `blocked by external artifact`: cannot be closed without acquiring a dataset/checkpoint/precomputed output.
- `blocked by missing paper detail`: cannot be closed by files alone because the paper leaves a required detail unspecified.
- `unknown`: present signal is insufficient to verify safely.

Important distinction: `file exists`, `file satisfies current strict contract`, and `file closes exactness blocker` are treated as three separate facts.

## Executive summary

Ready locally:

- Core repo entrypoints and scripts are present: `train.py`, `inference.py`, `scripts/prepare_gqa_data.py`, `scripts/prepare_vcr_data.py`, `scripts/validate_gqa_data.py`, `scripts/validate_vcr_data.py`, `scripts/eval_vcr_qar.py`.
- Legacy baseline configs and paper-equation configs are present. Hydra dry-run composition succeeds for `baseline_gqa`, `baseline_vcr_qa`, `baseline_vcr_qar`, `paper_vqa_gnn_gqa`, and `paper_vqa_gnn_vcr`.
- Paper-path code is present: `PaperVCRModel`, `PaperGQAModel`, `PaperMultiRelationGATLayer`, `PaperWarmupCosine`, and `paper_param_groups`.
- Exactness docs are present: `PAPER_PARITY_MATRIX.md`, `BLOCKED_EXACTNESS_ITEMS.md`, `TRAIN_PROTOCOL_PARITY.md`.
- Strict GQA package is now canonical at `data/gqa` and is `runtime-valid`: answer vocab has 1842 entries plus `idx_to_answer`/metadata; relation vocab has 624 relation ids; graph HDF5 files have `d_kg=600` and `graph_edge_types`; metadata is present.
- GQA visual HDF5 files are present via live symlinks to `data/gqa_legacy_latest/visual_features/*` and satisfy the runtime visual feature shape contract `100 x 2048`.
- Raw GQA acquisition inputs are present locally: official scene graphs, object features, object info JSON, balanced questions, and GloVe 840B 300d.
- ConceptNet Numberbatch, ConceptNet assertions, local `data/roberta-large`, and a Hugging Face `roberta-large` cache snapshot are present.

Missing locally:

- All required VCR real-data files and VCR processed artifacts are absent.
- Tang unbiased scene-graph outputs/checkpoint, VinVL artifacts/p-node cache, QA-GNN concept embedding/retrieval artifacts, and paper model checkpoints are absent.

Mismatched locally:

- Legacy GQA artifacts are preserved at `data/gqa_legacy_20260425_105927` (`data/gqa_legacy_latest` symlink). They remain legacy/mismatched, but they are no longer the canonical `data/gqa` path.
- Paper-equation configs compose, but their paper batch keys require split subgraph tensors that current datasets/local artifacts do not emit.

What already allows progress:

- The local codebase is ready for targeted artifact acquisition and for auditing/validating newly acquired packages.
- GQA strict runtime path is now ready for local smoke/runtime runs and for downstream experiment setup that honestly claims `runtime-valid` data, not paper reproduction.

Major blockers:

- Any VCR real run is blocked by missing VCR data.
- VCR exactness is further blocked by Tang SG, VinVL p-node features, QA-GNN concept path, and missing paper details around retrieval/joint training.
- Paper-equation GQA/VCR end-to-end paths are still blocked by the split subgraph batch contract and paper/external-artifact gaps.

## 1. Code and config inventory

### Core entrypoints

| Path | Status | Notes |
|---|---|---|
| `train.py` | `ready locally` | Uses Hydra; supports `optimizer_param_groups`. |
| `inference.py` | `ready locally` | Existing inference entrypoint. |
| `scripts/prepare_gqa_data.py` | `ready locally` | Paper-aligned GQA preprocessing path exists. |
| `scripts/prepare_vcr_data.py` | `ready locally` | VCR preparation script exists. |
| `scripts/validate_gqa_data.py` | `ready locally` | Strict validator exists; canonical `data/gqa` now passes strict validation. |
| `scripts/validate_vcr_data.py` | `ready locally` | Validator exists; local VCR data is missing. |
| `scripts/eval_vcr_qar.py` | `ready locally` | Post-hoc Q->AR evaluator exists. |

### Config paths

| Config group | Status | Notes |
|---|---|---|
| `src/configs/baseline_gqa.yaml` | `ready locally` | Demo default; real data via `datasets=gqa`. |
| `src/configs/baseline_vcr_qa.yaml` | `ready locally` | Demo default; real VCR requires `data/vcr`. |
| `src/configs/baseline_vcr_qar.yaml` | `ready locally` | Demo default; separate QA->R path. |
| `src/configs/model/vqa_gnn_gqa.yaml` | `ready locally` | `GQAVQAGNNModel`, `d_kg=600`, `num_relations=512`. |
| `src/configs/model/vqa_gnn_vcr.yaml` | `ready locally` | `VCRVQAGNNModel`, `d_kg=300`, scalar candidate scoring. |
| `src/configs/paper_vqa_gnn_gqa.yaml` | `ready locally` | Hydra compose dry-run succeeds; runtime blocked by paper subgraph data contract. |
| `src/configs/paper_vqa_gnn_vcr.yaml` | `ready locally` | Hydra compose dry-run succeeds; runtime blocked by VCR artifacts and paper subgraph data contract. |
| `src/configs/model/paper_vqa_gnn_gqa.yaml` | `ready locally` | `PaperGQAModel`, `d_kg=600`, 1842 classes, L=5. |
| `src/configs/model/paper_vqa_gnn_vcr.yaml` | `ready locally` | `PaperVCRModel`, `d_kg=1024`, L=5, p-node required. |

### Paper-specific implementation

| Component | Status | Notes |
|---|---|---|
| `src/model/paper_vqa_gnn.py::PaperVCRModel` | `ready locally` | Architecture code present; requires `p_features`, split scene/concept subgraphs, typed edges. |
| `src/model/paper_vqa_gnn.py::PaperGQAModel` | `ready locally` | Architecture code present; requires split visual/textual SG tensors. |
| `src/model/paper_rgat.py::PaperMultiRelationGATLayer` | `ready locally` | Relation-aware paper-equation layer present. |
| `src/lr_schedulers/paper_warmup_cosine.py` | `ready locally` | Paper warmup/cosine scheduler present. |
| `src/optim/paper_param_groups.py` | `ready locally` | Separate LM/GNN LR grouping present. |
| Joint VCR training capability | `blocked by missing paper detail` | No local capability implementing paper joint Q->A + QA->R training with shared LM; paper does not specify mixing/loss details. |

### Exactness docs

| Path | Status |
|---|---|
| `PAPER_PARITY_MATRIX.md` | `ready locally` |
| `BLOCKED_EXACTNESS_ITEMS.md` | `ready locally` |
| `TRAIN_PROTOCOL_PARITY.md` | `ready locally` |

Note: several paper/exactness files are currently local untracked/modified files. This audit records local availability, not git commit status.

## 2. VCR local asset audit

### Required raw/processed files

| Path | Status |
|---|---|
| `data/vcr/train.jsonl` | `missing locally` |
| `data/vcr/val.jsonl` | `missing locally` |
| `data/vcr/test.jsonl` | `missing locally` |
| `data/vcr/visual_features/train_features.h5` | `missing locally` |
| `data/vcr/visual_features/val_features.h5` | `missing locally` |
| `data/vcr/knowledge_graphs/train_graphs.h5` | `missing locally` |
| `data/vcr/knowledge_graphs/val_graphs.h5` | `missing locally` |

### Optional exactness-critical VCR artifacts

| Artifact | Status | Notes |
|---|---|---|
| Raw VCR feature directories | `missing locally` | Search found no `data/vcr` tree. |
| Tang unbiased SG outputs/checkpoint | `missing locally` | Only documentation/code references were found. |
| VinVL checkpoint/features/p-node cache | `missing locally` | No local VinVL or `p_features` cache found. |
| QA-GNN concept embeddings/code clone | `missing locally` | ConceptNet files exist, but QA-GNN 1024-d path does not. |
| Serialized concept retrieval outputs | `missing locally` | No local retrieval cache found. |

### VCR contract result

- Expected visual features: `36 x 2048` per image under current runtime contract.
- Expected graph group datasets: `node_features`, `adj_matrix`, `node_types`.
- Observed: no VCR HDF5 files exist, so contract status is `missing locally`.

`data/saved/audit_vcr_inference_smoke` exists, but it contains small inference output `.pth` files with `pred_label`/`label`; these are not model checkpoints and do not close any VCR data blocker.

## 3. GQA local asset audit

Update 2026-04-25: strict GQA artifacts were acquired/rebuilt and swapped into the canonical `data/gqa` path. The previous legacy package is preserved at `data/gqa_legacy_20260425_105927`; `data/gqa_legacy_latest` points to that backup.

### Processed files

| Path | Status | Observed detail |
|---|---|---|
| `data/gqa/gqa_answer_vocab.json` | `ready locally` | `answer_to_idx` length 1842; `idx_to_answer` and metadata present. |
| `data/gqa/gqa_relation_vocab.json` | `ready locally` | 624 relation ids; 310 predicates; special relations present. |
| `data/gqa/questions/train_balanced_questions.json` | `ready locally` | Dict with 943000 questions, 775.3MB. |
| `data/gqa/questions/val_balanced_questions.json` | `ready locally` | Dict with 132062 questions, 108.6MB. |
| `data/gqa/metadata/*` | `ready locally` | Train/val visual and KG metadata present. |

### Visual features

| Path | Status | First sample | Root keys | Attrs |
|---|---|---:|---:|---|
| `data/gqa/visual_features/train_features.h5` | `ready locally` | `[100, 2048]`, `float32` | 72140 | `num_boxes=100`, `feature_dim=2048` |
| `data/gqa/visual_features/val_features.h5` | `ready locally` | `[100, 2048]`, `float32` | 10234 | `num_boxes=100`, `feature_dim=2048` |

Contract interpretation: these files satisfy the checked runtime shape requirement `100 x 2048`. The canonical files are symlinks to the preserved legacy visual HDF5 files, with strict visual metadata now stored under `data/gqa/metadata`.

### Knowledge graphs

| Path | Status | First key | Observed schema | Strict expectation |
|---|---|---|---|---|
| `data/gqa/knowledge_graphs/train_graphs.h5` | `ready locally` / `runtime-valid` | full split | `node_features` use `d_kg=600`; `adj_matrix`, `node_types`, and `graph_edge_types` present; `graph_edge_type_count=624` | Passes strict validator and one-batch runtime check. |
| `data/gqa/knowledge_graphs/val_graphs.h5` | `ready locally` / `runtime-valid` | full split | `node_features` use `d_kg=600`; `adj_matrix`, `node_types`, and `graph_edge_types` present; `graph_edge_type_count=624` | Passes strict validator and one-batch runtime check. |

Root key counts match the question counts:

- train graphs: 943000 groups.
- val graphs: 132062 groups.

This is no longer partial availability. Canonical `data/gqa` satisfies the current strict GQA contract and passes runtime validation.

### Raw GQA preprocessing inputs

| Path/artifact | Status |
|---|---|
| `data/raw/gqa/questions/train_balanced_questions.json` | `ready locally` |
| `data/raw/gqa/questions/val_balanced_questions.json` | `ready locally` |
| `data/raw/gqa/train_sceneGraphs.json` | `ready locally` |
| `data/raw/gqa/val_sceneGraphs.json` | `ready locally` |
| `data/raw/gqa/objects/` | `ready locally` |
| `data/raw/gqa/gqa_objects_info.json` | `ready locally` |
| `data/raw/glove/glove.840B.300d.txt` | `ready locally` |
| `data/downloads/sceneGraphs.zip`, `objectFeatures.zip`, `glove.840B.300d.zip` | `ready locally` |

## 4. External artifact audit

| Artifact | Status | Observed detail | Exactness note |
|---|---|---|---|
| `data/conceptnet/numberbatch-en-19.08.txt.gz` | `ready locally` | 310M | Useful locally, but not QA-GNN 1024-d concept entity embeddings. |
| `data/conceptnet/conceptnet-assertions-5.7.0.csv.gz` | `ready locally` | 475M | Useful for preprocessing; not enough to reproduce paper retrieval exactly. |
| GQA official scene graphs | `ready locally` | `data/raw/gqa/train_sceneGraphs.json`, `data/raw/gqa/val_sceneGraphs.json` | Used to build strict GQA graphs. |
| GQA official object features | `ready locally` | `data/raw/gqa/objects/gqa_objects_*.h5`, `data/raw/gqa/gqa_objects_info.json` | Used for strict GQA graph alignment and visual metadata. |
| GloVe 840B 300d | `ready locally` | `data/raw/glove/glove.840B.300d.txt` | Used for 600-d textual SG node features. |
| `data/roberta-large` | `ready locally` | `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json` | Local checkpoint exists. |
| HF cache `models--roberta-large/snapshots/722cf37...` | `ready locally` | Includes `model.safetensors`, `config.json`, `tokenizer.json`, `vocab.json`, `merges.txt` | Configs still say `text_encoder_name: roberta-large`; offline use depends on HF cache resolution unless overridden. |
| Tang unbiased SG generator | `missing locally` | No local artifact found | `blocked by external artifact`. |
| VinVL | `missing locally` | No local checkpoint/features/cache found | `blocked by external artifact`. |
| QA-GNN concept path | `missing locally` | No code clone/embedding cache found | `blocked by external artifact` and `blocked by missing paper detail`. |
| Paper model checkpoints | `missing locally` | No `model_best.pth`, `.ckpt`, or training checkpoint found outside smoke inference outputs | Not ready for model-weight reproduction claims. |

Installed dependency spot check in `.venv`:

- Python 3.14.3, pytest 9.0.3.
- Present: torch 2.11.0, torchvision 0.26.0, transformers 5.5.3, hydra 1.3.2, omegaconf 2.3.0, h5py 3.16.0, numpy 1.26.4, pandas 2.3.3, tqdm 4.67.3, torchmetrics 1.9.0, wandb 0.25.1, comet_ml 3.57.3.
- Missing but not required by `requirements.txt`: spaCy, NLTK, sklearn.
- Minor environment note: importing torchmetrics produced a Matplotlib/font cache warning because the default user cache directories are not writable; it fell back to a temp cache.

## 5. Contract validation results

### Present and valid

| Contract | Result |
|---|---|
| GQA answer vocab | `ready locally`: 1842 answers; `idx_to_answer` and metadata present. |
| GQA relation vocab | `ready locally`: 624 relation ids; special relations present. |
| GQA questions | `ready locally`: train 943000, val 132062. |
| GQA visual features | `ready locally`: first sample shape `100 x 2048`, `float32`. |
| GQA graph artifacts | `ready locally` / `runtime-valid`: full train/val graphs use `d_kg=600` and include `graph_edge_types`. |
| GQA strict runtime path | `runtime-valid`: `validate_gqa_data.py` passed with 60 OK, 0 warnings, 0 errors; one-batch forward produced logits `(1, 1842)`. |
| Hydra config composition | `ready locally`: paper and baseline configs compose with `--cfg job`. |
| RoBERTa local availability | `ready locally`: local dir and HF cache snapshot exist. |

### Present but mismatched

| Contract | Result |
|---|---|
| Legacy GQA backup | `present but mismatched`: preserved at `data/gqa_legacy_20260425_105927`, not canonical. |
| Paper-equation data contract | `present but mismatched`: current artifacts/datasets do not emit split visual/textual or scene/concept subgraph batch keys. |

### Missing

| Contract | Result |
|---|---|
| VCR real raw/processed files | `missing locally`. |
| Tang/VinVL/QA-GNN exactness artifacts | `missing locally`. |
| Model checkpoints | `missing locally` except RoBERTa checkpoint/cache. |

### Unknown

| Item | Why |
|---|---|
| Exact paper training/evaluation numbers | No long training/evaluation was run; no paper-number claim is made. |

## 6. Exactness blocker map

| Blocker | Current local status | Can be closed by file download alone? | Next required work |
|---|---|---|---|
| VCR raw data | `missing locally` | Partly: raw data can be acquired under VCR ToU. | Requires preprocessing unless already acquired as processed private package. |
| VCR baseline processed HDF5 | `missing locally` | Yes if a package already matches repo runtime contract. | Otherwise run `scripts/prepare_vcr_data.py` after raw acquisition. |
| VCR Tang SG outputs | `blocked by external artifact` | No. Checkpoint acquisition alone is insufficient. | Run generator on VCR images and serialize typed scene-graph outputs. |
| VCR VinVL p-node features | `blocked by external artifact` | No. Checkpoint acquisition alone is insufficient. | Run/retrieve VG region features and cache `p_features` per candidate/context. |
| VCR QA-GNN concept embeddings/retrieval | `blocked by external artifact`; also `blocked by missing paper detail` | No. | Need QA-GNN embedding path plus deterministic retrieval/top-k/entity linking spec. |
| VCR joint Q->A + QA->R training | `blocked by missing paper detail` | No. | Paper does not specify task mixing/loss weighting. |
| GQA strict graphs | `ready locally` / `runtime-valid` | Already closed locally. | Keep canonical `data/gqa`; legacy backup remains at `data/gqa_legacy_20260425_105927`. |
| GQA relation vocab | `ready locally` | Already closed locally. | 624 relation ids; keep `model.num_relations` >= 624 for strict runs. |
| GQA metadata | `ready locally` | Already closed locally. | Train/val visual and KG metadata are under `data/gqa/metadata`. |
| RoBERTa-large | `ready locally` | Already present. | Optional config override to `data/roberta-large` for explicit offline path. |
| Paper checkpoints | `missing locally` | Only if original/relevant checkpoints are acquired. | Does not solve data/preprocessing exactness by itself. |

## 7. Recommended next steps

Step 1

Run a short canonical GQA smoke command using `data/gqa` without dataset overrides only if needed for the next phase. Treat the resulting claim as `runtime-valid`, not as a paper number.

Step 2

Acquire VCR real data under the VCR license. Minimum baseline target package: `train.jsonl`, `val.jsonl`, visual HDF5, and KG HDF5 under `data/vcr`. If raw-only data is acquired, run the VCR preparation path; if a processed private package is acquired, validate with `scripts/validate_vcr_data.py`.

Step 3

For strict paper-path VCR, separately acquire and integrate Tang SG outputs, VinVL p-node feature generation/cache, and QA-GNN-style concept retrieval/embedding artifacts. Treat this as a preprocessing/integration phase, not a simple file-placement phase, because several required transformations and paper details are not fully specified.

Step 4

After VCR baseline assets are present, decide whether to prioritize (a) VCR baseline `runtime-valid` data, (b) GQA training smoke on the strict package, or (c) paper-equation split-subgraph integration. Keep these as separate phases because they close different blockers.

## Final verdict

`local environment ready for targeted VCR artifact acquisition`

The repository/code side is locally strong enough to guide acquisition and validation. GQA strict artifacts are now `ready locally` and `runtime-valid`; VCR remains blocked by missing critical assets and exactness-specific external artifacts.
