# Data Contracts

Authoritative GQA data contract for the strict paper-aligned package. Runtime
errors and config comments link back here.

VCR contracts are intentionally out of scope of this document while VCR data
preparation remains open.

## GQA Strict Package Layout

```
data/gqa/
├── gqa_answer_vocab.json
├── gqa_relation_vocab.json
├── questions/
│   ├── train_balanced_questions.json
│   └── val_balanced_questions.json
├── visual_features/
│   ├── train_features.h5
│   └── val_features.h5
├── knowledge_graphs/
│   ├── train_graphs.h5
│   └── val_graphs.h5
└── metadata/
    ├── train_visual_features_metadata.json
    ├── val_visual_features_metadata.json
    ├── train_knowledge_graphs_metadata.json
    └── val_knowledge_graphs_metadata.json
```

## Validated Contract Values

| Field | Value | Source |
|---|---|---|
| `answer_to_idx` size | `1842` | `gqa_answer_vocab.json` |
| `relation_count_total` | `624` (= 4 specials + 2 × 310 predicates) | `gqa_relation_vocab.json` |
| visual feature shape | `float32[100, 2048]` | `*_features.h5` per imageId |
| graph `node_features` last dim | `600` (= 2 × 300d GloVe: name + attributes) | `*_graphs.h5` per question_id |
| graph attrs `d_kg` | `600` | `*_graphs.h5` file attrs |
| graph attrs `num_visual_nodes` | `100` | `*_graphs.h5` file attrs |
| graph attrs `max_kg_nodes` | `100` | `*_graphs.h5` file attrs |
| graph attrs `graph_edge_type_count` | `624` | `*_graphs.h5` file attrs |
| graph attrs `graph_mode` | `"official_scene_graph"` | `*_graphs.h5` file attrs |
| graph attrs `conceptnet_used` | `False` | `*_graphs.h5` file attrs |
| graph attrs `fully_connected_fallback_used` | `False` | `*_graphs.h5` file attrs |

The strict package was uploaded to Kaggle and then validated end-to-end
against restored real Kaggle data by `scripts/validate_gqa_data.py` with
runtime check enabled. Local payloads may be removed after upload to reclaim
disk space; restoring the private Kaggle dataset is required for local
runtime validation and training.

End-to-end evidence:

- train: `943000` questions, `72140` images, all imageIds covered;
- val: `132062` questions, `10234` images, all imageIds covered;
- sampled graph schemas include `node_features`, `adj_matrix`, `node_types`,
  `graph_edge_types`;
- sampled graphs have no fully-connected visual fallback;
- metadata no-match rate is approximately `9.5%` train / `9.6%` val;
- runtime path validates
  `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy`;
- runtime sanity: logits `(1, 1842)`, finite loss `7.326373`, finite metric.

## Runtime Batch Contract (`GQADataset` → `gqa_collate_fn`)

| Key | Type / Shape | Notes |
|---|---|---|
| `visual_features` | `float32[B, 100, 2048]` | object features |
| `question_input_ids` | `long[B, max_question_len]` | RoBERTa-large tokens |
| `question_attention_mask` | `long[B, max_question_len]` | |
| `graph_node_features` | `float32[B, 100, 600]` | textual scene-graph nodes |
| `graph_adj` | `float32[B, 201, 201]` | symmetric, with self-loops |
| `graph_edge_types` | `long[B, 201, 201]` | ids in `[0, 624)` |
| `graph_node_types` | `long[B, 201]` | `0=visual`, `1=question`, `2=kg` |
| `labels` | `long[B]` | answer indices in `[0, 1842)` |
| `question_id` | `list[str]` | metadata only |

`N_total = num_visual_nodes + 1 + max_kg_nodes = 201`.

## Source-of-Truth Model Path (Runtime)

| Layer | Class / file | Status |
|---|---|---|
| Model | `src/model/gqa_model.py::GQAVQAGNNModel` | runtime source-of-truth |
| Loss | `src/loss/gqa_loss.py::GQALoss` | hard cross-entropy |
| Metric | `src/metrics/gqa_metric.py::GQAAccuracy` | exact-match |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset` | strict mode |
| Collate | `src/datasets/gqa_collate.py::gqa_collate_fn` | |
| Hydra config | `src/configs/baseline_gqa.yaml` | |

`GQAVQAGNNModel` consumes `graph_edge_types` through every
`DenseGATLayer` (relation-aware attention bias) when `num_relations > 0`.
Setting `num_relations = 0` reverts to untyped GAT (ablation only,
`not paper-faithful yet`).

## Equation-Level Paper Path (Runtime)

| Layer | Class / file | Status |
|---|---|---|
| Model | `src/model/paper_vqa_gnn.py::PaperGQAModel` | runnable |
| RGAT layer | `src/model/paper_rgat.py::PaperMultiRelationGATLayer` | paper Eqs 1–6 |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset(emit_two_subgraphs=True)` | slices merged HDF5 |
| Collate | `src/datasets/gqa_collate.py::gqa_paper_collate_fn` | |
| Hydra config (train) | `src/configs/paper_vqa_gnn_gqa.yaml` | composes and runs |
| Hydra config (inference) | `src/configs/inference_paper_gqa.yaml` | composes and runs |

`PaperGQAModel` requires the two-subgraph batch (visual SG + q, textual
SG + q). `GQADataset(emit_two_subgraphs=True)` slices the same strict
merged-graph HDF5 into the two-subgraph layout at runtime; **HDF5 does
not need to be rebuilt**. The slice is statically lossless because
`gqa_preprocessing.py` never writes visual↔textual cross-edges in the
merged adjacency (only visual↔visual, textual↔textual,
q↔visual, q↔textual). Tests in
`tests/test_paper_gqa_two_subgraphs.py` enforce slice correctness,
collate stacking, and end-to-end forward through `PaperGQAModel`.

## Two-Subgraph Batch Contract (`PaperGQAModel`)

Convention: **q at the LAST index** of each subgraph. The merged-graph
layout produced by `gqa_preprocessing.py` is
`[visual(0..Nv-1), q(Nv), textual(Nv+1..Nv+Nt)]`; the slicer takes:

- visual subgraph = merged rows/cols `[0..Nv]` (q at index `Nv`);
- textual subgraph = permutation `[Nv+1..Nv+Nt, Nv]` (q at index `Nt`).

`Nv = num_visual_nodes = 100`, `Nt = max_kg_nodes = 100` for the strict
GQA package, so visual subgraph size = 101, textual subgraph size = 101.

| Key | Type / Shape | Notes |
|---|---|---|
| `question_input_ids` | `long[B, max_question_len]` | RoBERTa-large tokens |
| `question_attention_mask` | `long[B, max_question_len]` | |
| `visual_node_features` | `float32[B, 101, 2048]` | row at q-slot is zeros (overwritten) |
| `visual_adj` | `float32[B, 101, 101]` | symmetric, with self-loops |
| `visual_edge_types` | `long[B, 101, 101]` | ids in `[0, 624)` |
| `visual_node_types` | `long[B, 101]` | `0=visual`, `1=question` at q-slot |
| `visual_node_is_q_index` | `long[B]` | constant `100` |
| `textual_node_features` | `float32[B, 101, 600]` | row at q-slot is zeros |
| `textual_adj` | `float32[B, 101, 101]` | symmetric, with self-loops |
| `textual_edge_types` | `long[B, 101, 101]` | ids in `[0, 624)` |
| `textual_node_types` | `long[B, 101]` | `2=kg`, `1=question` at q-slot |
| `textual_node_is_q_index` | `long[B]` | constant `100` |
| `labels` | `long[B]` | answer indices in `[0, 1842)` |
| `question_id` | `list[str]` | metadata only |

The single shared relation vocab (`relation_count_total = 624`) is reused
in both subgraphs. Paper §5.1 GQA does not specify whether visual SG and
textual SG should share or split relation vocabularies; using the shared
vocab is documented in `src/configs/model/paper_vqa_gnn_gqa.yaml`.

## Strict Toggles in `GQADataset`

`GQADataset.expect_graph_edge_types: bool = True`
- `True` (default, paper-aligned): graph HDF5 must carry `graph_edge_types`.
- `False`: explicitly accept legacy untyped artifacts; edge types are
  zero-filled and the run is logged as `not paper-faithful yet`.

`GQADataset.strict_graph_schema: bool = True`
- `True` (default): probe the first graph entry on first HDF5 open and
  fail fast on `d_kg` mismatch / missing required datasets / missing
  `graph_edge_types` (when expected).
- `False`: skip the probe (debug only). The underlying tensor copy still
  raises later; this toggle does not silently hide mismatches.

Both toggles are exposed via `src/configs/datasets/gqa.yaml` and
`src/configs/datasets/gqa_eval.yaml`.

## Runtime Validation Commands

After restoring the strict GQA Kaggle package into `data/gqa`:

```bash
.venv/bin/python scripts/validate_gqa_data.py \
  --data-dir data/gqa \
  --answer-vocab data/gqa/gqa_answer_vocab.json \
  --relation-vocab data/gqa/gqa_relation_vocab.json \
  --split train val \
  --num-visual-nodes 100 \
  --feature-dim 2048 \
  --d-kg 600 \
  --max-kg-nodes 100 \
  --num-answers 1842 \
  --require-metadata
```

The runtime check now exercises `GQAVQAGNNModel` with the relation vocab
size taken from `gqa_relation_vocab.json`, so it covers the same
relation-aware GNN path the trainer uses.

## Known Approximations vs Paper

These are deliberate, documented approximations — `paper-aligned approximation`,
not `paper-faithful`. They cannot be eliminated without paper detail or
external artifacts:

- `GQAVQAGNNModel` (runtime fast-path) keeps merged-graph + DenseGAT
  relation bias; `PaperGQAModel` is provided as the equation-faithful
  alternative on the same data.
- offline pre-built scene-graph HDF5 instead of on-the-fly construction;
- entity linking via tokenization rather than NER / entity-linking;
- RoBERTa-large as closest public default text encoder;
- single shared relation vocabulary across visual SG and textual SG
  (paper does not specify the split);
- hidden widths and activations inside `f_r`, `f_h` MLPs (paper Eqs 1, 3
  state shape only);
- single attention head (paper Eq 5 is written with a scalar `gamma_ij`,
  taken as the most literal reading).
