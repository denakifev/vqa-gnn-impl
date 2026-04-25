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

The strict package was validated by `scripts/validate_gqa_data.py
--skip-runtime-check` with `0` errors before being uploaded to Kaggle.
Local payloads were intentionally removed after upload; restoring the
private Kaggle dataset is required for runtime validation and training.

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

## Equation-Level Paper Path (NOT Runtime)

| Layer | Class / file | Status |
|---|---|---|
| Model | `src/model/paper_vqa_gnn.py::PaperGQAModel` | equation reference |
| RGAT layer | `src/model/paper_rgat.py::PaperMultiRelationGATLayer` | paper Eqs 1–6 |
| Hydra config | `src/configs/paper_vqa_gnn_gqa.yaml` | composes only; not runnable |

`PaperGQAModel` requires a two-subgraph batch (visual SG + textual SG with
a shared q-node index). The current `GQADataset` emits a single merged
graph, so this path is `blocked by external artifact / preprocessing`
work. Tests cover its equation-level shape via mocked tensors only.

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
not `paper-faithful`:

- merged-graph `GQAVQAGNNModel` instead of two-subgraph `PaperGQAModel`;
- dense GAT + per-(head, relation) attention bias instead of the paper's
  full RGAT message passing;
- offline pre-built scene-graph HDF5 instead of on-the-fly construction;
- entity linking via tokenization rather than NER / entity-linking;
- RoBERTa-large as closest public default text encoder.
