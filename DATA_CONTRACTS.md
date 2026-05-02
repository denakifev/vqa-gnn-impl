# Data Contracts

Authoritative data contracts for the active GQA and VQA-2 paths. Runtime
errors and config comments link back here.

Repository scope:

- GQA main
- VQA-2 auxiliary
- no removed benchmark in active scope

## GQA Strict Package Layout

```text
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

## GQA Validated Values

| Field | Value |
|---|---|
| `answer_to_idx` size | `1842` |
| `relation_count_total` | `624` |
| visual feature shape | `float32[100, 2048]` |
| graph `node_features` last dim | `600` |
| graph attrs `d_kg` | `600` |
| graph attrs `num_visual_nodes` | `100` |
| graph attrs `max_kg_nodes` | `100` |
| graph attrs `graph_edge_type_count` | `624` |
| graph attrs `graph_mode` | `"official_scene_graph"` |
| graph attrs `conceptnet_used` | `False` |
| graph attrs `fully_connected_fallback_used` | `False` |

End-to-end evidence:

- train: `943000` questions, `72140` images, all imageIds covered;
- val: `132062` questions, `10234` images, all imageIds covered;
- metadata no-match rate approximately `9.5%` train / `9.6%` val;
- runtime path validates
  `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy`.

## GQA Runtime Batch Contract

| Key | Type / Shape |
|---|---|
| `visual_features` | `float32[B, 100, 2048]` |
| `question_input_ids` | `long[B, max_question_len]` |
| `question_attention_mask` | `long[B, max_question_len]` |
| `graph_node_features` | `float32[B, 100, 600]` |
| `graph_adj` | `float32[B, 201, 201]` |
| `graph_edge_types` | `long[B, 201, 201]` |
| `graph_node_types` | `long[B, 201]` |
| `labels` | `long[B]` |
| `question_id` | `list[str]` |

`N_total = num_visual_nodes + 1 + max_kg_nodes = 201`.

## GQA Source-Of-Truth Runtime Path

| Layer | Class / file |
|---|---|
| Model | `src/model/gqa_model.py::GQAVQAGNNModel` |
| Loss | `src/loss/gqa_loss.py::GQALoss` |
| Metric | `src/metrics/gqa_metric.py::GQAAccuracy` |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset` |
| Collate | `src/datasets/gqa_collate.py::gqa_collate_fn` |
| Hydra config | `src/configs/baseline_gqa.yaml` |

`GQAVQAGNNModel` consumes `graph_edge_types` through every `DenseGATLayer`
when `num_relations > 0`. Setting `num_relations = 0` reverts to untyped GAT
and is an ablation only.

## VQA-2 Package Layout

VQA-2 is a coursework extension in this repo. Its real-data package is
`not validated yet`.

```text
data/vqa/
├── answer_vocab.json
├── questions/
│   ├── train_questions.json
│   └── val_questions.json
├── annotations/
│   ├── train_annotations.json
│   └── val_annotations.json
├── visual_features/
│   ├── train_features.h5
│   └── val_features.h5
└── knowledge_graphs/
    ├── train_graphs.h5
    └── val_graphs.h5
```

## VQA-2 Runtime Batch Contract

| Key | Type / Shape |
|---|---|
| `visual_features` | `float32[B, 36, 2048]` |
| `question_input_ids` | `long[B, max_question_len]` |
| `question_attention_mask` | `long[B, max_question_len]` |
| `graph_node_features` | `float32[B, 30, 300]` |
| `graph_adj` | `float32[B, 67, 67]` |
| `graph_node_types` | `long[B, 67]` |
| `answer_scores` | `float32[B, 3129]` |
| `labels` | `long[B]` |
| `question_id` | `list[int]` |

`N_total = num_visual_nodes + 1 + max_kg_nodes = 67`.

## VQA-2 Runtime Path

| Layer | Class / file |
|---|---|
| Model | `src/model/vqa_gnn.py::VQAGNNModel` |
| Loss | `src/loss/vqa_loss.py::VQALoss` |
| Metric | `src/metrics/vqa_metric.py::VQAAccuracy` |
| Dataset | `src/datasets/vqa_dataset.py::VQADataset` |
| Collate | `src/datasets/vqa_collate.py::vqa_collate_fn` |
| Hydra config | `src/configs/baseline_vqa.yaml` |

VQA-2 uses soft VQA labels (`answer_scores`) and binary
cross-entropy-with-logits by default.

Validate a restored package with:

```bash
.venv/bin/python scripts/validate_vqa_data.py \
  --data-dir data/vqa \
  --answer-vocab data/vqa/answer_vocab.json \
  --split train val \
  --text-encoder data/roberta-large
```

## Known Approximations

- merged-graph DenseGAT fast path with relation-aware attention bias;
- Kaggle default graph hidden width is `512`, not a claim of exact paper width;
- text encoder is frozen by default for the practical Kaggle baseline;
- offline pre-built scene-graph HDF5 instead of on-the-fly construction;
- tokenization-based entity linking;
- RoBERTa-large as closest public default text encoder;
- single shared relation vocabulary across visual and textual scene graphs;
- underspecified MLP widths / activations inside relation and node update
  functions.
