# GQA Closure Report

Дата: 2026-05-02.

Статус: GQA engineering path is `implemented`, `validated`, and ready for
practical Kaggle training experiments. Numeric training results are still
`not validated yet`.

## Runnable Path

| Layer | Class / config |
|---|---|
| Model | `src/model/gqa_model.py::GQAVQAGNNModel` |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset` |
| Collate | `src/datasets/gqa_collate.py::gqa_collate_fn` |
| Hydra train | `src/configs/baseline_gqa.yaml` |
| Hydra inference | `src/configs/inference_gqa.yaml` |

Status: `paper-aligned approximation`. This is the single supported Kaggle
baseline path.

## Validated Data Contract

- `answer_to_idx` size = 1842, contiguous, coverage 100%.
- `relation_count_total` = 624, contiguous.
- train: 943,000 questions, 72,140 images, full image coverage.
- val: 132,062 questions, 10,234 images, full image coverage.
- visual feature shape: `float32[100, 2048]`.
- graph `node_features.shape[-1] = 600`.
- graph attrs include `d_kg=600`, `num_visual_nodes=100`,
  `max_kg_nodes=100`, `graph_edge_type_count=624`,
  `graph_mode=official_scene_graph`, `conceptnet_used=False`,
  `fully_connected_fallback_used=False`.
- metadata package exists; no-match rate about `9.5%` train / `9.6%` val.

## Runtime Validation

- `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy`: `validated`.
- `GQAVQAGNNModel` consumes `graph_edge_types` through DenseGAT relation bias:
  `validated`.
- `baseline_gqa.yaml` targets `GQAVQAGNNModel`: `validated`.

Runtime sanity against restored real Kaggle data:

- logits shape: `(1, 1842)`;
- loss finite: `7.326373`;
- random-init loss close to `ln(1842) = 7.519`.

## Remaining Items

- Real GQA train/eval numbers: `not validated yet`.
- Kaggle budget fit: `not validated yet`.
- Exact paper training hyperparameters: `paper-aligned approximation`.
- Current Kaggle defaults intentionally freeze the text encoder, use
  `d_hidden=512`, and limit validation to 10k samples.
- Possible future engineering: AMP, gradient accumulation, and timed Kaggle
  profiling.

## Out Of Scope

VQA-2 has been restored as a separate coursework extension via
`baseline_vqa.yaml`, but it is not part of this GQA closure claim and remains
`not validated yet` on real data.
