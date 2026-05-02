# VQA-GNN: Practical GQA + VQA-2 Baselines

Статус репозитория: `partially validated baseline`.

Текущая цель: практичные GQA и VQA-2 training paths, которые можно запускать
в Kaggle-лимитах и использовать как baselines перед архитектурными
изменениями. GQA остаётся `paper-aligned approximation`; VQA-2 — coursework
extension и `not validated yet` относительно реальных training numbers.

Проект не является `paper reproduced` и не является `fully paper-faithful`.
Реальные training/eval paper numbers отсутствуют.

## Active Paths

GQA, основной paper-aligned path:

- Model: `src/model/gqa_model.py::GQAVQAGNNModel`
- Dataset: `src/datasets/gqa_dataset.py::GQADataset`
- Collate: `src/datasets/gqa_collate.py::gqa_collate_fn`
- Loss: `src/loss/gqa_loss.py::GQALoss`
- Metric: `src/metrics/gqa_metric.py::GQAAccuracy`
- Train config: `src/configs/baseline_gqa.yaml`
- Inference config: `src/configs/inference_gqa.yaml`

VQA-2, coursework extension:

- Model: `src/model/vqa_gnn.py::VQAGNNModel`
- Dataset: `src/datasets/vqa_dataset.py::VQADataset`
- Collate: `src/datasets/vqa_collate.py::vqa_collate_fn`
- Loss: `src/loss/vqa_loss.py::VQALoss`
- Metric: `src/metrics/vqa_metric.py::VQAAccuracy`
- Train config: `src/configs/baseline_vqa.yaml`
- Inference config: `src/configs/inference_vqa.yaml`

Удалён отдельный strict paper-equation / two-subgraph контур. Оставшийся путь
сохраняет ключевые paper-aligned признаки для GQA: 1842-way classification,
Visual Genome/GQA features, textual scene-graph nodes, typed relation ids
через `graph_edge_types`, RoBERTa-large question encoder.

## Kaggle Profile

`baseline_gqa.yaml` и `baseline_vqa.yaml` настроены как практичные стартовые
профили:

- `model.freeze_text_encoder: true`
- `model.d_hidden: 512`
- `dataloader.batch_size: 4`
- `dataloader.num_workers: 2`
- `trainer.n_epochs: 3`
- `trainer.epoch_len: 1000`
- `datasets.val.limit: 10000`
- `writer.mode: offline`

Это намеренная engineering approximation под Kaggle. Для более дорогого run
можно увеличивать `d_hidden`, `batch_size`, `epoch_len`, `n_epochs`, снимать
`datasets.val.limit` и размораживать text encoder.

## GQA Data State

Strict GQA package был локально собран, uploaded to Kaggle и затем validated
end-to-end against restored real Kaggle data.

Validated GQA data/runtime contract:

- `answer_to_idx` содержит `1842` entries, contiguous, coverage 100%.
- relation vocab содержит `624` relation ids: 310 predicates + 4 specials.
- train split: 943,000 questions, 72,140 images, all imageIds covered.
- val split: 132,062 questions, 10,234 images, all imageIds covered.
- train/val visual HDF5 samples: `float32[100, 2048]`.
- graph HDF5 содержит `node_features`, `adj_matrix`, `node_types`,
  `graph_edge_types`.
- graph `node_features.shape[-1] == 600`.
- graph attrs include `d_kg=600`, `num_visual_nodes=100`,
  `max_kg_nodes=100`, `graph_edge_type_count=624`,
  `graph_mode=official_scene_graph`, `conceptnet_used=False`,
  `fully_connected_fallback_used=False`.
- runtime path validated:
  `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy`.

Локальный `data/gqa` может отсутствовать. Для local или Kaggle запуска нужно
восстановить private Kaggle dataset в expected layout или указать Hydra
`datasets.*.data_dir` на mounted path.

## VQA-2 Data State

VQA-2 path восстановлен из legacy-кода и подключён к активным `train.py` /
`inference.py`. Реальный VQA-2 package в этом репозитории сейчас
`not validated yet`.

Expected VQA-2 layout:

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

Runtime contract:

- visual features: `float32[36, 2048]`
- KG node features: `float32[30, 300]`
- graph adjacency: `float32[67, 67]`
- graph node types: `long[67]`
- answer scores: `float32[3129]`
- output logits: `float32[B, 3129]`

## Kaggle Smoke

```bash
OFFLINE=1 \
TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
BATCH_SIZE=4 \
N_EPOCHS=1 \
EPOCH_LEN=50 \
bash scripts/run_kaggle_smoke_test.sh
```

## Practical Train

GQA:

```bash
python train.py --config-name baseline_gqa datasets=gqa \
  datasets.train.data_dir=/kaggle/input/gqa-gnn-data \
  datasets.train.answer_vocab_path=/kaggle/input/gqa-gnn-data/gqa_answer_vocab.json \
  datasets.val.data_dir=/kaggle/input/gqa-gnn-data \
  datasets.val.answer_vocab_path=/kaggle/input/gqa-gnn-data/gqa_answer_vocab.json \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  trainer.save_dir=/kaggle/working/saved \
  trainer.progress_bar=auto \
  trainer.override=True
```

VQA-2:

```bash
python train.py --config-name baseline_vqa datasets=vqa \
  datasets.train.data_dir=/kaggle/input/vqa-gnn-data \
  datasets.train.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
  datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
  datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  trainer.save_dir=/kaggle/working/saved \
  trainer.progress_bar=auto \
  trainer.override=True
```

Эта команда использует Kaggle-friendly defaults из config. Для короткой
проверки можно добавить:

```bash
trainer.n_epochs=1 trainer.epoch_len=50 datasets.val.limit=1000
```

## Local Demo

```bash
.venv/bin/python train.py --config-name baseline_gqa \
  trainer.n_epochs=1 \
  trainer.epoch_len=2 \
  trainer.override=True
```

```bash
.venv/bin/python train.py --config-name baseline_vqa \
  trainer.n_epochs=1 \
  trainer.epoch_len=2 \
  trainer.override=True
```

Demo dataset synthetic и не является результатом статьи.

## Validation

Engineering tests:

```bash
.venv/bin/python -m pytest -q
```

Strict GQA data validation after restoring the private Kaggle package:

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

VQA-2 data validation:

```bash
.venv/bin/python scripts/validate_vqa_data.py \
  --data-dir data/vqa \
  --answer-vocab data/vqa/answer_vocab.json \
  --split train val \
  --num-visual-nodes 36 \
  --feature-dim 2048 \
  --d-kg 300 \
  --max-kg-nodes 30 \
  --num-answers 3129 \
  --text-encoder data/roberta-large
```

For file/schema-only validation, add `--skip-runtime-check`.

## Active Documents

- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — limitations, deviations, Kaggle risks.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — strict GQA data contract.
- [GQA_CLOSURE_REPORT.md](GQA_CLOSURE_REPORT.md) — итог GQA engineering-фазы.

## Claim Discipline

Разрешённые status tokens:

- `implemented`
- `validated`
- `paper-aligned`
- `paper-aligned approximation`
- `not paper-faithful yet`
- `runtime-valid`
- `not validated yet`

Запрещённые claims:

- `paper reproduced`
- `fully paper-faithful`
- `ready for full reproduction`
- `GQA real baseline validated`
