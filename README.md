# VQA-GNN: практичные GQA + VQA-2 baseline’ы

Статус репозитория: `partially validated baseline`.

Текущая цель: практичные GQA и VQA-2 training paths, которые можно запускать
в Kaggle-лимитах и использовать как baselines перед архитектурными
изменениями. Следующий активный research step — additive graph-link module
поверх GQA baseline. GQA остаётся `paper-aligned approximation`; VQA-2 —
coursework extension и `not validated yet` относительно реальных training
numbers.

Проект не является `paper reproduced` и не является `fully paper-faithful`.
Реальные training/eval paper numbers отсутствуют.

## Режим проекта

- `GQA` = основной benchmark и основной training path
- `VQA-2` = вспомогательный validated extension path
- удалённый третий benchmark находится вне активного scope репозитория

## Активные пути

GQA, основной paper-aligned path:

- Model: `src/model/gqa_model.py::GQAVQAGNNModel`
- Dataset: `src/datasets/gqa_dataset.py::GQADataset`
- Collate: `src/datasets/gqa_collate.py::gqa_collate_fn`
- Loss: `src/loss/gqa_loss.py::GQALoss`
- Metric: `src/metrics/gqa_metric.py::GQAAccuracy`
- Train config: `src/configs/baseline_gqa.yaml`
- Research configs:
  - `src/configs/graph_link_gqa.yaml`
  - `src/configs/graph_link_gqa_frozen.yaml`
  - `src/configs/graph_link_gqa_frozen_warm.yaml`
- Inference config: `src/configs/inference_gqa.yaml`

VQA-2, coursework extension:

- Model: `src/model/vqa_gnn.py::VQAGNNModel`
- Dataset: `src/datasets/vqa_dataset.py::VQADataset`
- Collate: `src/datasets/vqa_collate.py::vqa_collate_fn`
- Loss: `src/loss/vqa_loss.py::VQALoss`
- Metric: `src/metrics/vqa_metric.py::VQAAccuracy`
- Train config: `src/configs/baseline_vqa.yaml`
- Research configs:
  - `src/configs/graph_link_vqa.yaml`
  - `src/configs/graph_link_vqa_frozen.yaml`
- Inference config: `src/configs/inference_vqa.yaml`

Удалён отдельный strict paper-equation / two-subgraph контур. Оставшийся путь
сохраняет ключевые paper-aligned признаки для GQA: 1842-way classification,
Visual Genome/GQA features, textual scene-graph nodes, typed relation ids
через `graph_edge_types`, RoBERTa-large question encoder.

## Kaggle-профиль

`baseline_gqa.yaml` и `baseline_vqa.yaml` настроены как практичные стартовые
профили:

- `baseline_gqa`: frozen `roberta-large` question encoder
- `baseline_gqa`: decoder LR `1e-4` with split-LR infra preserved for follow-up runs
- `baseline_gqa`: `model.num_gnn_layers: 5`
- `baseline_gqa`: linear warmup + cosine decay
- `model.d_hidden: 512`
- `dataloader.batch_size: 4`
- `dataloader.num_workers: 2`
- `baseline_gqa`: `trainer.n_epochs: 6`
- `baseline_gqa`: `trainer.epoch_len: 4000`
- `baseline_gqa`: `trainer.max_grad_norm: 5.0`
- `baseline_gqa`: reproducible subset controls via
  `datasets.{train,val}.{limit,shuffle_index,shuffle_seed}`
- `datasets.val.limit: 10000`
- `writer.mode: offline`

Это всё ещё engineering approximation под Kaggle, а не полная paper
reproduction. Но активный GQA baseline теперь ближе к статье по optimizer /
scheduler / depth / text finetuning, сохраняя при этом практичный runtime.

## Исследовательский Graph-Link Module

На GQA path теперь добавлен отдельный additive research module:

- Module: `src/model/graph_link.py::SparseGraphLinkModule`
- Integration point: `src/model/gqa_model.py::GQAVQAGNNModel`
- Design note: [GRAPH_LINK_MODULE_DESIGN.md](GRAPH_LINK_MODULE_DESIGN.md)

Он делает question-conditioned sparse top-k linking между:

- visual scene nodes
- textual/kg nodes

в отдельной late-fusion ветке.

Текущая интеграция:

- sparse link construction
- bidirectional cross-attention
- pooled `link_repr`
- отдельный graph-link answer head
- финальное logit fusion:
  `baseline_logits + alpha * graph_link_logits`

Режимы:

- `baseline_gqa` — baseline control
- `graph_link_gqa` — baseline + graph-link module
- `graph_link_gqa_frozen` — frozen baseline + trainable graph-link module
- `graph_link_gqa_frozen_warm` — frozen baseline, warm-started from trained
  baseline checkpoint, trainable graph-link module only
- `baseline_vqa` — VQA-2 coursework baseline control
- `graph_link_vqa` — VQA-2 baseline + graph-link module
- `graph_link_vqa_frozen` — frozen VQA-2 baseline + trainable graph-link module

Freeze policy применяется через `freeze_policy` block в train config и
обрабатывается в `src/utils/model_freeze.py`.

Для каузального ответа на вопрос "откуда взялись +10pp на GQA?" главным
следующим режимом считается именно `graph_link_gqa_frozen_warm`:

- загружается обученный baseline checkpoint;
- baseline path полностью замораживается;
- обучаются только:
  - link scorer,
  - sparse cross-attention,
  - pooled `link_repr`,
  - `graph_link_classifier`,
  - scalar `graph_link_alpha`.

Это отделяет вклад нового модуля от co-adaptation baseline backbone.

## Зафиксированные метрики

Текущие зафиксированные practical baseline observations:

- GQA controlled frozen subset baseline (`train=100k`, `val=7k`): `val_GQA_Accuracy = 0.20`
- GQA graph-link controlled subset (`train=100k`, `val=7k`): `val_GQA_Accuracy = 0.30+`
- VQA-2 classic baseline: `val_VQA_Accuracy = 0.54`
- VQA-2 graph-link late-fusion run: `val_VQA_Accuracy = 0.55`

Подробный контекст запусков см. в [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Состояние GQA-данных

Strict GQA package был локально собран, uploaded to Kaggle и затем validated
end-to-end against restored real Kaggle data.

Проверенный GQA data/runtime contract:

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

## Состояние VQA-2-данных

VQA-2 path подключён к активным `train.py` / `inference.py`. Реальный VQA-2
package в этом репозитории сейчас
`not validated yet`.

Ожидаемая структура VQA-2:

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

Runtime-contract:

- visual features: `float32[36, 2048]`
- KG node features: `float32[30, 300]`
- graph adjacency: `float32[67, 67]`
- graph node types: `long[67]`
- answer scores: `float32[3129]`
- output logits: `float32[B, 3129]`

Настройка Kaggle symlink:

```bash
bash scripts/setup_vqa_kaggle_links.sh \
  --dataset-dir /kaggle/input/datasets/daakifev/vqa-gnn-data \
  --visual-features /kaggle/input/datasets/daakifev/vqa-gnn-visual-features/all_features.h5 \
  --target-dir /kaggle/working/data/vqa
```

Это создаст expected runtime layout под `data/vqa` через symlinks, чтобы не
держать длинный блок `mkdir` / `ln -sf` в ноутбуке. Без аргументов скрипт
использует эти же Kaggle пути по умолчанию.

## Kaggle smoke-проверка

```bash
OFFLINE=1 \
TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
BATCH_SIZE=4 \
N_EPOCHS=1 \
EPOCH_LEN=50 \
bash scripts/run_kaggle_smoke_test.sh
```

## Практический train

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

## Локальный demo-режим

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

## Проверка

Инженерные тесты:

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

## Активные документы

- [PROJECT_SCOPE.md](PROJECT_SCOPE.md) — explicit repository scope.
- [ARCHITECTURE_SPLIT.md](ARCHITECTURE_SPLIT.md) — active architecture split.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — limitations, deviations, Kaggle risks.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — GQA and VQA-2 data contracts.
- [GQA_CLOSURE_REPORT.md](GQA_CLOSURE_REPORT.md) — итог GQA engineering-фазы.

## Дисциплина claim’ов

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
