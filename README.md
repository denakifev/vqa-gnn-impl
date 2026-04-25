# VQA-GNN: воспроизводимый baseline

Статус репозитория на 2026-04-25: `partially validated baseline`.

Проект содержит аудированную PyTorch/Hydra-реализацию baseline для статьи:

> VQA-GNN: Reasoning with Multimodal Knowledge via Graph Neural Networks for
> Visual Question Answering, ICCV 2023 / arXiv:2205.11501.

Репозиторий не является `paper reproduced` и не является
`fully paper-faithful`. Архитектура, конфиги, контракты данных, валидаторы и
smoke/runtime paths реализованы, но численные результаты статьи не
воспроизведены.

## Актуальные Документы

Оставлены только активные handoff-документы:

- [README.md](README.md) — текущий статус, команды, состояние данных.
- [ARCHITECTURE_SPLIT.md](ARCHITECTURE_SPLIT.md) — архитектурный split VCR/GQA.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — оставшиеся блокеры и отклонения от статьи.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — strict GQA data contract, batch keys,
  source-of-truth model path.
- [GQA_CLOSURE_REPORT.md](GQA_CLOSURE_REPORT.md) — итог GQA-фазы: что закрыто,
  что blocked, какой path считать source-of-truth.

`AGENTS.md` сохранён как рабочие инструкции для будущих coding agents, а не как
status report.

## Текущее Состояние Данных

GQA:

- Полный paper-target препроцессинг GQA был завершён, strict-validated
  локально, uploaded to Kaggle, затем validated end-to-end against real Kaggle
  data after restore.
- Валидированный GQA package был загружен на Kaggle командой:

```bash
kaggle datasets create -p kaggle_staging/gqa-gnn-data --dir-mode zip
```

- После upload локальные full GQA processed/raw/staging payloads могут быть
  удалены, чтобы освободить место.
- Поэтому локальный `data/gqa` может не существовать между фазами.
- Для локальной работы с GQA нужно восстановить private Kaggle dataset в
  ожидаемый layout `data/gqa` или указать Hydra `datasets.*.data_dir` на
  восстановленный путь.

Validated GQA data/runtime contract:

- `answer_to_idx` содержит `1842` entries, contiguous, coverage 100%.
- relation vocab содержит `624` relation ids: 310 predicates + 4 specials,
  contiguous.
- train split: 943,000 questions, 72,140 images, all imageIds covered.
- val split: 132,062 questions, 10,234 images, all imageIds covered.
- train/val visual HDF5 samples имели форму `float32[100, 2048]`.
- train/val graph HDF5 содержали `node_features`, `adj_matrix`, `node_types`,
  `graph_edge_types`.
- graph `node_features.shape[-1] == 600`.
- graph attrs включали `d_kg=600`, `num_visual_nodes=100`,
  `max_kg_nodes=100`, `graph_edge_type_count=624`,
  `graph_mode=official_scene_graph`, `conceptnet_used=False`,
  `fully_connected_fallback_used=False`.
- sampled graphs schema-consistent, без fully-connected visual fallback.
- metadata files present; no-match rate ~9.5% train / ~9.6% val.
- runtime path validated:
  `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy`.
- runtime sanity: logits `(1, 1842)`, finite loss `7.326373`, finite metric;
  random-init loss close to `ln(1842)=7.519`.

VCR:

- Локальные VCR data пока не подготовлены.
- VCR — лицензированные данные. Любой Kaggle upload должен оставаться private.
- Baseline-level VCR assets, ожидаемые репозиторием:

```text
data/vcr/
├── train.jsonl
├── val.jsonl
├── visual_features/
│   ├── train_features.h5
│   └── val_features.h5
└── knowledge_graphs/
    ├── train_graphs.h5
    └── val_graphs.h5
```

## Что Реализовано

| Область | Статус |
|---|---|
| VCR Q->A candidate scoring | `implemented`, `validated` |
| VCR QA->R candidate scoring | `implemented`, `validated` |
| VCR Q->AR post-hoc evaluator | `implemented`, `validated` |
| GQA 1842-way classification (fast path, `GQAVQAGNNModel`) | `implemented`, `validated`, `paper-aligned approximation` |
| GQA paper-equation path (`PaperGQAModel`, two-subgraph batch) | `implemented`, `validated` end-to-end |
| GQA relation-aware message passing по `graph_edge_types` | `implemented`, `validated`, `paper-aligned approximation` |
| Общие GNN/text primitives | `implemented`, `validated` |
| Demo datasets и smoke paths | `runtime-valid` |
| Full strict GQA data package | `validated` end-to-end against real Kaggle data |
| Full VCR data package | `not validated yet` |

Основные entry points:

- `train.py` (configs: `baseline_gqa`, `paper_vqa_gnn_gqa`, `baseline_vcr_qa`, `baseline_vcr_qar`)
- `inference.py` (configs: `inference_gqa`, `inference_paper_gqa`, `inference_vcr`)
- `scripts/validate_gqa_data.py`
- `scripts/validate_vcr_data.py`
- `scripts/stage_vcr_for_kaggle.sh`

Для GQA paper-equation training с реальными данными:

```bash
python train.py --config-name paper_vqa_gnn_gqa datasets=gqa_paper \
  datasets.train.data_dir=data/gqa \
  datasets.train.answer_vocab_path=data/gqa/gqa_answer_vocab.json \
  datasets.val.data_dir=data/gqa \
  datasets.val.answer_vocab_path=data/gqa/gqa_answer_vocab.json
```

## Команды Проверки

Engineering tests:

```bash
.venv/bin/python -m pytest -q
```

Data-contract tests:

```bash
.venv/bin/python -m pytest tests/test_engineering_contracts.py -q
```

Strict GQA data validation после восстановления private Kaggle package:

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

VCR validation после подготовки лицензированных VCR assets:

```bash
.venv/bin/python scripts/validate_vcr_data.py \
  --data-dir data/vcr \
  --split train val \
  --num-visual-nodes 36 \
  --feature-dim 2048 \
  --d-kg 300 \
  --max-kg-nodes 30
```

## Kaggle

Private GQA dataset уже создан из `kaggle_staging/gqa-gnn-data`. Локальный
staging был удалён после upload.

VCR должен оставаться private, потому что VCR data лицензированы. Staging script
пишет license note и ожидаемый Kaggle layout:

```bash
VARIANT=full bash scripts/stage_vcr_for_kaggle.sh
kaggle datasets create -p kaggle_staging/vcr-gnn-data --dir-mode zip
```

Перед созданием Kaggle dataset нужно отредактировать `dataset-metadata.json`,
чтобы `id` содержал ваш Kaggle username. После upload нужно проверить, что
visibility остаётся private.

## Запрещённые Claims

Нельзя утверждать:

- `paper reproduced`
- `fully paper-faithful`
- `ready for full reproduction`
- `GQA real baseline validated`
- `VCR real baseline validated`
- `Q->AR reproduced`

Разрешённая текущая формулировка:

```text
partially validated baseline: architecture split, task contracts,
relation-aware GQA model path, strict GQA data package, validators и
demo/runtime checks реализованы или validated как задокументировано. GQA data
validated end-to-end against real Kaggle data; VCR data preparation остаётся
открытой задачей. Paper numbers не воспроизведены.
```
