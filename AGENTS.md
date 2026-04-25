# AGENTS.md

## Назначение

Рабочие правила для агентов, которые продолжают разработку и аудит VQA-GNN в
этом репозитории.

Актуальный статус проекта: `partially validated baseline`.

Репозиторий реализует архитектурный split:

- VCR: `VCRVQAGNNModel`, candidate scoring, отдельные Q->A и QA->R запуски.
- GQA: `GQAVQAGNNModel`, 1842-way classification.
- Общее ядро: `src/model/gnn_core.py`.

При этом репозиторий **не** является `paper reproduced` и **не** является
`fully paper-faithful`. Реальные paper numbers отсутствуют.

## Главные Правила

- Сначала читать структуру проекта, релевантные configs и существующие
  interfaces, потом менять код.
- Не выдавать demo datasets, synthetic smoke runs или engineering
  approximations за результаты статьи.
- Не создавать параллельный training framework: использовать `train.py`,
  `inference.py`, `src/trainer/`, `src/logger/`, `src/utils/`.
- Не обходить Hydra ручным parsing аргументов.
- Не менять формат batch без проверки полного контура:
  dataset -> collate -> model -> loss -> metrics -> trainer -> configs.
- Любое отклонение от статьи фиксировать в документации.
- Любой real-data claim должен быть подтверждён командой и результатом запуска.

## Источники Истины

Активные документы:

- [README.md](README.md) — текущий статус, команды, data state.
- [ARCHITECTURE_SPLIT.md](ARCHITECTURE_SPLIT.md) — разделение VCR/GQA.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — блокеры, limitations, deviations.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — strict GQA data contract.
- [GQA_CLOSURE_REPORT.md](GQA_CLOSURE_REPORT.md) — итог GQA-фазы.

Остальные старые отчёты удалены или считаются superseded.

Для GQA source-of-truth runtime model — `GQAVQAGNNModel`
(`baseline_gqa.yaml`). `PaperGQAModel` — equation reference, не runtime.

## Рабочие Статусы

Использовать только эти status tokens:

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
- `VCR real baseline validated`
- `Q->AR reproduced`

## Текущее Состояние Данных

### GQA

- Strict GQA package был локально собран и `validated`.
- Package был загружен на Kaggle из `kaggle_staging/gqa-gnn-data`.
- Full local GQA processed/raw/staging payloads были намеренно удалены после
  upload.
- Локальный `data/gqa` сейчас не обязан существовать.
- Для локальных GQA запусков нужно восстановить private Kaggle dataset в
  expected layout или указать Hydra `datasets.*.data_dir` на восстановленный
  путь.

Validated strict GQA contract до cleanup:

- `answer_to_idx` exists, size `1842`;
- relation vocab exists, relation count `624`;
- visual features train/val: `float32[100, 2048]`;
- graph datasets: `node_features`, `adj_matrix`, `node_types`,
  `graph_edge_types`;
- graph `node_features.shape[-1] == 600`;
- graph attrs include `d_kg=600`, `graph_edge_type_count=624`,
  `graph_mode=official_scene_graph`, `conceptnet_used=False`;
- `scripts/validate_gqa_data.py --skip-runtime-check` passed with `0` errors.

### VCR

- Local VCR real data не подготовлены.
- VCR data лицензированы; Kaggle dataset должен оставаться private.
- Baseline-level VCR package должен содержать:

```text
data/vcr/train.jsonl
data/vcr/val.jsonl
data/vcr/visual_features/train_features.h5
data/vcr/visual_features/val_features.h5
data/vcr/knowledge_graphs/train_graphs.h5
data/vcr/knowledge_graphs/val_graphs.h5
```

## Ключевые Контракты

### VCR

- Model: `src/model/vcr_model.py::VCRVQAGNNModel`.
- Dataset: `VCRDataset`, `VCRDemoDataset`.
- Collate: `make_vcr_collate_fn`.
- Loss: `VCRLoss`.
- Metrics: `VCRQAAccuracy`, `VCRQARAccuracy`, `VCRQARJointAccuracy`.
- Output contract: `[B*4, 1]`, один score на candidate.
- `num_answers` не является внешним параметром VCR.

### GQA

- Model: `src/model/gqa_model.py::GQAVQAGNNModel`.
- Dataset: `GQADataset`, `GQADemoDataset`.
- Collate: `gqa_collate_fn`.
- Loss: `GQALoss`.
- Metric: `GQAAccuracy`.
- Output contract: `[B, 1842]`.
- `graph_edge_types` находится в batch contract и consumed by
  relation-aware GQA message passing.

### Общее Ядро

- `DenseGATLayer`
- `HFQuestionEncoder`

Общее ядро должно оставаться слоем primitives, а не местом скрытого task
switch.

## Роли

### Архитектор Репозитория

Отвечает за `src/`, configs, imports/re-exports, Hydra composition и
архитектурную документацию.

### Агент Данных

Отвечает за dataset readers, HDF5/JSON/JSONL contracts, preprocessing scripts,
validation scripts и batch keys.

### Агент Модели

Отвечает за model forward contracts, GNN/text/fusion modules, model configs и
architectural deviations.

### Агент Обучения И Проверки

Отвечает за trainer/inferencer compatibility, losses, metrics, checkpoints,
smoke/runtime validation.

### Агент Документации И Воспроизводимости

Отвечает за README/status reports, runbooks, data contracts, limitations,
claims и handoff notes.

## Протокол Handoff

Каждый handoff должен содержать:

1. Цель выполненной части работы.
2. Изменённые файлы.
3. Критически связанные файлы, которые не менялись.
4. Изменения interfaces: batch keys, config keys, model I/O, data paths.
5. Обнаруженные или добавленные deviations от статьи.
6. Что должен проверить следующий агент.

Если задача не завершена, явно разделить:

- что уже сделано;
- что осталось;
- какие риски открыты.

## Критические Файлы И Каталоги

- `train.py`
- `inference.py`
- `requirements.txt`
- `.pre-commit-config.yaml`
- `.flake8`
- `src/configs/`
- `src/model/`
- `src/datasets/`
- `src/trainer/`
- `src/loss/`
- `src/metrics/`
- `src/logger/`
- `src/utils/init_utils.py`
- `src/utils/io_utils.py`
- `scripts/prepare_*data.py`
- `scripts/validate_*data.py`
- `LICENSE`

Runtime artifacts:

- `data/`
- `saved/`
- `data/saved/`
- `kaggle_staging/`

## Проверка Перед Завершением Задачи

- Hydra `_target_` paths согласованы.
- Batch keys согласованы между dataset, collate, model, loss, metrics и
  trainer.
- Документация отражает реальные assumptions и deviations.
- Нет ложных claims о данных, numbers или paper reproduction.
- `config.yaml`, `git_commit.txt`, `git_diff.patch`, checkpoints и logs не
  сломаны.
- Не добавлена лишняя инфраструктура без необходимости.

Минимальная проверка для runtime-sensitive изменений:

```bash
.venv/bin/python -m pytest -q
```

Для data-contract изменений:

```bash
.venv/bin/python -m pytest tests/test_engineering_contracts.py -q
```
