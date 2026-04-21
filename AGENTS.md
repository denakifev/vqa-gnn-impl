# AGENTS.md

## Назначение

Документ задаёт рабочие правила для агентов, которые продолжают разработку и
аудит VQA-GNN в этом репозитории.

Актуальный статус проекта: `partially validated baseline`.

Репозиторий уже не является только шаблоном. В нём реализован и проверен
архитектурный split:

- VCR: `VCRVQAGNNModel`, candidate scoring, отдельные Q→A и QA→R запуски.
- GQA: `GQAVQAGNNModel`, 1842-way classification.
- Общее ядро: `src/model/gnn_core.py`.

При этом репозиторий **не** является `paper reproduced` и **не** является
`fully paper-faithful`. Реальные paper numbers отсутствуют.

## Главные Правила

- Сначала читать структуру проекта, релевантные configs и существующие
  interfaces, потом менять код.
- Не выдавать demo datasets, synthetic smoke runs или engineering approximations
  за результаты статьи.
- Не создавать параллельный training framework: использовать `train.py`,
  `inference.py`, `src/trainer/`, `src/logger/`, `src/utils/`.
- Не обходить Hydra ручным parsing аргументов.
- Не менять формат batch без проверки полного контура:
  dataset → collate → model → loss → metrics → trainer → configs.
- Любое отклонение от статьи фиксировать в документации.
- Любой real-data claim должен быть подтверждён командой и результатом запуска.

## Источники Истины

Актуальные документы:

- [README.md](README.md) — краткий статус и основные команды.
- [VALIDATION_REPORT.md](VALIDATION_REPORT.md) — проверенные команды и результаты.
- [ARCHITECTURE_SPLIT.md](ARCHITECTURE_SPLIT.md) — разделение VCR/GQA.
- [PAPER_ALIGNMENT_REPORT.md](PAPER_ALIGNMENT_REPORT.md) — соответствие статье.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — блокеры и инженерные отклонения.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — runtime-контракты для реальных датасетов.
- [RUNBOOK.md](RUNBOOK.md) — практический запуск.
- [KAGGLE_GQA.md](KAGGLE_GQA.md), [KAGGLE_VCR.md](KAGGLE_VCR.md) — пути запуска на Kaggle.

`CLAUDE.md` удалён как устаревший документ.

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
- `Q→AR reproduced`

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
- `graph_edge_types` находится в batch contract, но пока не используется в
  message passing.

### Shared Core

- `DenseGATLayer`
- `HFQuestionEncoder`

Shared core должен оставаться слоем primitives, а не местом скрытого task switch.

## Контракты Данных

Перед запуском на реальных данных обязательно проверить [DATA_CONTRACTS.md](DATA_CONTRACTS.md).

Текущая локальная находка:

- VCR real data отсутствуют: `data/vcr/train.jsonl` не найден.
- GQA local graph artifacts имеют legacy `d_kg=300`, без `graph_edge_types`;
  current GQA config ожидает `d_kg=600`.

Это `blocker for first real run`, а не model bug.

## Роли

### Архитектор Репозитория

Отвечает за:

- структуру `src/` и `src/configs/`;
- import/re-export contracts;
- согласованность Hydra composition;
- документацию архитектурных решений.

Может менять:

- `src/model/`, `src/configs/model/`, package exports;
- top-level configs;
- корневую документацию.

### Агент Данных

Отвечает за:

- dataset readers;
- HDF5/JSON/JSONL contracts;
- preprocessing scripts;
- validation scripts;
- batch keys, которые выходят из dataset/collate.

Может менять:

- `src/datasets/`;
- `src/configs/datasets/`;
- `scripts/prepare_*data.py`;
- `scripts/validate_*data.py`.

### Агент Модели

Отвечает за:

- model forward contracts;
- GNN/text/fusion modules;
- model configs;
- архитектурные deviations от статьи.

Может менять:

- `src/model/`;
- `src/configs/model/`;
- связанные package exports.

### Агент Обучения И Проверки

Отвечает за:

- trainer/inferencer compatibility;
- losses;
- metrics;
- checkpoint compatibility;
- smoke/runtime validation.

Может менять:

- `src/trainer/`;
- `src/loss/`;
- `src/metrics/`;
- `src/configs/baseline*.yaml`;
- `src/configs/inference*.yaml`.

### Агент Документации И Воспроизводимости

Отвечает за:

- README/status reports;
- runbooks;
- data contracts;
- limitations and claims;
- handoff notes.

Может менять:

- корневые `.md`;
- инструкции по окружению, данным, запуску и ограничениям.

## Протокол Handoff

Каждый handoff должен содержать:

1. Цель выполненной части работы.
2. Изменённые файлы.
3. Критически связанные файлы, которые не менялись.
4. Изменения interfaces: batch keys, config keys, model I/O, data paths.
5. Обнаруженные или добавленные deviations от статьи.
6. Что должен проверить следующий агент.

Если задача не завершена, нужно явно разделить:

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

## Проверка Перед Завершением Задачи

- Hydra `_target_` paths согласованы.
- Batch keys согласованы между dataset, collate, model, loss, metrics и trainer.
- Документация отражает реальные assumptions и deviations.
- Нет ложных claims о данных, numbers или paper reproduction.
- `config.yaml`, `git_commit.txt`, `git_diff.patch`, checkpoints и logs не сломаны.
- Не добавлена лишняя инфраструктура без необходимости.

Минимальная проверка для runtime-sensitive изменений:

```bash
.venv/bin/python -m pytest -q
```

Для data-contract изменений:

```bash
.venv/bin/python -m pytest tests/test_engineering_contracts.py -q
```
