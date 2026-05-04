# AGENTS.md

## Назначение

Рабочие правила для агентов, которые продолжают разработку и аудит
GQA/VQA-2-focused VQA-GNN в этом репозитории.

Актуальный статус проекта: `partially validated baseline`.

Текущая стратегия: практичные GQA и VQA-2 train paths, которые можно запускать
в Kaggle лимитах, затем сравнивать с архитектурными изменениями. Текущий
research branch: additive graph-link module поверх GQA baseline. GQA цель —
`paper-aligned approximation`; VQA-2 — coursework extension, не результат
статьи.

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
- [PROJECT_SCOPE.md](PROJECT_SCOPE.md) — explicit repository scope.
- [ARCHITECTURE_SPLIT.md](ARCHITECTURE_SPLIT.md) — active architecture split.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — limitations, deviations, Kaggle risks.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — GQA and VQA-2 data contracts.
- [GQA_CLOSURE_REPORT.md](GQA_CLOSURE_REPORT.md) — итог GQA engineering-фазы.
- [GRAPH_LINK_MODULE_DESIGN.md](GRAPH_LINK_MODULE_DESIGN.md) — active
  research-module design and experiment modes.

Repository mode must stay explicit:

- GQA main
- VQA-2 auxiliary
- no removed benchmark in active scope

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
- `VQA-2 real baseline validated`

## Текущее Состояние Данных

Strict GQA package был локально собран, uploaded to Kaggle и затем
validated end-to-end against restored real Kaggle data.

Локальный `data/gqa` сейчас не обязан существовать. Для локальных GQA запусков
нужно восстановить private Kaggle dataset в expected layout или указать Hydra
`datasets.*.data_dir` на восстановленный путь.

Validated strict GQA contract:

- `answer_to_idx` exists, size `1842`;
- relation vocab exists, relation count `624`;
- visual features train/val: `float32[100, 2048]`;
- graph datasets: `node_features`, `adj_matrix`, `node_types`,
  `graph_edge_types`;
- graph `node_features.shape[-1] == 600`;
- graph attrs include `d_kg=600`, `graph_edge_type_count=624`,
  `graph_mode=official_scene_graph`, `conceptnet_used=False`,
  `fully_connected_fallback_used=False`;
- runtime path validated:
  `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy`.

VQA-2 real data package сейчас `not validated yet`. VQA-2 expected layout и
batch contract описаны в `DATA_CONTRACTS.md`. Перед real train запускать
`scripts/validate_vqa_data.py`.

## Ключевые Контракты

### GQA Runtime Path

- Model: `src/model/gqa_model.py::GQAVQAGNNModel`.
- Dataset: `GQADataset`, `GQADemoDataset`.
- Collate: `gqa_collate_fn`.
- Loss: `GQALoss`.
- Metric: `GQAAccuracy`.
- Output contract: `[B, 1842]`.
- `graph_edge_types` находится в batch contract и consumed by
  relation-aware GQA message passing.
- Optional research extension: `SparseGraphLinkModule`, включаемый только
  конфигом и не меняющий dataset contract.

### VQA-2 Runtime Path

- Model: `src/model/vqa_gnn.py::VQAGNNModel`.
- Dataset: `VQADataset`, `VQADemoDataset`.
- Collate: `vqa_collate_fn`.
- Loss: `VQALoss`.
- Metric: `VQAAccuracy`.
- Output contract: `[B, 3129]`.
- `answer_scores` находится в batch contract и consumed by `VQALoss` /
  `VQAAccuracy`.
- Не claim'ить VQA-2 как validated paper task.

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

Отвечает за dataset readers, HDF5/JSON contracts, preprocessing scripts,
validation scripts и batch keys.

### Агент Модели

Отвечает за model forward contracts, GNN/text/fusion modules, model configs и
architectural deviations.

При graph-link изменениях агент должен:

- сохранять baseline path доступным;
- не вносить скрытые изменения в baseline режим;
- держать on/off control и freeze policy явными в config.

### Агент Обучения И Проверки

Отвечает за trainer/inferencer compatibility, losses, metrics, checkpoints,
smoke/runtime validation и Kaggle budget fit.

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
