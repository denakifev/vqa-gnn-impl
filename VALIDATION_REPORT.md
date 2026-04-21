# Отчёт Валидации

Дата аудита: 2026-04-20.

Роль аудита: проверить текущее состояние репозитория, а не улучшать качество
модели.

## Краткий Итог

Подтверждено:

- VCR и GQA разделены на уровне model/config/dataset/collate/loss/metrics.
- VCR работает как candidate scoring: `[B*4, 1]`.
- GQA работает как answer-vocabulary classification: `[B, 1842]`.
- GQA GNN потребляет `graph_edge_types` через learned per-(head, relation)
  attention-bias в каждом dense GAT слое (relation-aware message passing).
- VCR object references рендерятся как per-instance identifiers
  (`person0`, `person2`) — identity preserved при одной категории.
- Post-hoc Q→AR evaluator (`scripts/eval_vcr_qar.py`) runtime-validated;
  inferencer persists `sample_id` для стабильного join.
- Package imports и backward-compatible exports работают внутри `.venv`.
- Тесты проходят: `81 passed` (56 baseline + 14 relation-aware/VCR-ids + 11 Q→AR evaluator).
- Demo smoke training работает для VCR Q→A, VCR QA→R и GQA с локальным tiny
  RoBERTa override.
- Demo inference config composition работает для VCR и GQA с declared override
  `inferencer.skip_model_load=True`.

Не подтверждено:

- real VCR training;
- local real GQA training;
- post-hoc Q→AR over paired real predictions (evaluator готов, predictions нет);
- paper numbers.

Главный gap:

- локальные GQA graph artifacts несовместимы с текущим paper-like GQA config:
  artifacts имеют 300-d `node_features` и не содержат `graph_edge_types`, а
  config ожидает `d_kg=600`.

Готовность:

- short sanity runs: `validated`;
- первый VCR real run: `blocked by data`, локальные VCR данные отсутствуют;
- первый local GQA real run: `blocked by data`, artifact/config mismatch;
- paper-faithful reproduction attempt: `future work`.

Финальный вердикт аудита: `improved paper-aligned baseline`.

## 1. Валидация Архитектурного Split

Статическая проверка:

| Слой | VCR | GQA | Статус |
|---|---|---|---|
| Model | `VCRVQAGNNModel` | `GQAVQAGNNModel` | `validated` |
| Config | `vqa_gnn_vcr.yaml` | `vqa_gnn_gqa.yaml` | `validated` |
| Dataset | `VCRDataset`, `VCRDemoDataset` | `GQADataset`, `GQADemoDataset` | `validated` |
| Collate | `make_vcr_collate_fn` | `gqa_collate_fn` | `validated` |
| Loss | `VCRLoss` | `GQALoss` | `validated` |
| Metrics | `VCRQAAccuracy`, `VCRQARAccuracy` | `GQAAccuracy` | `validated` |

Скрытого task switch внутри task-specific моделей нет. Старый `VQAGNNModel`
оставлен для backward compatibility, но paper-aligned baseline configs его не
выбирают.

Общее ядро:

- `src/model/gnn_core.py::DenseGATLayer`;
- `src/model/gnn_core.py::HFQuestionEncoder`.

Shared core оправдан только как набор переиспользуемых примитивов.

## 2. Runtime Validation

Проверка окружения:

| Команда | Результат | Статус |
|---|---|---|
| `python -V` | `Python 3.14.3` | observed |
| `python -c "import train, inference"` | падает: `ModuleNotFoundError: No module named 'hydra'` | system Python `not validated yet` |
| `.venv/bin/python -V` | `Python 3.14.3` | observed |
| `.venv/bin/python -c "import torch, hydra, omegaconf, transformers, h5py, comet_ml, pytest"` | imports OK | `.venv` `validated` |
| `.venv/bin/python -c "import train, inference"` | entrypoint imports OK | `validated` |

Версии в `.venv`:

- `torch 2.11.0`;
- `transformers 5.5.3`;
- `pytest 9.0.3`.

Неблокирующие warnings окружения:

- Matplotlib cache path `/Users/srs15/.matplotlib` не writable.
- Fontconfig cache paths не writable.
- Comet offline mode логирует warnings про import order и IP logging.

Эти warnings не блокировали tests или smoke runs.

## 3. Tests И Smoke Validation

Матрица тестов:

| Команда | Результат | Статус |
|---|---|---|
| `.venv/bin/python -m pytest -q` | `81 passed` | `validated` |
| `.venv/bin/python -m pytest tests/test_paper_path.py -q` | `46 passed` (32 baseline + 14 new: `TestRelationAwareDenseGAT`, `TestGQAModelConsumesEdgeTypes`, `TestVCRInstanceIdentifiers`) | `validated` |
| `.venv/bin/python -m pytest tests/test_eval_vcr_qar.py -q` | `11 passed` (`TestLoadPredictions`, `TestComputeJointReport`, `TestCLI`) | `validated` |
| `.venv/bin/python -m pytest tests/test_gqa_pipeline.py -q` | `5 passed` | `validated` |
| `.venv/bin/python -m pytest tests/test_gqa_ner_pipeline.py -q` | `4 passed` | `validated` |
| `.venv/bin/python -m pytest tests/test_engineering_contracts.py -q` | `15 passed` | `validated` |
| `python -m pytest -q` | падает: `No module named pytest` | system Python `not validated yet` |

Shape smoke validation:

| Контур | Наблюдаемые shapes | Статус |
|---|---|---|
| VCR demo | `visual_features [8,4,8]`, `question_input_ids [8,8]`, `graph_adj [8,8,8]`, logits `[8,1]`, labels `[2]` | `runtime-valid` |
| GQA demo | `visual_features [2,4,8]`, `question_input_ids [2,8]`, `graph_adj [2,8,8]`, `graph_edge_types [2,8,8]`, logits `[2,10]`, labels `[2]` | `runtime-valid` |

Смоук-проверки обучения:

| Config | Audit mode | Результат | Статус |
|---|---|---|---|
| `baseline_vcr_qa` | tiny local RoBERTa, one step, offline writer | train + val + checkpoint завершились | `runtime-valid` |
| `baseline_vcr_qar` | tiny local RoBERTa, one step, offline writer | train + val + checkpoint завершились | `runtime-valid` |
| `baseline_gqa` | tiny local RoBERTa, one step, offline writer | train + val + checkpoint завершились | `runtime-valid` |

Tiny model был создан в `/tmp/vqa_gnn_tiny_roberta` через `RobertaConfig`,
чтобы проверить stack compatibility без загрузки и запуска полного
`roberta-large` на CPU.

Смоук-проверки инференса:

| Config | Команда | Результат |
|---|---|---|
| `inference_vcr` | `inferencer.skip_model_load=True` | config composes; demo inference path `runtime-valid` |
| `inference_gqa` | `inferencer.skip_model_load=True` | config composes; demo inference path `runtime-valid` |

Hydra ambiguity вокруг `skip_model_load` исправлена: ключ объявлен в
`inference_vcr.yaml` и `inference_gqa.yaml`. Bare override syntax теперь
`validated`.

Проверенные override-команды:

```bash
.venv/bin/python inference.py --config-name inference_vcr inferencer.skip_model_load=True --cfg job
.venv/bin/python inference.py --config-name inference_gqa inferencer.skip_model_load=True --cfg job
```

## 4. Config И CLI Validation

Hydra composition команды:

```bash
.venv/bin/python train.py --config-name baseline_vcr_qa --cfg job
.venv/bin/python train.py --config-name baseline_vcr_qar --cfg job
.venv/bin/python train.py --config-name baseline_gqa --cfg job
.venv/bin/python inference.py --config-name inference_vcr --cfg job
.venv/bin/python inference.py --config-name inference_gqa --cfg job
```

Все команды собрали config успешно.

Подтверждённые selections:

- `baseline_vcr_qa`: `src.model.VCRVQAGNNModel`, `VCRDemoDataset`,
  `VCRLoss`, `VCRQAAccuracy`;
- `baseline_vcr_qar`: `src.model.VCRVQAGNNModel`, `VCRDemoDataset` с
  `task_mode=qar`, `VCRLoss`, `VCRQARAccuracy`;
- `baseline_gqa`: `src.model.GQAVQAGNNModel`, `GQADemoDataset`, `GQALoss`,
  `GQAAccuracy`.

Default caveats:

- Training configs по умолчанию используют `writer.mode=online`; локальные runs
  лучше запускать с `writer.mode=offline`, если Comet online credentials не настроены.
- Полная default model использует `roberta-large`; в новой среде потребуется
  HF cache/download или локальный checkpoint.

## 5. Real-Data Startup Audit

VCR Q→A real config:

```bash
.venv/bin/python train.py --config-name baseline_vcr_qa datasets=vcr_qa dataloader.num_workers=0 trainer.n_epochs=1 trainer.epoch_len=1 trainer.override=True trainer.save_dir=/tmp/vqa_gnn_audit_saved writer.mode=offline writer.run_name=audit_real_vcr_qa_missing_data
```

Результат:

```text
FileNotFoundError: VCR annotations (train) not found at: data/vcr/train.jsonl
```

Классификация: `missing data issue`, `blocker for first real run`.

VCR QA→R real config:

```bash
.venv/bin/python train.py --config-name baseline_vcr_qar datasets=vcr_qar dataloader.num_workers=0 trainer.n_epochs=1 trainer.epoch_len=1 trainer.override=True trainer.save_dir=/tmp/vqa_gnn_audit_saved writer.mode=offline writer.run_name=audit_real_vcr_qar_missing_data
```

Результат:

```text
FileNotFoundError: VCR annotations (train) not found at: data/vcr/train.jsonl
```

Классификация: `missing data issue`, `blocker for first real run`.

GQA real config:

```bash
.venv/bin/python train.py --config-name baseline_gqa datasets=gqa dataloader.num_workers=0 trainer.n_epochs=1 trainer.epoch_len=1 trainer.override=True trainer.save_dir=/tmp/vqa_gnn_audit_saved writer.mode=offline writer.run_name=audit_real_gqa_missing_data
```

Результат:

```text
ValueError: [GQA data contract] Graph HDF5 sample '02947959' has node_features with d_kg=300, but dataset config requests d_kg=600.
  file: data/gqa/knowledge_graphs/train_graphs.h5
  fix options (pick one):
    (a) rebuild graphs with the current scene-graph pipeline:
        python scripts/prepare_gqa_data.py knowledge-graphs ... --d-kg 600
    (b) align config to the stored artifact — override both
        model.d_kg=300 and datasets.train.d_kg=300 datasets.val.d_kg=300
        (non paper-faithful; see DATA_CONTRACTS.md).
  blocking a silent dim mismatch mid-batch is intentional.
```

Классификация: `data contract mismatch`, `blocker for first real run`.

Наблюдаемые локальные GQA artifacts:

| Artifact | Наблюдение |
|---|---|
| `data/gqa/gqa_answer_vocab.json` | 1842 answers |
| `data/gqa/visual_features/train_features.h5` | attrs `{}`, first sample `(100, 2048)` |
| `data/gqa/knowledge_graphs/train_graphs.h5` | attrs `{}`, first sample `node_features (2, 300)` |
| `graph_edge_types` в local graph HDF5 | отсутствует |

Текущий config ожидает:

- `d_kg=600`;
- `graph_edge_types` в batch contract;
- paper-like scene-graph HDF5 attrs.

Mismatch теперь ловится fail-fast validation в `src/datasets/gqa_dataset.py` на
первом HDF5 open. Legacy 300-d artifacts допустимы только через явные
non-paper-faithful overrides из [DATA_CONTRACTS.md](DATA_CONTRACTS.md).

## 6. Paper Alignment Validation

Уже `paper-aligned`:

- VCR candidate scoring contract.
- VCR Q→A и QA→R split.
- VCR CE по четырём candidate scores.
- GQA hard CE и exact-match metric.
- GQA `num_answers=1842`.
- Post-hoc Q→AR evaluator (`scripts/eval_vcr_qar.py`, `tests/test_eval_vcr_qar.py`).

`paper-aligned approximation`:

- dense GAT с learned per-(head, relation) attention bias;
- `roberta-large` как default text encoder;
- offline graph preprocessing;
- VCR object refs рендерятся как per-instance identifiers (`person0`/`person2`);
- deterministic GQA answer vocab по released balanced splits.

`not paper-faithful yet`:

- Локальные GQA artifacts не собраны текущим paper-like scene-graph path.
- VCR region grounding (detector features → mentions) не реализован.
- Paper-number run отсутствует.
- Numerical equivalence relation-aware attention bias с paper GNN form
  не verified.

## 7. Blockers И Risks

`blocked by data`:

- отсутствуют реальные VCR данные;
- локальные GQA graph artifacts имеют mismatch с `d_kg=600`;
- локальные GQA graph HDF5 не содержат `graph_edge_types`;
- `num_relations` нельзя свести к фактическому `relation_count_total`
  до пересборки graph artifacts.

`future work`:

- реальные Q→A, QA→R и GQA numbers;
- post-hoc Q→AR run over paired saved predictions;
- VCR region grounding (detector features → mentions);
- numerical equivalence relation-aware attention bias с paper GNN form;
- exact paper training protocol parity.

`non-blocking engineering deviation`:

- dense GAT;
- offline HDF5 preprocessing;
- Comet offline warnings;
- Matplotlib/fontconfig cache warnings.

## 8. Оценка Готовности

| Цель | Оценка | Причина |
|---|---|---|
| Short sanity runs | `runtime-valid` | tests и demo smoke passed |
| Relation-aware GQA message passing | `runtime-valid` | `graph_edge_types` передаются во все GAT слои; bias init at zero, unit tests verify consumption |
| VCR per-instance object identifiers | `runtime-valid` | `_vcr_object_mention` tests; demo sample shows `person0`/`person2` |
| Post-hoc Q→AR evaluator | `validated` | `tests/test_eval_vcr_qar.py` pass end-to-end including CLI |
| Real VCR baseline training | `blocked by data` | missing local data |
| Local real GQA baseline training | `blocked by data` | graph artifact mismatch |
| Long Kaggle-style GQA run | `not validated yet` | code path plausible, текущие local artifacts не являются proof |
| Paper-faithful reproduction attempt | `future work` | real numbers absent |

## Финальный Вердикт

`improved paper-aligned baseline`.

Более точная формулировка:

```text
improved paper-aligned baseline: architecture split, loss/metrics, tests,
Hydra composition, relation-aware GQA message passing, VCR per-instance
object identifiers и post-hoc Q→AR evaluator — всё validated на demo/tiny
path; real-data paper reproduction остаётся blocked by data/artifact gaps.
```
