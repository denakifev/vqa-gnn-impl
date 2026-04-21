# VQA-GNN: воспроизводимый baseline

Репозиторий содержит PyTorch/Hydra-реализацию для воспроизводимого аудита
статьи:

> VQA-GNN: Reasoning with Multimodal Knowledge via Graph Neural Networks for
> Visual Question Answering, ICCV 2023 / arXiv:2205.11501.

Актуальный статус на 2026-04-20: `improved paper-aligned baseline`
(baseline completion pass завершён; числа из статьи всё ещё не
воспроизведены и реальные данные остаются gap).

Реализован и проверен архитектурный split VCR/GQA, demo-smoke контур
работоспособен, GQA GNN теперь consumes `graph_edge_types`, VCR object
references рендерятся как per-instance identifiers, а Q→AR joint
accuracy считается через отдельный runtime-valid evaluator.

## Актуальные Документы

- [VALIDATION_REPORT.md](VALIDATION_REPORT.md) — матрица команд, тестов и результатов аудита.
- [ARCHITECTURE_SPLIT.md](ARCHITECTURE_SPLIT.md) — разделение VCR/GQA.
- [PAPER_ALIGNMENT_REPORT.md](PAPER_ALIGNMENT_REPORT.md) — статус соответствия статье.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — оставшиеся блокеры и приближения.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — runtime-контракты данных и fail-fast проверки.
- [GQA_DATA_PROCESSING_REPORT.md](GQA_DATA_PROCESSING_REPORT.md) — решения по препроцессингу GQA.
- [BASELINE_COMPLETION_REPORT.md](BASELINE_COMPLETION_REPORT.md) — итог прохода baseline completion перед экспериментами.
- [RUNBOOK.md](RUNBOOK.md) — практический регламент запуска.
- [KAGGLE_GQA.md](KAGGLE_GQA.md), [KAGGLE_VCR.md](KAGGLE_VCR.md) — Kaggle-инструкции.

## Что Реализовано

| Область | Статус | Файлы |
|---|---|---|
| VCR Q→A candidate scoring | `implemented`, `validated` | `src/model/vcr_model.py`, `src/datasets/vcr_collate.py`, `src/loss/vcr_loss.py`, `src/metrics/vcr_metric.py` |
| VCR QA→R candidate scoring | `implemented`, `validated` | `baseline_vcr_qar.yaml`, `VCRQARAccuracy` |
| VCR Q→AR формула | `implemented`, toy `validated` | `VCRQARJointAccuracy` |
| Q→AR joint evaluator (post-hoc) | `implemented`, `validated` | `scripts/eval_vcr_qar.py`, `tests/test_eval_vcr_qar.py` |
| GQA 1842-way classification | `implemented`, `validated` | `src/model/gqa_model.py`, `src/loss/gqa_loss.py`, `src/metrics/gqa_metric.py` |
| GQA relation-aware message passing | `implemented`, `validated` (типизированные edge bias в attention) | `src/model/gnn_core.py::DenseGATLayer`, `src/model/gqa_model.py` |
| VCR per-instance object identifiers | `implemented`, `validated` (approximation of region grounding) | `src/datasets/vcr_dataset.py::_vcr_object_mention` |
| Общее GNN/text ядро | `implemented`, `validated` | `src/model/gnn_core.py` |
| Demo datasets | `runtime-valid` | `VCRDemoDataset`, `GQADemoDataset` |
| Real datasets | `implemented`, частично `not validated yet` | `VCRDataset`, `GQADataset` |

## Что Проверено

Проверенный путь окружения: `.venv/bin/python`.

```bash
.venv/bin/python -m pytest -q
```

Результат последней проверки:

```text
81 passed
```

Целевые тесты:

```bash
.venv/bin/python -m pytest tests/test_paper_path.py -q
.venv/bin/python -m pytest tests/test_gqa_pipeline.py -q
.venv/bin/python -m pytest tests/test_gqa_ner_pipeline.py -q
.venv/bin/python -m pytest tests/test_engineering_contracts.py -q
```

Hydra composition проверена для:

```bash
.venv/bin/python train.py --config-name baseline_vcr_qa --cfg job
.venv/bin/python train.py --config-name baseline_vcr_qar --cfg job
.venv/bin/python train.py --config-name baseline_gqa --cfg job
.venv/bin/python inference.py --config-name inference_vcr inferencer.skip_model_load=True --cfg job
.venv/bin/python inference.py --config-name inference_gqa inferencer.skip_model_load=True --cfg job
```

Demo training/inference smoke-run проверялся с tiny RoBERTa override, чтобы не
запускать полный `roberta-large` на CPU во время аудита. Точные команды
зафиксированы в [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

## Что Пока Не Подтверждено

- Реальный VCR Q→A запуск не завершался (`blocked by data`).
- Реальный VCR QA→R запуск не завершался (`blocked by data`).
- Локальный GQA real run не проходит на текущих `data/gqa` artifacts (`blocked by data`).
- Paper-verified numbers of relation-aware GQA GNN не измерены (`future work`).
- VCR region grounding бинденный к detector features не реализован; per-instance identifier — это approximation (`future work`).
- Числа из статьи не воспроизведены (`future work`).

## Текущие Runtime Findings

VCR real-data configs падают ожидаемо, потому что локальные VCR данные отсутствуют:

```text
FileNotFoundError: VCR annotations (train) not found at: data/vcr/train.jsonl
```

GQA real-data config находит локальные artifacts, но теперь падает fail-fast на
явном data contract:

```text
ValueError: [GQA data contract] Graph HDF5 sample ... has node_features with d_kg=300, but dataset config requests d_kg=600.
```

Наблюдаемое состояние локальных GQA artifacts:

- `data/gqa/gqa_answer_vocab.json`: 1842 ответа.
- `data/gqa/visual_features/train_features.h5`: sample shape `(100, 2048)`.
- `data/gqa/knowledge_graphs/train_graphs.h5`: sample `node_features (2, 300)`.
- HDF5 attrs у локального GQA graph file пустые.
- `graph_edge_types` в локальном GQA graph file отсутствует.

Текущий GQA config ожидает `d_kg=600` и typed scene-graph edges. Значит graph
artifacts нужно пересобрать через текущий scene-graph preprocessing path или
явно запустить legacy non-paper-faithful режим с `d_kg=300` и
`expect_graph_edge_types=False`. Подробности: [DATA_CONTRACTS.md](DATA_CONTRACTS.md).

## Установка

Рекомендуемый путь:

```bash
source .venv/bin/activate
python -m pytest -q
```

Если окружение нужно собрать заново:

```bash
pip install -r requirements.txt
pip install pytest
```

Аудитное замечание: `pytest` есть в текущем `.venv`, но пока не указан в
`requirements.txt`. Системный Python в проверенной машине не содержит всех
зависимостей проекта.

## Быстрые Команды

Проверка composition:

```bash
python train.py --config-name baseline_vcr_qa --cfg job
python train.py --config-name baseline_vcr_qar --cfg job
python train.py --config-name baseline_gqa --cfg job
```

Inference demo без checkpoint:

```bash
python inference.py --config-name inference_vcr inferencer.skip_model_load=True
python inference.py --config-name inference_gqa inferencer.skip_model_load=True
```

## Точки Входа Для Реальных Данных

VCR Q→A:

```bash
python train.py --config-name baseline_vcr_qa \
  datasets=vcr_qa \
  datasets.train.data_dir=data/vcr \
  datasets.val.data_dir=data/vcr
```

VCR QA→R:

```bash
python train.py --config-name baseline_vcr_qar \
  datasets=vcr_qar \
  datasets.train.data_dir=data/vcr \
  datasets.val.data_dir=data/vcr
```

GQA:

```bash
python train.py --config-name baseline_gqa \
  datasets=gqa \
  datasets.train.data_dir=data/gqa \
  datasets.train.answer_vocab_path=data/gqa/gqa_answer_vocab.json \
  datasets.val.data_dir=data/gqa \
  datasets.val.answer_vocab_path=data/gqa/gqa_answer_vocab.json
```

Перед GQA запуском проверьте:

- `d_kg=600`;
- `graph_edge_types` присутствует;
- HDF5 attrs содержат `d_kg`, `num_visual_nodes`, `max_kg_nodes`;
- visual features имеют форму `100×2048`;
- answer vocabulary содержит 1842 класса.

## Q→AR Evaluation

Q→AR требует две отдельные inference prognon: Q→A и QA→R. Inferencer
сохраняет `sample_id` (VCR `annot_id` / GQA `question_id`) рядом с
`pred_label` и `label`, поэтому post-hoc evaluator может соединить
predictions по идентификатору:

```bash
python inference.py --config-name inference_vcr \
  datasets=vcr_qa \
  inferencer.save_path=saved/vcr_qa_preds \
  inferencer.from_pretrained=saved/vcr_qa_baseline/model_best.pth

python inference.py --config-name inference_vcr \
  datasets=vcr_qar \
  inferencer.save_path=saved/vcr_qar_preds \
  inferencer.from_pretrained=saved/vcr_qar_baseline/model_best.pth

python scripts/eval_vcr_qar.py \
  --qa-dir  saved/vcr_qa_preds/val \
  --qar-dir saved/vcr_qar_preds/val \
  --output  saved/vcr_qar_report.json
```

## Запрещённые Claims

Нельзя утверждать:

- `paper reproduced`;
- `fully paper-faithful`;
- `ready for full reproduction`;
- real VCR baseline validated;
- real GQA baseline validated;
- Q→AR reproduced.

Разрешённая формулировка:

```text
improved paper-aligned baseline:
architecture split, loss/metrics, tests, Hydra composition, relation-aware
GQA message passing, VCR per-instance object identifiers и post-hoc Q→AR
evaluator — всё validated на demo/tiny path; real-data paper reproduction
остаётся blocked by data/artifact gaps.
```
