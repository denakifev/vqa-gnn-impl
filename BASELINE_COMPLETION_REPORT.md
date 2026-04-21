# Baseline Completion Report

Дата: 2026-04-20.
Цель документа: зафиксировать результаты прохода baseline completion перед
реальными экспериментами. Документ фиксирует только изменения, тесты и
конфигурационные артефакты — не paper numbers.

## Финальный Вердикт

`improved paper-aligned baseline`.

Разрешённая формулировка:

```text
improved paper-aligned baseline: architecture split, loss/metrics, tests,
Hydra composition, relation-aware GQA message passing, VCR per-instance
object identifiers и post-hoc Q→AR evaluator — всё validated на demo/tiny
path; real-data paper reproduction остаётся blocked by data/artifact gaps.
```

## Scope Прохода

Задача: минимальная paper-faithful доработка baseline без экспериментов и
без погони за paper numbers. Работа велась по четырём направлениям:

- BLOCK A — GQA: relation-aware GNN + alignment configs.
- BLOCK B — VCR: per-instance object identifiers, Q→AR evaluator,
  protocol audit.
- BLOCK C — аудит paper-faithfulness: какие gaps закрыты, какие остались.
- BLOCK D — validation pass (tests + Hydra composition).

Не в scope:

- запуск реальных VCR/GQA experiments;
- сравнение с paper numbers;
- редизайн architecture;
- любые изменения, которые требуют новых артефактов данных.

## Что Сделано

### GQA Relation-Aware Message Passing

| Изменение | Файлы | Статус |
|---|---|---|
| `DenseGATLayer(num_relations=K)` принимает optional `edge_types` и добавляет learned per-(head, relation) scalar bias к attention logits pre-softmax | `src/model/gnn_core.py` | `implemented`, `validated` |
| `GQAVQAGNNModel.forward` прокидывает `graph_edge_types` во все GNN слои; fail-fast если `num_relations > 0`, но `graph_edge_types` отсутствует | `src/model/gqa_model.py` | `implemented`, `validated` |
| `num_relations: 512` в GQA config с explanatory comment | `src/configs/model/vqa_gnn_gqa.yaml` | `implemented` |
| `graph_edge_types` добавлен в `inferencer.device_tensors` | `src/configs/inference_gqa.yaml` | `implemented` |

Bias инициализирован нулём: training стартует бит-в-бит как untyped
baseline, relation-aware поведение появляется только по мере обучения
relation_bias. Backward compat: `num_relations=0` — no-op для VCR.

Численная эквивалентность с paper GNN form не verified →
`paper-aligned approximation`.

### VCR Per-Instance Object Identifiers

| Изменение | Файлы | Статус |
|---|---|---|
| `_vcr_object_mention(index, objects)` рендерит `person0`/`person2`; sanitizes non-alphanumeric chars | `src/datasets/vcr_dataset.py` | `implemented`, `validated` |
| `_vcr_tokens_to_text` переписан на use `_vcr_object_mention` | `src/datasets/vcr_dataset.py` | `implemented`, `validated` |

Два "person" в одном example теперь рендерятся как `person0` и
`person2` — text encoder видит identity. Это не true region grounding
(features детектора не подключаются к mentions), поэтому
`paper-aligned approximation`.

### Post-Hoc Q→AR Evaluator

| Изменение | Файлы | Статус |
|---|---|---|
| CLI evaluator: загружает per-sample `.pth` предсказания, keys by `sample_id`, joins Q→A и QA→R, считает Q→A/QA→R/joint accuracy | `scripts/eval_vcr_qar.py` | `implemented`, `validated` |
| Inferencer persists `sample_id` (VCR `annot_id`/GQA `question_id`) рядом с `pred_label`/`label` | `src/trainer/inferencer.py` | `implemented`, `validated` |
| Tests: happy path, disagreement failure, `--allow-missing`, missing `sample_id`, duplicate `sample_id`, JSON round-trip CLI | `tests/test_eval_vcr_qar.py` | `implemented`, `validated` |

Evaluator — pure post-hoc: не загружает model, не читает dataset, не
требует tokenizer. Safe для CPU-only машины после GPU training.

## Тесты

Baseline до прохода: `56 passed`.
Baseline после прохода: `81 passed`.

| Test set | Состав | Статус |
|---|---|---|
| `tests/test_paper_path.py` | 32 baseline + 14 новых (`TestRelationAwareDenseGAT`, `TestGQAModelConsumesEdgeTypes`, `TestVCRInstanceIdentifiers`) | 46 passed |
| `tests/test_eval_vcr_qar.py` | 11 новых (`TestLoadPredictions`, `TestComputeJointReport`, `TestCLI`) | 11 passed |
| `tests/test_gqa_pipeline.py` | без изменений | 5 passed |
| `tests/test_gqa_ner_pipeline.py` | без изменений | 4 passed |
| `tests/test_engineering_contracts.py` | без изменений | 15 passed |
| прочие | без изменений | остальные tests в full suite |
| **Full suite** | `.venv/bin/python -m pytest -q` | **81 passed** |

Hydra composition:

```bash
.venv/bin/python train.py --config-name baseline_vcr_qa --cfg job
.venv/bin/python train.py --config-name baseline_vcr_qar --cfg job
.venv/bin/python train.py --config-name baseline_gqa --cfg job
.venv/bin/python inference.py --config-name inference_vcr inferencer.skip_model_load=True --cfg job
.venv/bin/python inference.py --config-name inference_gqa inferencer.skip_model_load=True --cfg job
```

Все пять команд собрали config успешно.

## Обновлённая Документация

- [README.md](README.md) — status → `improved paper-aligned baseline`, test count 56 → 81, Q→AR evaluation section.
- [PAPER_ALIGNMENT_REPORT.md](PAPER_ALIGNMENT_REPORT.md) — relation-aware GQA, per-instance identifiers и Q→AR evaluator переведены в `implemented/validated`.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — gaps reclassified через новую token-систему `blocked by data` / `future work`.
- [VALIDATION_REPORT.md](VALIDATION_REPORT.md) — финальный вердикт обновлён, test count и blockers переведены на новый токен-set.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — добавлена секция `Relation-Aware GNN Contract (GQA)` с требованиями к `num_relations` и `graph_edge_types`; batch-keys секция отмечает persisting `sample_id` для Q→AR join.

## Оставшиеся Gaps

`blocked by data`:

- реальные VCR данные (`data/vcr/*.jsonl`, visual HDF5, KG HDF5);
- GQA graph artifacts с `d_kg=600`, `graph_edge_types`, attrs —
  требуется пересборка через `scripts/prepare_gqa_data.py`;
- `num_relations` → фактическое `relation_count_total` из
  `gqa_relation_vocab.json` после пересборки graphs.

`future work`:

- запустить реальные VCR Q→A, VCR QA→R, GQA experiments;
- выполнить post-hoc Q→AR evaluator на paired saved predictions;
- сравнить numbers со статьёй с documented deviations;
- verify numerical equivalence relation-aware attention bias с paper
  GNN form;
- VCR region grounding — связать per-instance mentions с detector
  features.

`non-blocking engineering deviation`:

- dense GAT и offline HDF5 preprocessing;
- Comet offline warnings и Matplotlib cache warnings.

## Запрещённые Claims

Нельзя утверждать:

- `paper reproduced`;
- `fully paper-faithful`;
- `ready for full reproduction`;
- real VCR baseline validated;
- real GQA baseline validated;
- Q→AR reproduced.

Разрешено только: `improved paper-aligned baseline` — в формулировке
из начала документа.
