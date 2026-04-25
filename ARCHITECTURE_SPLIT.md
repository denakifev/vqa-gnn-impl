# Архитектурный Split

Дата аудита: 2026-04-25.

Статус: VCR/GQA split `implemented` и `validated` на уровне кода, configs,
composition и smoke-runtime. Это не означает, что численные результаты статьи
были воспроизведены.

## Краткий Итог

Репозиторий больше не трактует VCR и GQA как одну и ту же задачу с разными
значениями `num_answers`.

| Задача | Model contract | Output |
|---|---|---|
| VCR Q->A / QA->R | candidate scoring | `[B*4, 1]` |
| GQA | answer classification | `[B, 1842]` |

Общий код ограничен primitives в `src/model/gnn_core.py`. Task-specific batch
semantics, heads, losses и metrics находятся вне shared core.

## Общее Ядро

| Компонент | Статус | Используется |
|---|---|---|
| `DenseGATLayer` | `implemented`, `validated`, `paper-aligned approximation` | VCR, GQA |
| `HFQuestionEncoder` | `implemented`, `validated`, `paper-aligned approximation` | VCR, GQA |

Для GQA `DenseGATLayer` consumes typed `graph_edge_types` через learned
per-relation attention bias, если `num_relations > 0`.

## Стек VCR

| Слой | Файл / config | Статус |
|---|---|---|
| Model | `src/model/vcr_model.py::VCRVQAGNNModel` | `implemented`, `validated` |
| Dataset | `src/datasets/vcr_dataset.py::VCRDataset`, `VCRDemoDataset` | `implemented`, demo `runtime-valid` |
| Collate | `src/datasets/vcr_collate.py::make_vcr_collate_fn` | `validated` |
| Loss | `src/loss/vcr_loss.py::VCRLoss` | `validated`, `paper-aligned` |
| Metrics | `src/metrics/vcr_metric.py` | `validated` |
| Configs | `baseline_vcr_qa.yaml`, `baseline_vcr_qar.yaml` | composition `validated` |

VCR runtime contract:

- каждый sample содержит 4 candidates;
- collate расширяет compact batch `B` в model batch `B*4`;
- model возвращает один scalar score на candidate;
- loss reshapes scores в `[B, 4]`;
- Q->A и QA->R используют отдельные configs;
- Q->AR считается post-hoc по paired Q->A и QA->R predictions.

VCR data пока не подготовлены локально. Real-data VCR path остаётся
`not validated yet`, а сами данные должны обрабатываться с учётом VCR license.

## Стек GQA

GQA имеет два полностью runnable архитектурных path. Оба читают тот же
strict GQA package.

1. **Merged-graph fast path** — `GQAVQAGNNModel` с одним merged graph и
   relation-aware DenseGAT bias. `baseline_gqa.yaml`,
   `inference_gqa.yaml`. `paper-aligned approximation`.
2. **Paper-equation path** — `PaperGQAModel` с двумя subgraphs
   (visual SG + q, textual SG + q), реализующий §4.1–§4.3.
   `paper_vqa_gnn_gqa.yaml`, `inference_paper_gqa.yaml`.
   `GQADataset(emit_two_subgraphs=True)` слайсит существующий
   merged-graph HDF5 в paper-batch на runtime — пересборка HDF5 не
   нужна. Slice статически lossless (`gqa_preprocessing.py` не пишет
   visual↔textual cross-edges).

| Слой | Файл / config | Статус |
|---|---|---|
| Fast-path model | `src/model/gqa_model.py::GQAVQAGNNModel` | `implemented`, `validated`, `paper-aligned approximation` |
| Paper-path model | `src/model/paper_vqa_gnn.py::PaperGQAModel` | `implemented`, end-to-end `validated` |
| Paper RGAT layer | `src/model/paper_rgat.py::PaperMultiRelationGATLayer` | `implemented`, `validated` |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset(+emit_two_subgraphs)`, `GQADemoDataset(+emit_two_subgraphs)` | `implemented`, both modes `validated` |
| Fast-path collate | `src/datasets/gqa_collate.py::gqa_collate_fn` | `validated` |
| Paper-path collate | `src/datasets/gqa_collate.py::gqa_paper_collate_fn` | `validated` |
| Loss | `src/loss/gqa_loss.py::GQALoss` | `validated`, `paper-aligned` |
| Metric | `src/metrics/gqa_metric.py::GQAAccuracy` | `validated` |
| Optimizer param groups | `src/optim/paper_param_groups.py` | `implemented`, `validated` |
| Scheduler | `src/lr_schedulers/paper_warmup_cosine.py` | `implemented`, `validated` |
| Fast-path train config | `baseline_gqa.yaml` (`model: vqa_gnn_gqa`) | composition `validated` |
| Fast-path inference config | `inference_gqa.yaml` | composition `validated` |
| Paper-path train config | `paper_vqa_gnn_gqa.yaml` (`model: paper_vqa_gnn_gqa`) | composition `validated`, runnable |
| Paper-path inference config | `inference_paper_gqa.yaml` | composition `validated`, runnable |

GQA runtime contract:

- compact batch `B`, без candidate expansion;
- `num_answers: 1842`;
- model возвращает logits `[B, 1842]`;
- loss — hard-label cross entropy;
- metric — exact-match accuracy;
- `graph_edge_types` является частью batch contract и consumed by GQA
  relation-aware GNN path.

Full strict GQA data package был собран и validated перед локальным cleanup.
Сейчас он хранится на Kaggle, а не под локальным `data/gqa`.

## Выбор Через Hydra

Top-level configs выбирают task stack:

```bash
python train.py --config-name baseline_vcr_qa --cfg job
python train.py --config-name baseline_vcr_qar --cfg job
python train.py --config-name baseline_gqa --cfg job
```

Ожидаемые selections:

- VCR QA: `VCRVQAGNNModel`, `VCRLoss`, `VCRQAAccuracy`;
- VCR QAR: `VCRVQAGNNModel`, `VCRLoss`, `VCRQARAccuracy`;
- GQA: `GQAVQAGNNModel`, `GQALoss`, `GQAAccuracy`.

Скрытого task switch внутри `VCRVQAGNNModel` или `GQAVQAGNNModel` нет.

## Обратная Совместимость

`src/model/vqa_gnn.py::VQAGNNModel` остаётся для старых imports и tests.
Paper-aligned configs используют `VCRVQAGNNModel` или `GQAVQAGNNModel`
напрямую.

`src/model/__init__.py` exports:

- `VCRVQAGNNModel`
- `GQAVQAGNNModel`
- `VQAGNNModel`
- `DenseGATLayer`
- `HFQuestionEncoder`

## Архитектурные Ограничения

`paper-aligned approximation`:

- dense GAT не доказан как численно эквивалентный paper GNN;
- relation-aware GQA attention bias — реализованный typed-edge mechanism, но не
  independently verified equation-level match;
- RoBERTa-large — closest explicit public default, но provenance exact paper
  text encoder не independently verified;
- graph construction выполняется offline в HDF5, а не on-the-fly.

`not paper-faithful yet`:

- VCR object references рендерятся как per-instance text identifiers, но не
  привязаны напрямую к detector regions;
- VCR KG construction является question-level и может не совпадать с
  candidate-aware graph construction, если статья использовала такой вариант;
- real paper-number runs не завершены.
