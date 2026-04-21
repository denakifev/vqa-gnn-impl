# Архитектурный Split VCR/GQA

Дата аудита: 2026-04-20.

Статус: `validated` для разделения VCR/GQA на уровне кода, Hydra composition
и smoke-runtime. Это не означает, что численные результаты статьи
воспроизведены.

## Краткий Итог

Текущий paper path больше не является одной общей моделью с разным
`num_answers`. Разделение реализовано явно:

- VCR использует `VCRVQAGNNModel` и режим `candidate scoring`.
- GQA использует `GQAVQAGNNModel` и режим `1842-way classification`.
- Общие компоненты вынесены только в `src/model/gnn_core.py`.
- `train.py`, `inference.py`, `Trainer` и `BaseTrainer` остались общим
  служебным слоем.

Старый `VQAGNNModel` сохранён только как backward-compatible class и re-export
точка для старых импортов. Paper-aligned baseline configs его не используют.

## Почему Старое Объединение Было Ложным

Старое объединение через один `VQAGNNModel` и разные значения `num_answers`
смешивало два разных output contracts:

| Контур | Правильный contract | Почему `num_answers` не решает задачу |
|---|---|---|
| VCR Q→A / QA→R | `[B*4, 1]`, один score на candidate | Это reranking четырёх вариантов, а не classification по словарю |
| GQA | `[B, 1842]`, logits по answer vocabulary | Это direct answer classification без candidate expansion |

Поэтому VCR и GQA разделены на уровне model, config, dataset, collate, loss и
metrics.

## Общее Ядро

Общее ядро находится в `src/model/gnn_core.py`.

| Компонент | Статус | Использование |
|---|---|---|
| `DenseGATLayer` | `implemented`, `validated`, `paper-aligned approximation` | VCR и GQA |
| `HFQuestionEncoder` | `implemented`, `validated`, `paper-aligned approximation` | VCR и GQA |

Общий код оправдан только на уровне primitives: text encoder и dense GAT.
Task-specific output heads, batch semantics и losses вынесены из shared layer.

## Стек Только Для VCR

| Слой | Файл / config | Статус |
|---|---|---|
| Model | `src/model/vcr_model.py::VCRVQAGNNModel` | `implemented`, `validated` |
| Model config | `src/configs/model/vqa_gnn_vcr.yaml` | `validated` |
| Dataset | `src/datasets/vcr_dataset.py::VCRDataset`, `VCRDemoDataset` | `implemented`, demo `runtime-valid` |
| Collate | `src/datasets/vcr_collate.py::make_vcr_collate_fn` | `validated` |
| Loss | `src/loss/vcr_loss.py::VCRLoss` | `validated`, `paper-aligned` |
| Metrics | `src/metrics/vcr_metric.py::VCRQAAccuracy`, `VCRQARAccuracy`, `VCRQARJointAccuracy` | `validated` |
| Training configs | `baseline_vcr_qa.yaml`, `baseline_vcr_qar.yaml` | composition `validated` |

Validated VCR contract:

- dataset item содержит четыре QA/QAR encodings;
- collate расширяет compact batch `B` в model batch `B*4`;
- model outputs `logits` shape `[B*4, 1]`;
- `VCRLoss` reshapes scores в `[B, 4]`;
- Q→A и QA→R используют отдельные top-level configs;
- Q→AR — joint metric по predictions из двух task runs, а не single default
  training metric.

## Стек Только Для GQA

| Слой | Файл / config | Статус |
|---|---|---|
| Model | `src/model/gqa_model.py::GQAVQAGNNModel` | `implemented`, `validated` |
| Model config | `src/configs/model/vqa_gnn_gqa.yaml` | `validated` |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset`, `GQADemoDataset` | `implemented`, demo `runtime-valid` |
| Collate | `src/datasets/gqa_collate.py::gqa_collate_fn` | `validated` |
| Loss | `src/loss/gqa_loss.py::GQALoss` | `validated`, `paper-aligned` |
| Metric | `src/metrics/gqa_metric.py::GQAAccuracy` | `validated` |
| Training config | `baseline_gqa.yaml` | composition `validated` |

Validated GQA contract:

- batch остаётся compact `B`, без candidate expansion;
- config задаёт `num_answers: 1842`;
- model outputs `logits` shape `[B, num_answers]`;
- `GQALoss` использует hard cross-entropy по одному answer index;
- `GQAAccuracy` использует exact match;
- `graph_edge_types` присутствует в GQA batch contract, но пока не используется
  в `GQAVQAGNNModel` message passing.

## Выбор Через Hydra

Task stack выбирается top-level Hydra config:

```bash
python train.py --config-name baseline_vcr_qa
python train.py --config-name baseline_vcr_qar
python train.py --config-name baseline_gqa
```

Composition audit через `--cfg job` подтвердил:

- `baseline_vcr_qa` выбирает `src.model.VCRVQAGNNModel`, `VCRDemoDataset`,
  `VCRLoss`, `VCRQAAccuracy`;
- `baseline_vcr_qar` выбирает `src.model.VCRVQAGNNModel`, `VCRDemoDataset`
  с `task_mode=qar`, `VCRLoss`, `VCRQARAccuracy`;
- `baseline_gqa` выбирает `src.model.GQAVQAGNNModel`, `GQADemoDataset`,
  `GQALoss`, `GQAAccuracy`.

Скрытого runtime switch внутри `VCRVQAGNNModel` или `GQAVQAGNNModel` нет.

## Обратная Совместимость

`src/model/vqa_gnn.py::VQAGNNModel` остаётся для старых imports и tests.
`src/model/__init__.py` re-exports:

- `VCRVQAGNNModel`;
- `GQAVQAGNNModel`;
- `VQAGNNModel`;
- `DenseGATLayer`;
- `HFQuestionEncoder`.

Import audit для package exports прошёл в проектном `.venv`.

## Оставшиеся Ограничения

`paper-aligned approximation`:

- dense GAT не verified как численный эквивалент paper GNN;
- `roberta-large` — closest explicit public default, но exact paper checkpoint
  не independently verified;
- graph construction — offline HDF5, не on-the-fly;
- entity linking остаётся simplified.

`not paper-faithful yet`:

- GQA `graph_edge_types` не используются в message passing;
- local `data/gqa/knowledge_graphs/*.h5` artifacts не согласованы с current GQA
  config: graph node features имеют width `300`, `graph_edge_types` отсутствует,
  HDF5 attrs пустые, а current config/model expects `d_kg=600`;
- VCR real-data path локально не runnable, потому что `data/vcr/train.jsonl`
  отсутствует;
- real end-to-end paper numbers не получены.

## Ссылки На Валидацию

Подробная command matrix и runtime results находятся в
[VALIDATION_REPORT.md](VALIDATION_REPORT.md).
