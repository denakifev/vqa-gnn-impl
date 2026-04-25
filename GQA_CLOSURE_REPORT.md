# GQA Closure Report

Дата: 2026-04-25.
Scope: GQA only. VCR не входит.
Статус: `GQA closed for current phase`.

## Назначение

Этот документ фиксирует, что по GQA можно перестать дорабатывать в рамках
текущей фазы, и что именно blocked внешними артефактами или underspecified
paper details. Он же явно фиксирует source-of-truth path, чтобы будущие
итерации не возвращались к выбору модели.

## Source-of-Truth Path (GQA)

| Layer | Class / config | Назначение |
|---|---|---|
| Model | `src/model/gqa_model.py::GQAVQAGNNModel` | runtime source-of-truth |
| Loss | `src/loss/gqa_loss.py::GQALoss` | hard cross-entropy |
| Metric | `src/metrics/gqa_metric.py::GQAAccuracy` | exact-match |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset` | strict mode |
| Collate | `src/datasets/gqa_collate.py::gqa_collate_fn` | |
| Optimizer | `src/optim/paper_param_groups.py::paper_param_groups` (lm_lr=2e-5, other_lr=2e-4) | |
| Scheduler | `src/lr_schedulers/paper_warmup_cosine.py::PaperWarmupCosine` | |
| Hydra config | `src/configs/baseline_gqa.yaml` | training |
| Hydra config | `src/configs/inference_gqa.yaml` | inference |

`PaperGQAModel` (`src/model/paper_vqa_gnn.py`) и
`paper_vqa_gnn_gqa.yaml` оставлены как **equation reference**, не как
runtime path. Их назначение — фиксировать paper Eqs 1–9 на уровне shape
checks; они не запускаются end-to-end под текущим `GQADataset`.

## 1. Data Path Status

| Контракт | Статус |
|---|---|
| `answer_to_idx` size = 1842 | `validated` |
| `relation_count_total` = 624 (4 specials + 2×310 predicates) | `validated` |
| visual feature shape `float32[100, 2048]` | `validated` |
| graph `node_features.shape[-1] = 600` | `validated` |
| graph attrs `d_kg=600`, `num_visual_nodes=100`, `max_kg_nodes=100`, `graph_edge_type_count=624`, `graph_mode=official_scene_graph`, `conceptnet_used=False` | `validated` |
| metadata package | `validated` |
| relation specials `__no_edge__`, `__self__`, `__question_visual__`, `__question_textual__` | `validated` |

Strict package был `validated` локально и uploaded to Kaggle.
Local payloads намеренно удалены. Локальный runtime-сheck требует
restore from private Kaggle dataset.

## 2. Runtime Path Status

| Item | Статус |
|---|---|
| `GQAVQAGNNModel` consumes `graph_edge_types` через DenseGAT relation bias | `validated` |
| `gqa_collate_fn` соответствует model batch contract | `validated` |
| `baseline_gqa.yaml` targets `GQAVQAGNNModel` | `validated` (locked by test) |
| `baseline_gqa.yaml::model.num_relations >= 624` | `validated` (locked by test) |
| `inference_gqa.yaml::model.num_relations >= 624` | `validated` (locked by test) |
| `expect_graph_edge_types=True`, `strict_graph_schema=True` defaults | `validated` |
| Fail-fast probe на d_kg / missing edge types / missing required datasets | `validated` |
| `paper_vqa_gnn_gqa.yaml` явно помечен как equation reference, не runtime | `validated` |

## 3. Paper-Equation Path Status

| Item | Статус |
|---|---|
| `PaperMultiRelationGATLayer` (Eqs 1–6) shape/structure | `validated` |
| `PaperGQAModel` head Eqs 7–9 (concat 3D → 1842) | `validated` (shape) |
| Two-subgraph batch contract (visual SG + textual SG, q-node index) | `not implemented`, `blocked by external artifact / preprocessing` |
| End-to-end run `PaperGQAModel` против реальных GQA данных | `not runnable` без новой preprocessing pipeline |

`PaperGQAModel` валидирован равенствами, но не подключен к runtime, и
сознательно не подключается в этой фазе. Полное runtime включение требует:
- preprocessing, который выдает отдельные visual SG и textual SG;
- индексы q-node для каждого subgraph;
- новый dataset / collate / model config;
- внешние артефакты, которые сейчас отсутствуют.

## 4. Training Protocol Parity Status

| Item | Статус |
|---|---|
| AdamW + two-LR groups (lm_lr=2e-5, other_lr=2e-4) | `validated` |
| Linear warmup + cosine decay to 0 (`PaperWarmupCosine`) | `validated` |
| 50 epochs, weight_decay 1e-2 | `paper-aligned` (config) |
| Warmup длина для GQA | `blocked by missing paper detail` (paper не restate) |
| Batch size, dropout, max seq len, grad accumulation | `blocked by missing paper detail` |
| Number of seeds / runs | `blocked by missing paper detail` |
| Real training/eval numbers | out of scope для этой фазы |

## 5. Tests And Validation

Engineering tests (`pytest -q`): **97 passed**.

Покрытие GQA:

- `tests/test_paper_path.py` — collate, loss, metric, demo dataset,
  relation-aware DenseGATLayer, `GQAVQAGNNModel` consumes edge types,
  num_relations=0 ablation;
- `tests/test_engineering_contracts.py` — d_kg mismatch fails fast,
  missing `graph_edge_types` fails fast, opt-in для legacy artifacts,
  HDF5 schema probe, Hydra inference key contract, **runtime config
  contracts (model target, num_relations >= 624, paper-path target)** —
  добавлено в этой фазе;
- `tests/test_paper_vqa_gnn_model.py` — `PaperGQAModel` head shapes;
- `tests/test_paper_rgat.py` — `PaperMultiRelationGATLayer` Eqs 1–6;
- `tests/test_paper_scheduler_and_optim.py` — warmup-cosine, two-LR
  param groups;
- `tests/test_gqa_pipeline.py`, `tests/test_gqa_ner_pipeline.py` —
  preprocessing pipelines.

Hydra composition (`baseline_gqa`, `inference_gqa`, `paper_vqa_gnn_gqa`):
все три композятся; targets и `num_relations` совпадают с ожидаемыми.

`scripts/validate_gqa_data.py --skip-runtime-check` ранее прошёл с
`0` errors на full strict package перед удалением. Runtime check теперь
использует `GQAVQAGNNModel` (а не legacy `VQAGNNModel`), поэтому при
restore Kaggle package валидатор реально упражняет relation-aware path.

## 6. Docs Updated

- `README.md` — добавлены ссылки на `DATA_CONTRACTS.md` и
  `GQA_CLOSURE_REPORT.md`.
- `GAP_ANALYSIS.md` — GQA gaps пере-классифицированы (`blocked by
  external artifact`, `blocked by missing paper detail`); добавлены
  `PaperGQAModel`, `paper_param_groups`, `PaperWarmupCosine` как
  validated.
- `ARCHITECTURE_SPLIT.md` — два GQA path явно разделены: runtime
  source-of-truth (`GQAVQAGNNModel`) vs equation reference
  (`PaperGQAModel`).
- `AGENTS.md` — добавлены `DATA_CONTRACTS.md` и `GQA_CLOSURE_REPORT.md`,
  явно зафиксирован source-of-truth.
- `DATA_CONTRACTS.md` — создан заново; на него ссылаются runtime errors
  и configs.
- `src/configs/model/paper_vqa_gnn_gqa.yaml` — header расширен:
  equation reference, не runtime.
- `src/configs/model/vqa_gnn_gqa.yaml` — `num_relations: 624` с
  объяснением why (validated relation vocab).
- `scripts/validate_gqa_data.py` — переключен на `GQAVQAGNNModel` с
  `num_relations` из validated relation vocab.
- `tests/test_engineering_contracts.py` — добавлены тесты,
  фиксирующие GQA model config contract.

## 7. Remaining Blockers

Эти пункты **не закрываются в этой фазе**.

### 7.1. Blocked by external artifact

- Local strict GQA package — restore из private Kaggle dataset
  (если потребуется local runtime / training).
- Two-subgraph paper preprocessing для `PaperGQAModel` end-to-end.

### 7.2. Blocked by missing paper detail

- Точная длина warmup для GQA (paper приводит её только в VCR блоке).
- Batch size, dropout rate, max sequence length, gradient accumulation.
- Число random seeds / runs для усреднения.
- Equation-level эквивалентность merged-graph
  `GQAVQAGNNModel + relation bias` ↔ paper RGAT (доказательно
  невозможно без paper hyperparams и без официальной reference
  реализации).

### 7.3. Out of scope для current phase

- Real GQA training/eval numbers.
- Reproduction of paper's reported GQA accuracy.

## Финальный Вердикт

`GQA closed for current phase`.

GQA архитектура, contracts, validators, configs, source-of-truth path,
training-protocol primitives и tests доведены до согласованного
состояния. Все runtime config contracts заблокированы тестами против
регрессии. Local data restore остаётся опциональным и требуется только
для фактических тренировок.

Можно переходить к VCR.
