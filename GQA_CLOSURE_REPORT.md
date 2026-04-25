# GQA Closure Report

Дата: 2026-04-25.
Scope: GQA only. VCR не входит.
Статус: `GQA closed end-to-end (engineering)`.

Последнее обновление: closed two-subgraph batch contract для
`PaperGQAModel`. `GQADataset(emit_two_subgraphs=True)` слайсит
существующий strict merged-graph HDF5 в paper-batch. HDF5 пересобирать
не нужно — slice статически lossless. Все engineering items по GQA
закрыты; остаются только paper underspec (зафиксированы как
`paper-aligned approximation`) и numeric reproduction (ожидает обучения).

## Назначение

Этот документ фиксирует, что по GQA можно перестать дорабатывать в рамках
текущей фазы, и что именно blocked внешними артефактами или underspecified
paper details. Он же явно фиксирует source-of-truth path, чтобы будущие
итерации не возвращались к выбору модели.

## Two Runnable GQA Paths

После закрытия two-subgraph contract в репозитории два полностью
runnable GQA path. Оба читают **один и тот же strict GQA package**.

### A. Merged-graph runtime path (default fast-path)

| Layer | Class / config |
|---|---|
| Model | `src/model/gqa_model.py::GQAVQAGNNModel` |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset` (default `emit_two_subgraphs=False`) |
| Collate | `src/datasets/gqa_collate.py::gqa_collate_fn` |
| Hydra train | `src/configs/baseline_gqa.yaml` |
| Hydra inference | `src/configs/inference_gqa.yaml` |

Status: `paper-aligned approximation` (DenseGAT + per-(head, relation)
attention bias). Быстрее, проще; используется как baseline.

### B. Paper-equation runtime path

| Layer | Class / config |
|---|---|
| Model | `src/model/paper_vqa_gnn.py::PaperGQAModel` |
| RGAT layer | `src/model/paper_rgat.py::PaperMultiRelationGATLayer` (paper Eqs 1–6) |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset(emit_two_subgraphs=True)` |
| Collate | `src/datasets/gqa_collate.py::gqa_paper_collate_fn` |
| Hydra train | `src/configs/paper_vqa_gnn_gqa.yaml` |
| Hydra inference | `src/configs/inference_paper_gqa.yaml` |

Status: paper Eqs 1–9 structurally implemented in the most literal available
reading; medium-speed. Детали, которые статья не задаёт явно, остаются
`paper-aligned approximation`. Использует существующий merged-graph HDF5
через runtime slicer.

### Shared layers

| Layer | Class / config |
|---|---|
| Loss | `src/loss/gqa_loss.py::GQALoss` (hard cross-entropy) |
| Metric | `src/metrics/gqa_metric.py::GQAAccuracy` (exact-match) |
| Optimizer | `src/optim/paper_param_groups.py::paper_param_groups` (lm_lr=2e-5, other_lr=2e-4) |
| Scheduler | `src/lr_schedulers/paper_warmup_cosine.py::PaperWarmupCosine` |

## 1. Data Path Status

| Контракт | Статус |
|---|---|
| `answer_to_idx` size = 1842, contiguous, coverage 100% | `validated` |
| `relation_count_total` = 624 (4 specials + 2×310 predicates), contiguous | `validated` |
| train questions/images: 943,000 questions, 72,140 images, all imageIds covered | `validated` |
| val questions/images: 132,062 questions, 10,234 images, all imageIds covered | `validated` |
| visual feature shape `float32[100, 2048]`, attrs `num_boxes=100`, `feature_dim=2048` | `validated` |
| graph `node_features.shape[-1] = 600` | `validated` |
| graph attrs `d_kg=600`, `num_visual_nodes=100`, `max_kg_nodes=100`, `graph_edge_type_count=624`, `graph_mode=official_scene_graph`, `conceptnet_used=False`, `fully_connected_fallback_used=False` | `validated` |
| sampled graph schema: `node_features`, `adj_matrix`, `node_types`, `graph_edge_types`; no fully-connected visual fallback | `validated` |
| metadata package, no-match rate ~9.5% train / ~9.6% val | `validated` |
| relation specials `__no_edge__`, `__self__`, `__question_visual__`, `__question_textual__` | `validated` |

Strict package был `validated` локально, uploaded to Kaggle, затем повторно
validated end-to-end against real Kaggle data после restore. Local payloads
могут быть удалены между фазами; при необходимости их нужно восстановить из
private Kaggle dataset.

## 2. Runtime Path Status

| Item | Статус |
|---|---|
| `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy` | `validated` |
| `GQAVQAGNNModel` consumes `graph_edge_types` через DenseGAT relation bias | `validated` |
| `gqa_collate_fn` соответствует model batch contract | `validated` |
| `baseline_gqa.yaml` targets `GQAVQAGNNModel` | `validated` (locked by test) |
| `baseline_gqa.yaml::model.num_relations >= 624` | `validated` (locked by test) |
| `inference_gqa.yaml::model.num_relations >= 624` | `validated` (locked by test) |
| `expect_graph_edge_types=True`, `strict_graph_schema=True` defaults | `validated` |
| Fail-fast probe на d_kg / missing edge types / missing required datasets | `validated` |
| `paper_vqa_gnn_gqa.yaml` targets runnable `PaperGQAModel` two-subgraph path | `validated` |

Runtime validation result against restored real Kaggle data:

- logits shape: `(1, 1842)`;
- loss finite: `7.326373`;
- metric finite;
- sanity check: random-init loss close to `ln(1842) = 7.519`.

HF load note: `RobertaModel` reports `lm_head.*` unexpected and
`pooler.dense.*` missing. This is expected for `HFQuestionEncoder`: it does
not use LM head or pooler and reads CLS from `last_hidden_state` directly.

Shape note: `avg_total_nodes ~120` while padded contract is `201`
(`100 visual + 1 question + 100 textual`). This confirms padding is required
and fits the configured graph size.

## 3. Paper-Equation Path Status

| Item | Статус |
|---|---|
| `PaperMultiRelationGATLayer` (Eqs 1–6) shape/structure | `validated` |
| `PaperGQAModel` head Eqs 7–9 (concat 3D → 1842) | `validated` (shape) |
| Two-subgraph batch contract (visual SG + q, textual SG + q) | `implemented`, `validated` через `GQADataset(emit_two_subgraphs=True)` + `gqa_paper_collate_fn` |
| Slice from merged HDF5 lossless (no visual↔textual cross-edges) | `validated` (статически по коду препроцессинга + tests) |
| Hydra train config (`paper_vqa_gnn_gqa.yaml`) | `validated`, runnable |
| Hydra inference config (`inference_paper_gqa.yaml`) | `validated`, runnable |
| End-to-end forward `PaperGQAModel` через paper collate | `validated` test |

Полное runtime включение **не требует** пересборки HDF5: slicer берёт
существующие merged graph артефакты и эмиттит two-subgraph batch с q
в последнем индексе каждой subgraph. `gqa_preprocessing.py` по
конструкции пишет только visual↔visual, textual↔textual,
q↔visual и q↔textual edges, поэтому slice статически lossless.

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

Engineering tests (`pytest -q`): **111 passed**.

Покрытие GQA:

- `tests/test_paper_path.py` — collate, loss, metric, demo dataset,
  relation-aware DenseGATLayer, `GQAVQAGNNModel` consumes edge types,
  num_relations=0 ablation;
- `tests/test_engineering_contracts.py` — d_kg mismatch fails fast,
  missing `graph_edge_types` fails fast, opt-in для legacy artifacts,
  HDF5 schema probe, Hydra inference key contract, runtime config
  contracts (model target, num_relations >= 624, paper-path target);
- `tests/test_paper_vqa_gnn_model.py` — `PaperGQAModel` head shapes;
- `tests/test_paper_rgat.py` — `PaperMultiRelationGATLayer` Eqs 1–6;
- `tests/test_paper_scheduler_and_optim.py` — warmup-cosine, two-LR
  param groups;
- `tests/test_paper_gqa_two_subgraphs.py` — **slice correctness**,
  paper-batch shapes, `gqa_paper_collate_fn` stacking, no legacy-key
  leakage, end-to-end forward `PaperGQAModel` через paper collate,
  Hydra compose для `paper_vqa_gnn_gqa` и `inference_paper_gqa`,
  `num_relations_visual/textual >= 624`, `device_tensors` ⊇
  `PAPER_TENSOR_KEYS`. Добавлено в этой фазе;
- `tests/test_gqa_pipeline.py`, `tests/test_gqa_ner_pipeline.py` —
  preprocessing pipelines.

Hydra composition (`baseline_gqa`, `inference_gqa`,
`paper_vqa_gnn_gqa`, `inference_paper_gqa`): все четыре композятся;
targets, num_relations и `emit_two_subgraphs` совпадают с ожидаемыми.

`scripts/validate_gqa_data.py` validated full strict package end-to-end
against restored real Kaggle data: schema checks, metadata checks,
relation-vocab alignment, coverage checks и runtime path через
`GQAVQAGNNModel(num_relations=624)`.

## 6. Docs And Code Updated In This Closure Pass

- `src/datasets/gqa_dataset.py` — добавлен флаг `emit_two_subgraphs`
  (`GQADataset` и `GQADemoDataset`); внутренний хелпер
  `_to_two_subgraphs_item` слайсит merged graph в paper-batch.
- `src/datasets/gqa_collate.py` — добавлен `gqa_paper_collate_fn` и
  `PAPER_TENSOR_KEYS`.
- `src/configs/model/paper_vqa_gnn_gqa.yaml` — `num_relations_visual` и
  `num_relations_textual` подняты до `624` (общий relation vocab); header
  переписан как runtime path, не "equation reference, not runtime".
- `src/configs/paper_vqa_gnn_gqa.yaml` — defaults переключены на
  `gqa_paper_demo`; описан реальный train command (`datasets=gqa_paper`).
- `src/configs/datasets/gqa_paper.yaml`, `gqa_paper_demo.yaml`,
  `gqa_paper_eval.yaml` — новые dataset configs с
  `emit_two_subgraphs=True`.
- `src/configs/inference_paper_gqa.yaml` — новый inference config для
  paper path с правильным `device_tensors` списком.
- `tests/test_paper_gqa_two_subgraphs.py` — новый test file
  (slice correctness, collate, end-to-end forward, Hydra contracts).
- `DATA_CONTRACTS.md` — добавлен раздел "Two-Subgraph Batch Contract",
  paper path помечен как runnable; подняты detailed approximation notes.
- `GAP_ANALYSIS.md` — two-subgraph gap снят; перечислены оставшиеся
  paper-underspec пункты.

## 7. Remaining Items (NOT engineering)

Все engineering-gaps закрыты. Оставшиеся пункты принципиально не
закрываются без paper detail или без обучения.

### 7.1. Blocked by missing paper detail (paper-aligned approximation)

- Точная длина warmup для GQA (paper restate'ит warmup только в VCR блоке).
- Batch size, dropout rate, max sequence length, gradient accumulation.
- Число random seeds / runs для усреднения.
- Hidden widths и activations внутри `f_r`, `f_h` MLPs.
- Число attention heads (paper Eq 5 со scalar gamma).
- Visual SG vs textual SG: общий или раздельные relation vocabs (используем общий 624).
- Entity linking via tokenization vs NER.
- Exact RoBERTa-large checkpoint provenance.

Все эти пункты задокументированы как `paper-aligned approximation` в
коде (`paper_rgat.py`, `paper_vqa_gnn.py`, configs) и в
`DATA_CONTRACTS.md` → "Known Approximations vs Paper".

### 7.2. Numeric reproduction (out of scope here, ожидает обучения)

- Real GQA training/eval numbers.
- Reproduction of paper's reported GQA accuracy.

## Финальный Вердикт

`GQA closed end-to-end (engineering)`.

Закрыты:

- runtime path (`GQAVQAGNNModel` + merged graph): validated end-to-end;
- paper-equation path (`PaperGQAModel` + two-subgraph batch): validated
  end-to-end через slicer существующего HDF5;
- все contracts, validators, configs, training-protocol primitives,
  тесты против регрессии.

Не закрыты только paper underspec (приближения зафиксированы) и сами
числа обучения (это уже после closure).

Можно переходить к VCR без возврата к GQA.
