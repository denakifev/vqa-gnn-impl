# Анализ Разрывов

Дата аудита: 2026-04-25.

Цель: фиксировать оставшиеся reproducibility gaps без завышения paper
faithfulness.

## Текущий Статус

Статус репозитория: `partially validated baseline`.

Validated или implemented:

- VCR/GQA architecture split: `implemented`, `validated`.
- GQA 1842-way classification: `implemented`, `validated`.
- GQA relation-aware message passing с `graph_edge_types`: `implemented`,
  `validated`, `paper-aligned approximation`. Runtime model
  `num_relations=624` совпадает с validated relation vocab пакета.
- Strict GQA preprocessing package: `validated`, uploaded to Kaggle, затем
  validated end-to-end against real Kaggle data after restore.
- GQA paper-equation path (`PaperGQAModel`, `PaperMultiRelationGATLayer`):
  `implemented`, `validated` end-to-end via
  `GQADataset(emit_two_subgraphs=True)` + `gqa_paper_collate_fn`. Slice
  out of merged HDF5 statically lossless (no visual↔textual cross-edges
  by construction). Configs `paper_vqa_gnn_gqa.yaml` и
  `inference_paper_gqa.yaml` runnable.
- GQA training-protocol primitives (`paper_param_groups`,
  `PaperWarmupCosine`): `implemented`, `validated`.
- VCR candidate scoring для Q->A и QA->R: `implemented`, `validated`.
- VCR post-hoc Q->AR evaluator: `implemented`, `validated`.
- Demo и engineering tests: `runtime-valid`.

Пока не завершено:

- Full licensed VCR data package: `not validated yet`.
- Real training numbers для GQA и VCR: `not validated yet`.
- Paper reproduction по reported metrics не выполнен.

## Состояние GQA Data

GQA data layer больше не blocked by legacy local artifacts. Strict paper-target
package был собран, uploaded to Kaggle и затем validated end-to-end against
real Kaggle data after restore.

Validated свойства strict package:

| Contract | Статус |
|---|---|
| `answer_to_idx` существует | `validated` |
| answer vocab size равен `1842`, contiguous, coverage 100% | `validated` |
| relation vocab существует | `validated` |
| relation count равен `624` (310 predicates + 4 specials), contiguous | `validated` |
| train split: 943,000 questions, 72,140 images, full image coverage | `validated` |
| val split: 132,062 questions, 10,234 images, full image coverage | `validated` |
| visual features имеют форму `float32[100, 2048]`, attrs `num_boxes=100`, `feature_dim=2048` | `validated` |
| graph datasets включают `graph_edge_types` | `validated` |
| graph `node_features.shape[-1] == 600` | `validated` |
| graph attrs включают `d_kg=600`, `num_visual_nodes=100`, `max_kg_nodes=100`, `graph_edge_type_count=624` | `validated` |
| `graph_mode=official_scene_graph`, `conceptnet_used=False`, `fully_connected_fallback_used=False` | `validated` |
| sampled graphs schema-consistent, без fully-connected visual fallback | `validated` |
| metadata package существует и проходит validator; no-match rate ~9.5% train / ~9.6% val | `validated` |
| runtime path `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy` | `validated` |

Состояние хранения:

- uploaded to Kaggle из `kaggle_staging/gqa-gnn-data`;
- local full `data/gqa`, `data/raw/gqa`, GQA downloads, GQA staging и GQA
  mini/debug payloads были намеренно удалены;
- local GQA runs требуют сначала восстановить private Kaggle package.

Runtime validation sanity:

- logits shape `(1, 1842)`;
- finite loss `7.326373`, close to `ln(1842)=7.519` for random-init model;
- metric finite;
- Roberta load warnings about `lm_head.*` and `pooler.dense.*` are expected:
  `HFQuestionEncoder` uses CLS from `last_hidden_state`, not LM head/pooler.

Оставшиеся GQA gaps:

| Gap | Статус | Классификация |
|---|---|---|
| Real GQA training/eval numbers | `not validated yet` | `future work` (paper numbers, ожидает обучения) |
| Exact paper warmup length для GQA, batch size, dropout, max seq len, grad accumulation, seed N-runs | `paper-aligned approximation` | `blocked by missing paper detail` |
| Hidden widths / activations внутри `f_r`, `f_h` MLPs, число attention heads (Eq 5 со scalar gamma) | `paper-aligned approximation` | `blocked by missing paper detail` |
| Visual SG vs textual SG: общий или раздельные relation vocabs | `paper-aligned approximation` | `blocked by missing paper detail` (используем общий 624) |
| Entity linking via tokenization vs NER | `paper-aligned approximation` | `blocked by missing paper detail` |
| Exact RoBERTa-large checkpoint provenance | `paper-aligned approximation` | `blocked by missing paper detail` |
| Local availability full GQA package | optional / may be absent | restore from Kaggle when needed |

Все engineering-gaps по GQA закрыты. Оставшееся — paper underspec и
реальные числа обучения. См. `GQA_CLOSURE_REPORT.md`.

## VCR Gaps

| Gap | Статус | Классификация |
|---|---|---|
| Licensed VCR annotations/features локально отсутствуют | `not validated yet` | `blocked by data` |
| VCR visual features assumed as `36x2048` pre-extracted regions | `paper-aligned approximation` | data preparation requirement |
| VCR KG graph construction является question-level | `paper-aligned approximation` | possible paper deviation |
| Object references являются per-instance text identifiers, а не detector-region binding | `paper-aligned approximation` | future exactness work |
| Real VCR Q->A и QA->R runs | `not validated yet` | `blocked by data` |
| Real Q->AR score по paired predictions | `not validated yet` | requires real predictions |

Baseline-level VCR assets, которые ещё нужно подготовить:

```text
data/vcr/train.jsonl
data/vcr/val.jsonl
data/vcr/visual_features/train_features.h5
data/vcr/visual_features/val_features.h5
data/vcr/knowledge_graphs/train_graphs.h5
data/vcr/knowledge_graphs/val_graphs.h5
```

VCR data должны оставаться private на Kaggle.

## Gaps Общей Модели

| Gap | Статус | Классификация |
|---|---|---|
| Dense GAT vs exact paper GNN | `paper-aligned approximation` | `future work` |
| Relation-aware bias vs paper equation-level semantics | `paper-aligned approximation` | `future work` |
| RoBERTa-large exact checkpoint provenance | `paper-aligned approximation` | `future work` |
| Offline graph HDF5 вместо on-the-fly graph construction | `paper-aligned approximation` | engineering deviation |

## Gaps Протокола Обучения

| Gap | Статус |
|---|---|
| GQA real training numbers | `not validated yet` |
| VCR Q->A real training numbers | `not validated yet` |
| VCR QA->R real training numbers | `not validated yet` |
| VCR Q->AR real joint metric | `not validated yet` |
| Full paper hyperparameter parity | `paper-aligned approximation` |

## Разрешённые И Запрещённые Claims

Разрешено:

- `partially validated baseline`
- `implemented`
- `validated`
- `runtime-valid`
- `paper-aligned approximation`
- `not paper-faithful yet`
- `not validated yet`

Запрещено:

- `paper reproduced`
- `fully paper-faithful`
- `ready for full reproduction`
- `GQA real baseline validated`
- `VCR real baseline validated`
- `Q->AR reproduced`

Безопасная формулировка:

```text
GQA strict data package validated end-to-end against real Kaggle data, но real
paper numbers not validated. VCR data preparation остаётся открытой.
Репозиторий является partially validated baseline, а не paper reproduction.
```
