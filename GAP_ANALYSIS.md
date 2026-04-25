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
- Strict GQA preprocessing package: `validated`, uploaded to Kaggle, removed
  locally после upload.
- GQA paper-equation reference path (`PaperGQAModel`,
  `PaperMultiRelationGATLayer`): `implemented`, equation-shape `validated`.
  Runtime запуск blocked by two-subgraph batch contract — `not runtime-ready`.
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
package был собран и validated перед cleanup.

Validated свойства strict package:

| Contract | Статус |
|---|---|
| `answer_to_idx` существует | `validated` |
| answer vocab size равен `1842` | `validated` |
| relation vocab существует | `validated` |
| relation count равен `624` | `validated` |
| visual features имеют форму `float32[100, 2048]` | `validated` |
| graph datasets включают `graph_edge_types` | `validated` |
| graph `node_features.shape[-1] == 600` | `validated` |
| graph attrs включают `d_kg=600`, `graph_edge_type_count=624` | `validated` |
| metadata package существует и проходит validator | `validated` |

Состояние хранения:

- uploaded to Kaggle из `kaggle_staging/gqa-gnn-data`;
- local full `data/gqa`, `data/raw/gqa`, GQA downloads, GQA staging и GQA
  mini/debug payloads были намеренно удалены;
- local GQA runs требуют сначала восстановить private Kaggle package.

Оставшиеся GQA gaps:

| Gap | Статус | Классификация |
|---|---|---|
| Real GQA training/eval numbers | `not validated yet` | `future work`, out of scope this phase |
| Two-subgraph paper batch contract (visual SG + textual SG, q-node index) для `PaperGQAModel` | `paper-aligned approximation` | `blocked by external artifact / preprocessing` |
| Exact paper optimizer/schedule parity (warmup length для GQA, batch size, dropout, seed N-runs) | `paper-aligned approximation` | `blocked by missing paper detail` |
| Local availability full GQA package | intentionally absent | restore from Kaggle when needed |

GQA path является `closed for current phase`. См. `GQA_CLOSURE_REPORT.md`.

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
GQA strict data package validated и uploaded to Kaggle, но real paper numbers
not validated. VCR data preparation остаётся открытой. Репозиторий является
partially validated baseline, а не paper reproduction.
```
