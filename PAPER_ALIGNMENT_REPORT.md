# Отчёт О Соответствии Статье

Дата аудита: 2026-04-20.

Документ фиксирует только то, что подтверждено кодом, конфигами и runtime
проверками. Статусные токены оставлены в исходном виде, чтобы они совпадали с
аудитными отчётами:

- `implemented`
- `validated`
- `paper-aligned`
- `paper-aligned approximation`
- `not paper-faithful yet`
- `runtime-valid`
- `not validated yet`

## Краткий Итог

Текущий baseline — `improved paper-aligned baseline`: paper-like
постановка VCR/GQA реализована, demo runtime валиден, GQA GNN теперь
использует `graph_edge_types`, VCR object refs рендерятся как
per-instance identifiers, post-hoc Q→AR evaluator runtime-validated, но
real-data path всё ещё blocked by data.

Подтверждено:

- VCR и GQA архитектурно разделены.
- VCR реализован как candidate scoring.
- VCR object references рендерятся как per-instance identifiers
  (`person0`, `person2`), что сохраняет identity при одной категории.
- GQA реализован как vocabulary classification с `num_answers=1842`.
- GQA GNN потребляет `graph_edge_types` через learned per-(head, relation)
  attention-bias в каждом dense GAT слое (relation-aware message passing).
- Loss/metrics для VCR и GQA проходят тесты.
- Post-hoc Q→AR evaluator (`scripts/eval_vcr_qar.py`) runtime-validated
  по парным prediction files с `sample_id` join.
- Demo smoke-run проходит для VCR Q→A, VCR QA→R и GQA.

Не подтверждено:

- paper numbers (`future work`);
- real VCR run (`blocked by data`);
- local real GQA run на текущих `data/gqa` artifacts (`blocked by data`);
- end-to-end paper-number сравнение relation-aware GQA (`future work`).

## Уже Согласовано Со Статьёй

### VCR

| Компонент | Статус | Доказательство |
|---|---|---|
| Q→A как 4-way candidate scoring | `implemented`, `validated`, `paper-aligned` | `VCRDemoDataset`, `make_vcr_collate_fn`, `VCRVQAGNNModel`, `VCRLoss`; shape audit: logits `[8, 1]`, labels `[2]` |
| QA→R как отдельный прогон | `implemented`, `validated`, `paper-aligned` | `baseline_vcr_qar.yaml`, `task_mode=qar`, `VCRQARAccuracy` |
| Loss по четырём кандидатам | `implemented`, `validated`, `paper-aligned` | `src/loss/vcr_loss.py`, tests in `tests/test_paper_path.py` |
| Q→A / QA→R metrics | `implemented`, `validated`, `paper-aligned` | `VCRQAAccuracy`, `VCRQARAccuracy`, targeted tests pass |
| Q→AR joint formula | `implemented`, `validated`, `paper-aligned` | `VCRQARJointAccuracy` tested on toy predictions |
| Post-hoc end-to-end Q→AR evaluator | `implemented`, `validated`, `paper-aligned` | `scripts/eval_vcr_qar.py` + `tests/test_eval_vcr_qar.py`; inferencer persists `sample_id` for stable join |
| VCR per-instance object identifiers | `implemented`, `validated`, `paper-aligned approximation` | `_vcr_object_mention`; two "person" mentions rendered as `person0`/`person2` so text encoder sees identity |

### GQA

| Компонент | Статус | Доказательство |
|---|---|---|
| Hard answer classification | `implemented`, `validated`, `paper-aligned` | `GQAVQAGNNModel`, `GQALoss`, `GQAAccuracy` |
| Answer space size | `implemented`, `validated`, `paper-aligned` | `vqa_gnn_gqa.yaml` задаёт `num_answers: 1842`; local vocab содержит 1842 entries |
| Exact-match accuracy | `implemented`, `validated`, `paper-aligned` | `src/metrics/gqa_metric.py`, tests pass |
| Official scene-graph preprocessing path in code | `implemented`, `validated`, `paper-aligned` на уровне preprocessing-кода | `scripts/prepare_gqa_data.py`, `src/datasets/gqa_preprocessing.py`; tests validate relation vocab и scene-graph edge types |
| Relation-aware GNN message passing | `implemented`, `validated`, `paper-aligned approximation` | `DenseGATLayer(num_relations>0)` добавляет learned per-(head, relation) bias в attention logits; `GQAVQAGNNModel.forward` передаёт `graph_edge_types` во все слои; новые тесты `TestRelationAwareDenseGAT`, `TestGQAModelConsumesEdgeTypes` |

### Shared Core

| Компонент | Статус | Доказательство |
|---|---|---|
| RoBERTa-style HF text encoder | `implemented`, smoke `validated`, `paper-aligned approximation` | `HFQuestionEncoder`; smoke использовал local tiny RoBERTa; default config использует `roberta-large` |
| Три GNN layers по config | `implemented`, `validated`, `paper-aligned` | `num_gnn_layers: 3` в VCR/GQA configs |
| Только переиспользуемые primitives в shared core | `implemented`, `validated` | `DenseGATLayer`, `HFQuestionEncoder` находятся в `src/model/gnn_core.py` |

## Согласовано Со Статьёй, Но Приблизительно

### VCR

| Компонент | Статус | Почему это approximation |
|---|---|---|
| Object references | `paper-aligned approximation` | Per-instance identifiers (`person0`/`person2`) сохраняют identity, но не связаны с region detector features |
| Visual features | `paper-aligned approximation` | Runtime ожидает pre-extracted `36×2048` style features; точный paper detector/path не verified |
| KG construction | `paper-aligned approximation` | Offline HDF5 graphs вместо on-the-fly ConceptNet querying |
| Entity extraction | `paper-aligned approximation` | Tokenization/string matching вместо подтверждённого paper entity linker |
| Q→AR evaluation regime | `paper-aligned approximation` | QA→R training использует GT answer (стандарт VCR); Q→AR теперь считается post-hoc через паринг сохранённых Q→A и QA→R predictions, но точный paper test regime не проверен численно |

### GQA

| Компонент | Статус | Почему это approximation |
|---|---|---|
| Scene-graph preprocessing code | `paper-aligned approximation` | Использует official scene graphs, GloVe object/attribute features и bbox alignment, но точный paper code недоступен |
| Answer vocabulary construction | `paper-aligned approximation` | Deterministic 1842-class vocab из released balanced splits, а не official paper artifact |
| Offline graph artifacts | `paper-aligned approximation` | Runtime читает HDF5 artifacts, а не строит graph динамически |
| Relation-aware attention bias | `paper-aligned approximation` | Learned per-(head, relation) scalar bias — минимальная paper-faithful инъекция типизированных edges; эквивалентность с paper GNN формой не verified численно |

### Shared Core

| Компонент | Статус | Почему это approximation |
|---|---|---|
| Dense GAT | `paper-aligned approximation` | Статья не полностью специфицирует message passing; dense GAT не verified численно против paper |
| Pooling/fusion | `paper-aligned approximation` | Attention pooling plausible и runtime-valid, но exact paper fusion details не independently verified |

## Пока Не Полностью Соответствует Статье

### VCR

| Gap | Статус | Impact |
|---|---|---|
| Real VCR data отсутствуют локально | `blocked by data` | Блокирует первый local VCR real run |
| Region grounding (mentions → detector features) | `not paper-faithful yet` | Per-instance identifiers снимают ambiguity, но не связаны с regions напрямую |
| Paper train protocol details | `not paper-faithful yet` | Separate Q→A/QA→R configs присутствуют; точное совпадение с paper training regime (LR warmup, scheduler, gradient accumulation, max_context_len) не доказано |

### GQA

| Gap | Статус | Impact |
|---|---|---|
| Local `data/gqa` graph artifacts mismatch current config | `blocked by data` | Блокирует первый local GQA real run: stored node features `(N, 300)`, config expects `d_kg=600` |
| Local `data/gqa` graph artifacts без `graph_edge_types` и attrs | `blocked by data` | Указывает на старый non-paper-like artifact path; надо пересобрать `scripts/prepare_gqa_data.py knowledge-graphs` |
| `num_relations` config aligned с real relation vocab size | `blocked by data` | Default `512` — безопасный upper bound; после пересборки graphs нужно выставить фактическое `relation_count_total` из `gqa_relation_vocab.json` |
| Real GQA baseline numbers отсутствуют | `future work` | Блокирует paper reproduction claims |

### Shared Core

| Gap | Статус | Impact |
|---|---|---|
| Exact GNN/message passing from paper unknown | `not paper-faithful yet` | Блокирует exact reproduction claims |
| Exact RoBERTa checkpoint/training details не independently verified | `paper-aligned approximation` | Non-blocking для первого baseline, blocking для strict reproduction |

## Открытые Gaps Перед Реальным Воспроизведением

`blocked by data`:

- VCR: подготовить `data/vcr/train.jsonl`, `data/vcr/val.jsonl`, visual HDF5 и KG HDF5.
- GQA local: пересобрать `data/gqa/knowledge_graphs/*.h5` текущим scene-graph
  pipeline (`d_kg=600`, `graph_edge_types`, attrs) или явно override
  model/dataset `d_kg=300` для старого artifact run.
- Добавить или задокументировать `pytest` как test dependency; в текущем
  `.venv` он есть, но в `requirements.txt` не указан.

`future work`:

- запустить реальные VCR Q→A, VCR QA→R и GQA experiments;
- выполнить Q→AR через `scripts/eval_vcr_qar.py` по парным saved predictions;
- сравнить numbers со статьёй с documented deviations;
- свести `num_relations` к фактическому `relation_count_total` и протестировать relation-aware GQA на реальных scene-graph edges;
- подтвердить или адаптировать VCR region-grounded object references.

`non-blocking engineering deviation`:

- dense adjacency и dense GAT для малых graph;
- offline graph preprocessing;
- Comet offline warnings и Matplotlib cache warnings в локальной среде.
