# Анализ Разрывов

Дата аудита: 2026-04-20.

Цель файла: зафиксировать реальные gaps текущего репозитория после
архитектурного split VCR/GQA. Это не roadmap улучшения модели, а audit map для
воспроизводимости.

## Краткий Итог

Текущий code baseline находится в состоянии `improved paper-aligned baseline`:

- VCR/GQA split: `implemented`, `validated`.
- GQA relation-aware message passing: `implemented`, `validated`,
  `paper-aligned approximation` (per-(head, relation) attention bias,
  zero-init → no-op на старте).
- VCR per-instance object identifiers: `implemented`, `validated`,
  `paper-aligned approximation` (disambiguates same-category refs; не
  true region grounding).
- Post-hoc Q→AR evaluator: `implemented`, `validated` (`scripts/eval_vcr_qar.py`
  + `tests/test_eval_vcr_qar.py`).
- Unit and smoke runtime: `runtime-valid` на demo/tiny audit settings.
- Real VCR run: `blocked by data` (локальных VCR данных нет).
- Local real GQA run: `blocked by data` (текущие `data/gqa` graph
  artifacts не соответствуют текущему config contract).
- Paper reproduction by numbers: `future work`.

## VCR Gaps

| Gap | Статус | Классификация | Доказательство |
|---|---|---|---|
| Нет локальных VCR real-data artifacts | `not validated yet` | `blocked by data` | `datasets=vcr_qa` и `datasets=vcr_qar` падают на `data/vcr/train.jsonl` |
| Object references: per-instance identifiers, но не region-grounded | `implemented`, `paper-aligned approximation` | `future work` | `src/datasets/vcr_dataset.py::_vcr_object_mention` рендерит `person0`/`person2`, но features региона detector'а не подключаются |
| Visual features assumed as pre-extracted `36×2048` | `paper-aligned approximation` | `future work` | `src/configs/datasets/vcr_*.yaml`, `vcr_dataset.py` |
| KG — offline HDF5 и question-level | `paper-aligned approximation` | `non-blocking engineering deviation` | `VCRDataset._load_graph` |
| Post-hoc Q→AR evaluator над paired predictions | `implemented`, `validated` | — | `scripts/eval_vcr_qar.py`, `tests/test_eval_vcr_qar.py`; inferencer persists `sample_id` |
| Exact paper train protocol не доказан | `not validated yet` | `future work` | Q→A и QA→R — separate configs; paper-number run отсутствует |

## GQA Gaps

| Gap | Статус | Классификация | Доказательство |
|---|---|---|---|
| Local graph HDF5 uses 300-d node features | `not validated yet` | `blocked by data` | `data/gqa/knowledge_graphs/train_graphs.h5` first sample `node_features (2, 300)` |
| Current config/model expects `d_kg=600` | `validated` config, local data mismatch | `blocked by data` | `src/configs/model/vqa_gnn_gqa.yaml`, `src/configs/datasets/gqa.yaml` |
| Local graph HDF5 lacks `graph_edge_types` | `not validated yet` | `blocked by data` | first HDF5 group без `graph_edge_types` |
| Local graph HDF5 attrs empty | `not validated yet` | `non-blocking engineering deviation` | HDF5 attrs `{}`; `GQADataset` probes first graph sample and fails fast |
| Relation-aware message passing | `implemented`, `validated`, `paper-aligned approximation` | — | `DenseGATLayer(num_relations>0)` добавляет learned per-(head, relation) bias в attention logits; `GQAVQAGNNModel.forward` передаёт `graph_edge_types`; tests `TestRelationAwareDenseGAT`, `TestGQAModelConsumesEdgeTypes` |
| `num_relations` config aligned с real relation vocab | `not validated yet` | `blocked by data` | default `512` — safe upper bound; точное `relation_count_total` неизвестно до пересборки graphs |
| Scene-graph preprocessing code не запущен end-to-end на full raw data | `not validated yet` | `blocked by data` | unit tests покрывают toy graph construction |
| Real GQA baseline numbers отсутствуют | `not validated yet` | `future work` | нет completed train/val metrics on real data |

Важное различие:

- `scripts/prepare_gqa_data.py` и `src/datasets/gqa_preprocessing.py` являются
  paper-like на уровне кода: official scene graphs, GloVe object/attribute
  features, bbox alignment, `d_kg=2*glove_dim`, `graph_edge_types`, attrs,
  `conceptnet_used=False`.
- Локальные runtime artifacts в `data/gqa/knowledge_graphs/` не собраны этим
  current path или были получены со старыми settings. Mismatch теперь ловится
  явно через `src/datasets/gqa_dataset.py`, а не всплывает tensor-copy crash
  mid-batch.

## Gaps Общего Ядра

| Gap | Статус | Классификация | Impact |
|---|---|---|---|
| Dense GAT вместо confirmed paper GNN | `paper-aligned approximation` | `non-blocking engineering deviation` | Valid for smoke/runtime, not exact reproduction |
| Relation-aware attention bias: equivalence с paper GNN form не verified | `paper-aligned approximation` | `future work` | Learned per-(head, relation) scalar bias — минимальная paper-faithful инъекция typed edges; числовая эквивалентность с paper не проверена |
| Text encoder default requires HF artifact | `paper-aligned approximation` | `non-blocking engineering deviation` | `roberta-large` cached locally in audit env; new env may need download |
| Pooling/fusion not numerically matched to paper | `paper-aligned approximation` | `future work` | Нет ablation/numerical validation |

## Training Protocol Gaps

| Gap | Статус | Классификация |
|---|---|---|
| No real end-to-end VCR Q→A run | `not validated yet` | `blocked by data` |
| No real end-to-end VCR QA→R run | `not validated yet` | `blocked by data` |
| No real end-to-end GQA run | `not validated yet` | `blocked by data` |
| Paired Q→AR evaluator over saved predictions | `implemented`, `validated` | `future work` для real runs (evaluator готов, не хватает saved predictions) |
| Exact LR schedule/optimizer parity with paper unknown | `paper-aligned approximation` | `non-blocking engineering deviation` |
| Default writer is Comet online | `runtime-valid` with override | `non-blocking engineering deviation` |

## Gaps Контура Данных

| Область | Статус | Классификация | Следующее действие |
|---|---|---|---|
| VCR annotations/features/KG | `not validated yet` | `blocked by data` | Подготовить `data/vcr` according to configs |
| GQA local graph artifacts | `not validated yet` | `blocked by data` | Пересобрать через `scripts/prepare_gqa_data.py prepare-all` или `knowledge-graphs ... --d-kg 600`; либо явно запустить legacy non-paper-faithful config с `d_kg=300` и `expect_graph_edge_types=False` |
| GQA relation vocab | `not validated yet` локально | `blocked by data` | Убедиться, что `gqa_relation_vocab.json` и HDF5 `graph_edge_types` существуют; выставить `model.num_relations` ≥ `relation_count_total` |
| Test dependency declaration | `not validated yet` | `non-blocking engineering deviation` | `pytest` есть в `.venv`, но отсутствует в `requirements.txt` |
| Environment path | `validated` только в `.venv` | `non-blocking engineering deviation` | Использовать `.venv/bin/python` или активировать `.venv`; system Python lacks dependencies |

## Ожидаемое Влияние На Воспроизводимость

`blocked by data`:

- VCR не может начать real training, пока не появятся data files.
- GQA local full run не может стартовать с текущими artifacts, потому что
  fail-fast validation raises:
  `ValueError: [GQA data contract] ... node_features with d_kg=300, but dataset config requests d_kg=600`.
- `num_relations` нельзя свести к фактическому `relation_count_total`
  до пересборки graph artifacts.

`future work`:

- numerical equivalence relation-aware attention bias с paper GNN form
  не verified;
- region-grounded VCR object references (per-instance identifiers
  реализованы как approximation);
- completed paper-number runs для VCR Q→A, VCR QA→R, GQA;
- post-hoc Q→AR run over paired saved predictions (evaluator и инфра
  готовы, не хватает real predictions).

`non-blocking engineering deviation`:

- dense GAT и offline HDF5 preprocessing допустимы для improved
  paper-aligned baseline, если они честно задокументированы;
- Comet offline и Matplotlib cache warnings не инвалидируют model/runtime checks;
- demo smoke results — только runtime evidence, не paper evidence.

## Запрещённые Claims

Нельзя утверждать:

- `paper reproduced`;
- `fully paper-faithful`;
- `ready for full reproduction`;
- `GQA real baseline validated`;
- `VCR real baseline validated`;
- `Q→AR reproduced`.

Разрешённая формулировка:

- `improved paper-aligned baseline`: architecture split, losses, metrics,
  relation-aware GQA message passing, VCR per-instance object identifiers,
  post-hoc Q→AR evaluator и config composition — всё validated на demo/tiny
  path; real-data reproduction остаётся `blocked by data` и `future work`.
