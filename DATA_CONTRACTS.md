# Data Contracts

Дата актуализации: 2026-04-20.

Цель: зафиксировать runtime contracts для real-data paths. Документ описывает,
какие файлы, HDF5 datasets, batch keys и размерности считаются валидными для
текущего `improved paper-aligned baseline`.

Статус: contracts `implemented`, fail-fast checks `validated` через
`tests/test_engineering_contracts.py`.

## Общие Правила

- Demo datasets (`VCRDemoDataset`, `GQADemoDataset`) нужны только для smoke/CI.
  Они `runtime-valid`, но не являются paper evidence.
- Real datasets (`VCRDataset`, `GQADataset`) читают JSON/JSONL + HDF5 и
  проверяют shape/schema на первом HDF5 open.
- Ошибка contract должна быть исправлена через пересборку данных или явный
  legacy override. Silent fallback для paper-like запуска недопустим.
- `strict_graph_schema=False` допустим только как debugging escape hatch; такой
  запуск нельзя описывать как `paper-aligned`.

## VCR Contract

### Обязательные Файлы

```text
data/vcr/
├── train.jsonl
├── val.jsonl
├── visual_features/
│   ├── train_features.h5
│   └── val_features.h5
└── knowledge_graphs/
    ├── train_graphs.h5
    └── val_graphs.h5
```

### Annotation JSONL

Каждая строка должна содержать:

- `annot_id`;
- `img_fn`;
- `question`;
- `answer_choices` с 4 candidates;
- `answer_label` для train/val;
- `rationale_choices` с 4 candidates для QA→R;
- `rationale_label` для train/val QA→R;
- optional `objects`.

### Visual HDF5

Ключ: `img_fn`.

Ожидаемый sample:

```text
float32[num_visual_nodes, d_visual]
```

Настройки по умолчанию:

- `num_visual_nodes=36`;
- `d_visual=2048`.

### Graph HDF5

Ключ: `annot_id`.

Обязательные datasets в каждой group:

```text
node_features  float32[N_kg_actual, d_kg]
adj_matrix     float32[N_total, N_total]
node_types     int32/int64[N_total]
```

Где:

```text
N_total = num_visual_nodes + 1 + max_kg_nodes
```

Настройки по умолчанию:

- `d_kg=300`;
- `max_kg_nodes=30`.

### VCR Batch Keys

`VCRDataset` + `make_vcr_collate_fn` создают:

- `visual_features`;
- `qa_input_ids`, `qa_attention_mask`;
- `qar_input_ids`, `qar_attention_mask`;
- `graph_node_features`;
- `graph_adj`;
- `graph_node_types`;
- `answer_label`;
- `rationale_label`;
- `annot_ids` (persisted by inferencer as `sample_id` для post-hoc join).

VCR model output — один scalar score на candidate. Это `candidate scoring`, а
не answer-vocabulary classification.

Object mentions в question/answers/rationale проходят через
`_vcr_object_mention`, который рендерит per-instance identifiers
(`person0`, `person2`). Это disambiguates same-category references
(два "person" не сливаются в одну строку), но не привязывает упоминания
к region detector features напрямую.

## GQA Contract

### Обязательные Файлы

```text
data/gqa/
├── gqa_answer_vocab.json
├── questions/
│   ├── train_balanced_questions.json
│   └── val_balanced_questions.json
├── visual_features/
│   ├── train_features.h5
│   └── val_features.h5
└── knowledge_graphs/
    ├── train_graphs.h5
    └── val_graphs.h5
```

### Answer Vocabulary

`gqa_answer_vocab.json` должен содержать:

- `answer_to_idx`;
- `idx_to_answer`;
- optional `metadata`.

Paper target:

```text
num_answers = 1842
```

Текущий `baseline_gqa.yaml` задаёт `num_answers=1842`.

### Question JSON

Каждый question item должен содержать:

- `question`;
- `imageId`;
- `answer` для train/val.

### Visual HDF5

Ключ: `imageId`.

Paper-like expected sample:

```text
float32[100, 2048]
```

Optional file attrs проверяются, если присутствуют:

- `num_boxes`;
- `feature_dim`.

### Graph HDF5

Ключ: GQA `question_id`.

Paper-like required datasets в каждой group:

```text
node_features     float32[N_textual_nodes, d_kg]
adj_matrix        float32[N_total, N_total]
node_types        int32/int64[N_total]
graph_edge_types  int32/int64[N_total, N_total]
```

Default paper-like config:

```text
d_kg = 600
num_visual_nodes = 100
max_kg_nodes = 100
N_total = 201
```

Рекомендуемые file attrs:

- `d_kg`;
- `num_visual_nodes`;
- `max_kg_nodes`;
- `graph_mode=official_scene_graph`;
- `conceptnet_used=False`;
- `graph_edge_type_count`.

Если attrs присутствуют и не совпадают с config, `GQADataset` поднимает
`ValueError` до продолжения training.

### Relation-Aware GNN Contract (GQA)

`GQAVQAGNNModel` теперь consumes `graph_edge_types` в каждом dense GAT
слое. Требования:

- `graph_edge_types` должен присутствовать в batch и в
  `inferencer.device_tensors`.
- Все значения `graph_edge_types` должны быть в диапазоне
  `[0, model.num_relations)`. Out-of-range values поднимают явный
  `ValueError` в `DenseGATLayer.forward`.
- `model.num_relations` (default `512`) — safe upper bound. После
  пересборки graph artifacts значение можно свести к фактическому
  `relation_count_total` из `gqa_relation_vocab.json`.
- VCR model конструируется с `num_relations=0` и не использует
  `graph_edge_types` — это backward-compatible untyped путь.

### GQA Batch Keys

`GQADataset` + `gqa_collate_fn` создают:

- `visual_features`;
- `question_input_ids`;
- `question_attention_mask`;
- `graph_node_features`;
- `graph_adj`;
- `graph_edge_types` (consumed by relation-aware GAT);
- `graph_node_types`;
- `labels`;
- `question_id` (persisted by inferencer as `sample_id` для post-hoc join).

GQA model output — `[B, num_answers]`. Это `1842-way classification`, а не
candidate scoring.

## Текущая Находка По Local GQA Artifacts

Во время аудита наблюдалось:

```text
data/gqa/gqa_answer_vocab.json: 1842 answers
data/gqa/visual_features/train_features.h5: sample shape (100, 2048)
data/gqa/knowledge_graphs/train_graphs.h5: sample node_features (2, 300)
data/gqa/knowledge_graphs/train_graphs.h5: no graph_edge_types
data/gqa/knowledge_graphs/train_graphs.h5: empty file attrs
```

Текущий strict GQA config ожидает `d_kg=600` и `graph_edge_types`. Поэтому
real-data startup намеренно падает:

```text
ValueError: [GQA data contract] ... node_features with d_kg=300, but dataset config requests d_kg=600.
```

Классификация:

- `blocked by data`, пока artifacts не пересобраны с typed scene-graph edges
  (`d_kg=600`, `graph_edge_types`, attrs).

## Варианты Исправления GQA

### Paper-Like Fix

Пересобрать graphs текущим scene-graph preprocessing path:

```bash
python scripts/prepare_gqa_data.py prepare-all \
  --train-questions data/raw/gqa/questions/train_balanced_questions.json \
  --val-questions data/raw/gqa/questions/val_balanced_questions.json \
  --scene-graphs data/raw/gqa/train_sceneGraphs.json data/raw/gqa/val_sceneGraphs.json \
  --glove data/raw/glove/glove.840B.300d.txt \
  --raw-features data/raw/gqa/objects \
  --info-json data/raw/gqa/gqa_objects_info.json \
  --processed-dir data/gqa
```

Если остальные artifacts уже валидны, можно пересобрать только graph artifacts:

```bash
python scripts/prepare_gqa_data.py knowledge-graphs \
  --questions-json data/raw/gqa/questions/train_balanced_questions.json \
  --scene-graphs data/raw/gqa/train_sceneGraphs.json data/raw/gqa/val_sceneGraphs.json \
  --glove data/raw/glove/glove.840B.300d.txt \
  --raw-features data/raw/gqa/objects \
  --info-json data/raw/gqa/gqa_objects_info.json \
  --relation-vocab data/gqa/gqa_relation_vocab.json \
  --output data/gqa/knowledge_graphs/train_graphs.h5 \
  --d-kg 600
```

Повторите graph-only команду для `val_balanced_questions.json` и
`data/gqa/knowledge_graphs/val_graphs.h5`.

Затем провалидируйте:

```bash
python scripts/validate_gqa_data.py \
  --data-dir data/gqa \
  --answer-vocab data/gqa/gqa_answer_vocab.json \
  --text-encoder data/roberta-large
```

### Legacy Debug Run

Чтобы исследовать старые локальные artifacts без claims о paper-faithfulness,
нужно явно согласовать dimensions и разрешить отсутствующие edge types:

```bash
python train.py --config-name baseline_gqa \
  datasets=gqa \
  model.d_kg=300 \
  datasets.train.d_kg=300 \
  datasets.val.d_kg=300 \
  datasets.train.expect_graph_edge_types=False \
  datasets.val.expect_graph_edge_types=False
```

Такой запуск остаётся `not paper-faithful yet`, потому что typed scene-graph
relations отсутствуют.

## Команды Валидации

Engineering contracts покрываются:

```bash
.venv/bin/python -m pytest tests/test_engineering_contracts.py -q
```

Полный local test suite на момент аудита:

```bash
.venv/bin/python -m pytest -q
```

Проверенный результат:

```text
81 passed
```
