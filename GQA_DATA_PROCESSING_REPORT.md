# GQA Data Processing Report

## Scope

Этот документ фиксирует финальный GQA preprocessing path для текущего репозитория.
Он описывает:

- какие raw-файлы требуются;
- как строится processed layout;
- какие решения являются article-like;
- какие приближения всё ещё остаются;
- как подготовить Kaggle-ready package;
- как проверить готовность данных до реального запуска.

Репозиторий не заявляет, что это точный исходный paper artifact из статьи.
Это максимально близкий и воспроизводимый GQA data path внутри текущего шаблона.

## Required Raw Inputs

Нужны следующие исходные файлы:

- `train_balanced_questions.json`
- `val_balanced_questions.json`
- `train_sceneGraphs.json` / `val_sceneGraphs.json` или эквивалентный combined scene-graph export
- официальный GQA object-features release:
  - `gqa_objects.h5` или директория shard-файлов `gqa_objects_*.h5`
  - `gqa_objects_info.json`
- GloVe `300d` файл, передаваемый в `--glove`

Рекомендуемый raw layout:

```text
data/raw/gqa/
├── questions/
│   ├── train_balanced_questions.json
│   └── val_balanced_questions.json
├── train_sceneGraphs.json
├── val_sceneGraphs.json
├── objects/
│   ├── gqa_objects_0.h5
│   ├── gqa_objects_1.h5
│   └── ...
└── gqa_objects_info.json
```

## Processed Layout

После preprocessing expected runtime layout такой:

```text
data/gqa/
├── questions/
│   ├── train_balanced_questions.json
│   └── val_balanced_questions.json
├── visual_features/
│   ├── train_features.h5
│   └── val_features.h5
├── knowledge_graphs/
│   ├── train_graphs.h5
│   └── val_graphs.h5
├── metadata/
│   ├── train_visual_features_metadata.json
│   ├── val_visual_features_metadata.json
│   ├── train_knowledge_graphs_metadata.json
│   ├── val_knowledge_graphs_metadata.json
│   └── package_manifest.json
├── gqa_answer_vocab.json
└── gqa_relation_vocab.json
```

## Final Pipeline

Главная команда:

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

Нижележащие команды можно запускать отдельно:

```bash
python scripts/prepare_gqa_data.py answer-vocab ...
python scripts/prepare_gqa_data.py visual-features ...
python scripts/prepare_gqa_data.py knowledge-graphs ...
```

## Article-Like Decisions

### Questions

- используются официальные `train_balanced_questions.json` и `val_balanced_questions.json`;
- legacy VQA-v2 conversion path удалён из GQA source-of-truth;
- runtime читает те же official balanced splits.

### Visual Features

- runtime expects official GQA object-feature semantics: `100 x 2048`;
- `imageId` coverage проверяется против question splits;
- packaged HDF5 keyed by `imageId`;
- split-specific metadata фиксирует coverage и shape.

### Answer Vocabulary

- answer normalization намеренно лёгкая:
  - lowercase
  - trim
  - collapse internal whitespace
- агрессивная VQA-v2 нормализация не используется, чтобы не схлопывать GQA classes искусственно;
- fixed answer space строится детерминированно;
- tie-breaking: `frequency desc`, затем `lexicographic`;
- в текущем released balanced setup union `train+val` после этой нормализации даёт ровно `1842` уникальных ответа.

Важно:

- это решение является **инференсом из released balanced splits**, а не прямой цитатой статьи;
- train-only balanced split даёт меньше классов (`1833`), поэтому union `train+val` здесь выбран как более близкий путь к paper target `1842`.

### KG Construction

- для GQA больше не используется ConceptNet path;
- граф строится как `question-context node + visual scene graph + textual scene graph`;
- visual SG опирается на official object features и bbox alignment к official scene graphs;
- textual SG node feature = `concat(GloVe(object_name), GloVe(mean_attributes))`;
- relation edges берутся из official scene-graph predicates и сохраняются как dense `graph_edge_types`;
- fully-connected fallback удалён;
- split metadata фиксирует no-match rate, average textual nodes, average visual nodes и relation-edge coverage.

### Entity Extraction / Linking

Новый linker детерминированный и rule-based:

- normalizes text;
- генерирует n-gram candidate spans;
- предпочитает longest multiword match;
- умеет singularize tail token;
- убирает stop-words и boilerplate question tokens;
- не сводится к простому token split / биграммам без приоритета.

Этот matcher теперь используется не для ConceptNet retrieval, а для question-to-scene-graph grounding и приоритезации textual SG nodes.

## Remaining Approximations

Ниже перечислены оставшиеся отклонения от статьи:

- KG по-прежнему строится офлайн, а не on-the-fly во время forward;
- используется открытый GloVe `300d`, а не закрытый paper artifact;
- exact internal paper matcher не опубликован, поэтому rule-based grounding остаётся приближением;
- bbox alignment между scene-graph objects и official object features делается greedy IoU-правилом;
- runtime baseline сохраняет dense graph contract; relation ids теперь сохраняются в данных, но текущий GNN runtime остаётся инженерной аппроксимацией paper message passing.

Эти отклонения должны описываться как **article-like approximation**, а не как точное воспроизведение.

## Validation

Строгая проверка:

```bash
python scripts/validate_gqa_data.py \
  --data-dir data/gqa \
  --answer-vocab data/gqa/gqa_answer_vocab.json \
  --text-encoder data/roberta-large
```

Что проверяется:

- размер answer vocab;
- contiguous indices;
- relation vocab и special relation ids;
- shape visual features;
- coverage `imageId`;
- coverage `question_id`;
- наличие KG metadata и запрет `fully_connected_fallback`;
- no-match rate;
- average graph size;
- наличие `graph_edge_types`;
- runtime compatibility `dataset -> collate -> model -> loss -> metric`;
- one-batch forward.

Если нужен только data-only check без model forward:

```bash
python scripts/validate_gqa_data.py \
  --data-dir data/gqa \
  --answer-vocab data/gqa/gqa_answer_vocab.json \
  --skip-runtime-check
```

## Kaggle Packaging

Full:

```bash
bash scripts/stage_gqa_for_kaggle.sh
```

Mini:

```bash
VARIANT=mini MINI_N=500 bash scripts/stage_gqa_for_kaggle.sh
```

Или напрямую:

```bash
python scripts/prepare_gqa_data.py stage \
  --data-dir data/gqa \
  --output-dir kaggle_staging/gqa-gnn-data \
  --variant full
```

Staged package включает:

- processed runtime files;
- `dataset-metadata.json`;
- `metadata/package_manifest.json`.

## Ready-for-Run Checklist

Перед реальным запуском baseline должны быть выполнены все пункты:

- `prepare-all` completed without errors;
- `validate_gqa_data.py` passes;
- answer vocab size matches expected `1842`;
- `train_features.h5` and `val_features.h5` are `100 x 2048`;
- KG metadata shows `fully_connected_fallback_used = false`;
- one-batch runtime forward succeeds.
