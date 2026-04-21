# Отчёт по обработке данных GQA

## Область охвата

Этот документ фиксирует финальный путь препроцессинга GQA в текущем репозитории.
Он описывает:

- какие raw-файлы требуются;
- как формируется обработанная структура каталогов;
- какие решения считаются близкими к статье;
- какие приближения всё ещё остаются;
- как готовить пакет для Kaggle;
- как проверить готовность данных до реального запуска.

Репозиторий не утверждает, что это точный исходный артефакт статьи.
Это максимально близкий и воспроизводимый GQA data path внутри текущего шаблона.

## Необходимые исходные данные

Нужны следующие raw-файлы:

- `train_balanced_questions.json`
- `val_balanced_questions.json`
- `train_sceneGraphs.json` / `val_sceneGraphs.json` или эквивалентный объединённый export
- официальный релиз GQA object features:
  - `gqa_objects.h5` или директория shard-файлов `gqa_objects_*.h5`
  - `gqa_objects_info.json`
- файл GloVe `300d`, передаваемый через `--glove`

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

## Обработанная Структура Каталогов

После препроцессинга runtime ожидает такую структуру:

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

## Финальный Pipeline

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

Внутренние команды можно запускать и по отдельности:

```bash
python scripts/prepare_gqa_data.py answer-vocab ...
python scripts/prepare_gqa_data.py visual-features ...
python scripts/prepare_gqa_data.py knowledge-graphs ...
```

## Решения, близкие к статье

### Вопросы

- используются официальные `train_balanced_questions.json` и `val_balanced_questions.json`;
- legacy-путь через конвертацию в формат VQA v2 удалён из source-of-truth для GQA;
- runtime читает те же официальные balanced splits.

### Визуальные признаки

- runtime ожидает официальную семантику GQA object features: `100 × 2048`;
- покрытие `imageId` проверяется относительно question splits;
- итоговый HDF5 упаковывается с ключами по `imageId`;
- split-specific metadata фиксирует покрытие и форму тензоров.

### Словарь ответов

- нормализация ответов намеренно лёгкая:
  - перевод в lowercase;
  - обрезка пробелов по краям
  - схлопывание повторяющихся пробелов внутри строки
- агрессивная нормализация из VQA v2 не используется, чтобы искусственно не склеивать классы GQA;
- фиксированное пространство ответов строится детерминированно;
- разрешение тай-брейков: сначала `frequency desc`, затем `lexicographic`;
- в текущем released balanced setup объединение `train+val` после этой нормализации
  даёт ровно `1842` уникальных ответа.

Важно:

- это решение является **инференсом по released balanced splits**, а не прямой цитатой статьи;
- только train-balanced split даёт меньше классов (`1833`), поэтому здесь выбран
  объединение `train+val` как более близкий путь к целевому значению статьи `1842`.

### Построение графа знаний

- для GQA больше не используется путь через ConceptNet;
- граф строится как `question-context node + visual scene graph + textual scene graph`;
- visual SG опирается на официальные object features и bbox alignment к официальным scene graphs;
- textual SG feature узла = `concat(GloVe(object_name), GloVe(mean_attributes))`;
- relation edges берутся из официальных scene-graph predicates и сохраняются как dense `graph_edge_types`;
- fully-connected fallback удалён;
- split metadata фиксирует no-match rate, среднее число textual nodes, среднее число
  visual nodes и покрытие relation edges.

### Извлечение сущностей / привязка

Новый linker детерминированный и rule-based:

- нормализует текст;
- генерирует кандидатные span'ы из n-грамм;
- предпочитает самые длинные multiword match;
- умеет singularize tail token;
- удаляет стоп-слова и boilerplate-токены из вопроса;
- не сводится к простому `split` по токенам или биграммам без приоритета.

Этот matcher теперь используется не для ConceptNet retrieval, а для
привязки вопроса к scene graph и приоритезации textual SG nodes.

## Оставшиеся приближения

Ниже перечислены отклонения от статьи, которые всё ещё остаются:

- KG по-прежнему строится офлайн, а не во время `forward`;
- используется открытый GloVe `300d`, а не закрытый артефакт статьи;
- точный внутренний matcher статьи не опубликован, поэтому rule-based grounding остаётся приближением;
- bbox alignment между scene-graph objects и official object features строится жадным правилом по IoU;
- runtime baseline сохраняет dense graph contract; relation ids теперь едут в данных,
  но сам GNN runtime остаётся инженерной аппроксимацией механизма message passing из статьи.

Эти отклонения нужно описывать как **приближение, близкое к статье**, а не как точное воспроизведение.

## Валидация

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
- relation vocab и специальные relation ids;
- shape visual features;
- покрытие `imageId`;
- покрытие `question_id`;
- наличие KG metadata и запрет `fully_connected_fallback`;
- no-match rate;
- средний размер графа;
- наличие `graph_edge_types`;
- runtime-совместимость `dataset -> collate -> model -> loss -> metric`;
- one-batch forward.

Если нужен только data-only check без model forward:

```bash
python scripts/validate_gqa_data.py \
  --data-dir data/gqa \
  --answer-vocab data/gqa/gqa_answer_vocab.json \
  --skip-runtime-check
```

## Упаковка для Kaggle

Полный вариант:

```bash
bash scripts/stage_gqa_for_kaggle.sh
```

Мини-вариант:

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

Подготовленный пакет включает:

- обработанные runtime files;
- `dataset-metadata.json`;
- `metadata/package_manifest.json`.

## Чек-лист готовности к запуску

Перед реальным запуском baseline должны быть выполнены все пункты:

- `prepare-all` завершился без ошибок;
- `validate_gqa_data.py` проходит успешно;
- размер answer vocab совпадает с ожидаемым `1842`;
- `train_features.h5` и `val_features.h5` имеют форму `100 × 2048`;
- KG metadata показывает `fully_connected_fallback_used = false`;
- one-batch runtime forward завершается успешно.
