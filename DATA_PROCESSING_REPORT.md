# VQA-GNN Data Processing Report

Этот документ фиксирует фактически выполненный data preprocessing для текущего
репозитория и может использоваться как основа для будущего отчёта по
воспроизведению.

Статус на момент составления:
- подготовка данных завершена;
- полный train ещё не запускался;
- все артефакты данных провалидированы на Kaggle;
- visual features и data artifacts уже выгружены в Kaggle Datasets.

## 1. Цель этапа

Подготовить полный набор артефактов для обучения VQA-GNN на VQA v2 в текущем
репозитории:

- `questions/*.json`
- `annotations/*.json`
- `answer_vocab.json`
- `visual_features/{train_features.h5,val_features.h5}`
- `knowledge_graphs/{train_graphs.h5,val_graphs.h5}`

Целевая среда обучения:
- train/inference: Kaggle
- preprocessing: локально + Colab + Kaggle publishing

## 2. Ключевые решения

В ходе подготовки был выбран следующий practically reproducible setup:

- visual features: public bottom-up attention features `K=36`, `2048-dim`
- источник visual features: Kaggle dataset
  `itsmariodias/coco-bottom-up-attention-features`
- graph node features: ConceptNet Numberbatch English-only, `d_kg=300`
- KG edge structure: без `assertions`, то есть fully-connected KG subgraph
- text encoder plan: `roberta-large`

Почему так:

- точный detector из статьи публично недоступен;
- старые официальные прямые ссылки на `trainval_36.zip` оказались нерабочими;
- Kaggle dataset с готовыми `.npy`-features оказался совместим с ожиданиями
  репозитория;
- `d_kg=300` лучше соответствует full Numberbatch и является более близким к
  paper-style setup, чем усечённый `100-dim` вариант.

## 3. Выполненные шаги

### 3.1. VQA v2 questions и annotations

С официального сайта VQA v2 были скачаны и переименованы:

- `train_questions.json`
- `val_questions.json`
- `train_annotations.json`
- `val_annotations.json`

Проверенные размеры:

- `train_questions.json`: `443757`
- `val_questions.json`: `214354`
- `train_annotations.json`: `443757`
- `val_annotations.json`: `214354`

Вывод:
- базовые split'ы VQA v2 подготовлены корректно.

### 3.2. Answer vocabulary

Был выполнен запуск:

```bash
python scripts/prepare_answer_vocab.py \
    --annotations data/vqa/annotations/train_annotations.json \
    --output data/vqa/answer_vocab.json \
    --top-n 3129
```

Полученный результат:

- vocabulary size: `3129`
- total unique normalized answers: `160024`
- total answer occurrences: `4437139`
- coverage of all answer occurrences: `89.2%`

Первые 10 ответов в словаре:

- `no`
- `yes`
- `2`
- `1`
- `white`
- `3`
- `red`
- `black`
- `blue`
- `0`

Вывод:
- `answer_vocab.json` построен корректно и соответствует стандартному setup
  VQA v2.

### 3.3. Visual features

#### 3.3.1. Что было найдено

Старые публичные ссылки на original bottom-up attention features оказались
нерабочими:

- `storage.googleapis.com/up-down-attention/trainval_36.zip` отвечал `403`
- `imagecaption.blob.core.windows.net/.../trainval_36.zip` отвечал `409`

Поэтому вместо них был использован Kaggle dataset:

- `itsmariodias/coco-bottom-up-attention-features`

#### 3.3.2. Что было проверено

По preview и реальной проверке содержимого было установлено:

- число `.npy` файлов: `123287`
- имя файла совпадает с `image_id`, например `1000.npy`
- shape одного файла: `(36, 2048)`
- dtype: `float32`

Это exactly совпадает с контрактом текущего репозитория:

- `num_visual_nodes = 36`
- `d_visual = 2048`

#### 3.3.3. Конвертация

Конвертация была выполнена из `.npy` в единый HDF5:

- output: `all_features.h5`
- number of entries: `123287`
- sample verification: `shape=36x2048`, `dtype=float32`
- итоговый размер файла: `24G`

#### 3.3.4. Kaggle publishing

Visual features были вынесены в отдельный Kaggle dataset:

- `daakifev/vqa-gnn-visual-features`

Это решение было принято специально, чтобы:

- не смешивать самый тяжёлый артефакт с мелкими файлами;
- не перезаливать `24G` при каждом обновлении остальных данных;
- упростить будущий train notebook.

### 3.4. Knowledge graphs

#### 3.4.1. Numberbatch

Был скачан:

- `numberbatch-en-19.08.txt.gz`

Реальный размер на диске:

- `311M` compressed

#### 3.4.2. Найденная проблема в preprocessing

Во время первого debug-run выяснилось, что
`scripts/prepare_kg_graphs.py` загружал `0` embeddings из Numberbatch.

Причина:

- loader ожидал full ConceptNet URI format (`/c/en/cat`);
- скачанный english-only Numberbatch использовал plain tokens
  (`cat`, `fire_truck`, `tennis_ball`).

#### 3.4.3. Исправление

В репозитории был исправлен loader в:

- [scripts/prepare_kg_graphs.py](/Users/srs15/Documents/CourseWorkR&D/paper_model_replication/scripts/prepare_kg_graphs.py)

После фикса скрипт стал корректно загружать:

- `516782 English entity embeddings (dim=300)`

#### 3.4.4. Debug run

Был выполнен debug-run на `1000` train-вопросов:

- output: `train_graphs_debug.h5`
- только `4` вопроса из `1000` потребовали zero-vector fallback

Это подтвердило:

- корректность формата Numberbatch;
- работоспособность `d_kg=300`;
- корректность HDF5 graph schema.

#### 3.4.5. Полный train graph

Был выполнен полный preprocessing:

- input questions: `443757`
- output: `data/vqa/knowledge_graphs/train_graphs.h5`
- итоговый размер: `4.9G`
- fallback questions without entity matches: `1358`

#### 3.4.6. Полный val graph

Был выполнен полный preprocessing:

- input questions: `214354`
- output: `data/vqa/knowledge_graphs/val_graphs.h5`
- итоговый размер: `2.4G`
- fallback questions without entity matches: `592`

#### 3.4.7. Какой вариант KG был зафиксирован

Для первого reproducible baseline был выбран более простой и стабильный
вариант:

- без `ConceptNet assertions`
- KG subgraph fully connected

Это является явным отклонением от статьи и должно быть задокументировано в
любом будущем отчёте.

### 3.5. Обновление конфигов

После построения KG с `d_kg=300` были обновлены конфиги:

- [src/configs/model/vqa_gnn.yaml](/Users/srs15/Documents/CourseWorkR&D/paper_model_replication/src/configs/model/vqa_gnn.yaml)
- [src/configs/datasets/vqa.yaml](/Users/srs15/Documents/CourseWorkR&D/paper_model_replication/src/configs/datasets/vqa.yaml)
- [src/configs/datasets/vqa_eval.yaml](/Users/srs15/Documents/CourseWorkR&D/paper_model_replication/src/configs/datasets/vqa_eval.yaml)

Итоговое значение:

- `d_kg: 300`

Demo configs (`vqa_demo*`) не менялись и остались на `100`.

### 3.6. Малый Kaggle dataset с JSON + KG

Для остальных артефактов был собран отдельный dataset root:

- `questions/`
- `annotations/`
- `knowledge_graphs/`
- `answer_vocab.json`

Размер собранного root:

- `7.8G`

Этот набор был опубликован как отдельный Kaggle dataset:

- `daakifev/vqa-gnn-data`

## 4. Полная валидация

На Kaggle был собран runtime root на symlink'ах/реальных путях и выполнен полный
прогон:

```bash
python scripts/validate_data.py \
    --data-dir /kaggle/working/vqa_runtime \
    --split train val \
    --answer-vocab /kaggle/working/vqa_runtime/answer_vocab.json \
    --num-visual-nodes 36 \
    --feature-dim 2048 \
    --d-kg 300 \
    --num-answers 3129 \
    --max-kg-nodes 30 \
    --sample-size 200
```

Итог validation summary:

- Passed: `27`
- Warnings: `0`
- Errors: `0`

Проверено:

- `answer_vocab.json` существует и имеет contiguous indices `[0, 3128]`
- все `train/val questions` и `annotations` найдены
- visual features:
  - HDF5 opened OK
  - `123287` image entries
  - все требуемые `image_id` покрыты
  - sample shape `(36, 2048)` OK
  - sample dtype `float32` OK
- KG graphs:
  - `443757` train question entries
  - `214354` val question entries
  - все `question_id` покрыты
  - schema с `d_kg=300` OK

Вывод:
- data pipeline на текущем этапе готов к запуску обучения.

## 5. Фактические артефакты

### Локальные артефакты

- `data/vqa/questions/train_questions.json`
- `data/vqa/questions/val_questions.json`
- `data/vqa/annotations/train_annotations.json`
- `data/vqa/annotations/val_annotations.json`
- `data/vqa/answer_vocab.json`
- `data/vqa/knowledge_graphs/train_graphs.h5`
- `data/vqa/knowledge_graphs/val_graphs.h5`

### Kaggle datasets

- visual features:
  - `daakifev/vqa-gnn-visual-features`
- data artifacts:
  - `daakifev/vqa-gnn-data`

## 6. Зафиксированные отклонения от статьи

Следующие отклонения обязательно должны упоминаться в будущем отчёте:

1. Visual features не из точного detector pipeline статьи.
   Используется публичный bottom-up attention setup `K=36`, `2048-dim`.

2. KG pipeline статьи воспроизведён приближённо.
   Используется ConceptNet Numberbatch English-only + simple tokenization.

3. Entity extraction упрощён.
   Используется tokenization + stopword filtering, а не NER/entity linking.

4. KG edge structure упрощена.
   Без `assertions` KG nodes fully connected.

5. Adjacency structure partially assumed.
   Используется:
   - visual↔question
   - kg↔question
   - kg↔kg
   - visual↔visual fully connected

6. Точный numerical reproduction статьи пока не подтверждён.
   На момент составления документа train ещё не запускался.

## 7. Что было дополнительно исправлено в коде по пути

В ходе подготовки данных были внесены важные инженерные правки:

1. [scripts/prepare_visual_features.py](/Users/srs15/Documents/CourseWorkR&D/paper_model_replication/scripts/prepare_visual_features.py)
   был исправлен так, чтобы корректно поддерживать official 6-column TSV format
   Anderson bottom-up attention release.

2. [scripts/prepare_kg_graphs.py](/Users/srs15/Documents/CourseWorkR&D/paper_model_replication/scripts/prepare_kg_graphs.py)
   был исправлен так, чтобы корректно читать english-only Numberbatch format
   (`cat`, `fire_truck`) наряду с URI format (`/c/en/cat`).

3. Конфиги реального VQA-run были синхронизированы с `d_kg=300`.

## 8. Осталось до первого train

На момент завершения data preprocessing до первого реального запуска обучения
остаются только runtime steps:

1. Подготовить доступ к `roberta-large` на Kaggle.
2. Подключить оба Kaggle dataset в train notebook:
   - `daakifev/vqa-gnn-visual-features`
   - `daakifev/vqa-gnn-data`
3. Собрать runtime root.
4. Запустить smoke-train на реальных данных.
5. После smoke-train запустить полный train.

## 9. Краткий итог

Data preprocessing для текущего baseline можно считать завершённым.

Готово:

- VQA v2 questions/annotations
- answer vocabulary
- visual features
- KG graphs
- Kaggle datasets
- full validation

Не готово:

- offline snapshot `roberta-large`
- первый train run
- отчёт о численных результатах

Итоговый статус:
- data pipeline готов для первого обучения на Kaggle.
