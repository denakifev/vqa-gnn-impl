# Практический регламент запуска VQA-GNN

Пошаговый практический план запуска VQA-GNN с максимальной воспроизводимостью.
Целевая среда обучения: Kaggle. Подготовка данных: локально или Colab.

Статья: **VQA-GNN: Reasoning with Multimodal Knowledge via Graph Neural Networks for VQA**
(arXiv:2205.11501, ICCV 2023)

**Основные бенчмарки**: VCR и GQA — именно они воспроизводятся.
Устаревший VQA v2 контур описан в конце документа.

---

## 1. Аудит готовности к запуску

### Что реализовано и проверено (VCR + GQA)

| Компонент | Статус | Примечание |
|---|---|---|
| `VCRVQAGNNModel` | `implemented`, `validated` | VCR: скалярная оценка кандидата, выход dim=1 |
| `GQAVQAGNNModel` | `implemented`, `validated` | GQA: классификатор по словарю ответов, выход dim=1842 |
| `VCRDataset` / `VCRDemoDataset` | `implemented`, частично `validated` | реальный путь требует JSONL + HDF5 артефактов; demo-путь `runtime-valid` |
| `GQADataset` / `GQADemoDataset` | `implemented`, частично `validated` | реальный путь требует JSON + HDF5 артефактов; demo-путь `runtime-valid` |
| `VCRLoss`, `GQALoss` | `implemented`, `validated` | CE по кандидатам / по словарю |
| `VCRQAAccuracy`, `VCRQARAccuracy`, `GQAAccuracy` | `implemented`, `validated` | согласованы со статьёй на toy/demo проверках |
| `prepare_vcr_data.py` | `implemented` | KG-подграфы из JSONL |
| `prepare_gqa_data.py` | `implemented` | словарь ответов + scene-graph графы |
| `validate_vcr_data.py` | `implemented` | проверяет JSONL + HDF5 |
| `validate_gqa_data.py` | `implemented` | проверяет JSON + HDF5 |
| `stage_vcr_for_kaggle.sh` | `implemented` | mini/full упаковка для приватного Kaggle Dataset |
| `stage_gqa_for_kaggle.sh` | `implemented` | mini/full упаковка для Kaggle |
| `run_kaggle_smoke_test.sh` | `implemented` | GQA и VCR режимы |
| Конфиги VCR/GQA | `validated` | `baseline_vcr_{qa,qar}.yaml`, `baseline_gqa.yaml` успешно собираются |
| Demo smoke-тесты | `runtime-valid` | проверяют пайплайн без реальных данных |
| Числа из статьи | **не подтверждены** | требуется реальный сквозной запуск |

### Ключевые внешние блокеры

| Блокер | Серьёзность | Обходной путь |
|---|---|---|
| Визуальные признаки VCR (R2C Faster RCNN, ~2048-dim) | **критический** | нет; нужны заранее извлечённые |
| Визуальные признаки GQA на базе Visual Genome | **критический** | нет; нужны заранее извлечённые |
| Точный детектор из статьи не опубликован | средний | bottom-up признаки как инженерное приближение |
| VCR лицензия — нельзя публиковать публично | организационный | только приватный Kaggle Dataset |
| Числа из статьи не подтверждены | информационный | честно документируем |

### Контрольные документы перед реальным запуском

- [VALIDATION_REPORT.md](VALIDATION_REPORT.md) — что реально запускалось и что прошло.
- [DATA_CONTRACTS.md](DATA_CONTRACTS.md) — обязательные batch/HDF5-контракты и fail-fast поведение.
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — какие gaps блокируют первый реальный запуск и paper-faithful reproduction.

### Оценка объёма данных

| Артефакт | Примерный объём для упаковки |
|---|---|
| Полный GQA Kaggle Dataset | ~12–20 GB |
| Полный VCR Kaggle Dataset | ~12–18 GB |
| ConceptNet Numberbatch | ~450 MB (сжатый) |
| roberta-large checkpoint | ~1.4 GB |
| GQA mini (500 вопросов) | < 100 MB |
| VCR mini (200 аннотаций) | < 50 MB |

---

## 2. Быстрый старт без реальных данных (смоук-тест)

Смоук-тест проверяет весь training pipeline на синтетических данных и НЕ
воспроизводит статью. Базовые configs используют `roberta-large`; в свежей
среде это может потребовать локальный checkpoint/cache или загрузку с Hugging
Face. Для быстрого CPU-аудита использовался tiny RoBERTa override; точные
команды зафиксированы в [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

```bash
# Смоук-тест GQA:
python train.py --config-name baseline_gqa

# Смоук-тест VCR Q→A:
python train.py --config-name baseline_vcr_qa

# Смоук-тест VCR QA→R:
python train.py --config-name baseline_vcr_qar
```

---

## 3. Полный пошаговый план

### Этап 0: Подготовка окружения

```bash
git clone <your-repo-url> paper_model_replication
cd paper_model_replication
pip install -r requirements.txt

# Проверка установки
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from transformers import AutoTokenizer; print('transformers OK')"
python -c "import h5py; print('h5py OK')"
python -c "import comet_ml; print('comet_ml OK')"
```

---

### Этап 1: Получение данных

#### GQA (CC BY 4.0)

Скачайте вручную с https://cs.stanford.edu/people/dorarad/gqa/download.html :

```bash
mkdir -p data/gqa/questions
# После скачивания:
mv train_balanced_questions.json data/gqa/questions/
mv val_balanced_questions.json   data/gqa/questions/
# Визуальные признаки разместить в data/gqa/visual_features/
```

Подробнее: [KAGGLE_GQA.md](KAGGLE_GQA.md)

#### VCR (VCR Terms of Use, только некоммерческое использование)

Скачайте вручную с https://visualcommonsense.com/ (требует регистрации):

```bash
mkdir -p data/vcr
# После скачивания:
mv train.jsonl data/vcr/
mv val.jsonl   data/vcr/
# Визуальные признаки разместить в data/vcr/visual_features/
```

**Ограничение**: не загружайте VCR как публичный датасет на Kaggle.

Подробнее: [KAGGLE_VCR.md](KAGGLE_VCR.md)

#### Внешние источники

`GQA` использует официальные scene graphs + GloVe 300d.

`VCR` использует `ConceptNet Numberbatch`:

Уже лежит в `data/conceptnet/`. Если нет:

```bash
mkdir -p data/conceptnet
wget -P data/conceptnet \
  https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz

# Опционально (улучшает структуру KG рёбер):
wget -P data/conceptnet \
  https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
```

---

### Этап 2: Препроцессинг GQA

```bash
# Один pipeline, согласованный со статьёй:
python scripts/prepare_gqa_data.py prepare-all \
  --train-questions data/raw/gqa/questions/train_balanced_questions.json \
  --val-questions data/raw/gqa/questions/val_balanced_questions.json \
  --scene-graphs data/raw/gqa/train_sceneGraphs.json data/raw/gqa/val_sceneGraphs.json \
  --glove data/raw/glove/glove.840B.300d.txt \
  --raw-features data/raw/gqa/objects \
  --info-json data/raw/gqa/gqa_objects_info.json \
  --processed-dir data/gqa

# Строгая валидация:
python scripts/validate_gqa_data.py \
  --data-dir data/gqa \
  --answer-vocab data/gqa/gqa_answer_vocab.json \
  --text-encoder data/roberta-large
```

Дополнительный отчёт по решениям и приближениям: [GQA_DATA_PROCESSING_REPORT.md](GQA_DATA_PROCESSING_REPORT.md)

Минимальный runtime contract для paper-like GQA:

- answer vocabulary: 1842 classes;
- visual features: `100×2048`;
- graph `node_features`: `d_kg=600`;
- graph `graph_edge_types`: present;
- graph HDF5 attrs: `d_kg`, `num_visual_nodes`, `max_kg_nodes` желательно
  присутствуют для ранней проверки.

Текущие локальные `data/gqa/knowledge_graphs/*.h5` в этой рабочей копии имеют
legacy `d_kg=300` и не проходят strict config. Это `blocker for first real run`,
а не paper result. Детали и допустимые legacy overrides описаны в
[DATA_CONTRACTS.md](DATA_CONTRACTS.md).

---

### Этап 3: Препроцессинг VCR

```bash
# Шаг 1: KG-подграфы train (~2–4 часа, CPU):
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/train.jsonl \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --output data/vcr/knowledge_graphs/train_graphs.h5 \
  --d-kg 300

# Шаг 2: KG-подграфы val:
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/val.jsonl \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --output data/vcr/knowledge_graphs/val_graphs.h5 \
  --d-kg 300

# Шаг 3: Проверка покрытия visual features:
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/val.jsonl \
  --validate-visual data/vcr/visual_features/val_features.h5

# Шаг 4: Валидация:
python scripts/validate_vcr_data.py \
  --data-dir data/vcr \
  --split train val \
  --d-kg 300
```

---

### Этап 4: Подготовка Kaggle Datasets

**GQA (можно публичным или приватным):**
```bash
# Мини-вариант для sanity-проверок:
VARIANT=mini MINI_N=500 bash scripts/stage_gqa_for_kaggle.sh

# Полный вариант для реального обучения:
VARIANT=full bash scripts/stage_gqa_for_kaggle.sh

# Загрузка:
kaggle datasets create -p kaggle_staging/gqa-gnn-data
```

**VCR (только приватный):**
```bash
# Мини-вариант:
VARIANT=mini MINI_N=200 bash scripts/stage_vcr_for_kaggle.sh

# Полный вариант:
VARIANT=full bash scripts/stage_vcr_for_kaggle.sh

# Загрузка (приватный):
kaggle datasets create -p kaggle_staging/vcr-gnn-data
```

После загрузки убедитесь, что VCR Dataset имеет **Private** visibility.

---

### Этап 5: Обучение на Kaggle

#### GQA

```bash
# Смоук-тест (mini-данные):
TASK=gqa \
OFFLINE=1 \
TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
GQA_DATA_DIR=/kaggle/input/gqa-gnn-data-mini \
EPOCH_LEN=50 N_EPOCHS=1 \
bash scripts/run_kaggle_smoke_test.sh

# Полное обучение:
python train.py --config-name baseline_gqa \
  datasets=gqa \
  datasets.train.data_dir=/kaggle/input/gqa-gnn-data \
  datasets.train.answer_vocab_path=/kaggle/input/gqa-gnn-data/gqa_answer_vocab.json \
  datasets.val.data_dir=/kaggle/input/gqa-gnn-data \
  datasets.val.answer_vocab_path=/kaggle/input/gqa-gnn-data/gqa_answer_vocab.json \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  dataloader.batch_size=4 \
  trainer.n_epochs=20 trainer.epoch_len=5000 \
  trainer.save_dir=/kaggle/working/saved \
  writer.mode=offline writer.run_name=gqa_baseline
```

#### VCR Q→A

```bash
# Смоук-тест:
TASK=vcr VCR_TASK_MODE=qa \
OFFLINE=1 \
TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
VCR_DATA_DIR=/kaggle/input/vcr-gnn-data-mini \
EPOCH_LEN=50 N_EPOCHS=1 \
bash scripts/run_kaggle_smoke_test.sh

# Полное обучение Q→A:
python train.py --config-name baseline_vcr_qa \
  datasets=vcr_qa \
  datasets.train.data_dir=/kaggle/input/vcr-gnn-data \
  datasets.val.data_dir=/kaggle/input/vcr-gnn-data \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  dataloader.batch_size=4 \
  trainer.n_epochs=20 trainer.epoch_len=5000 \
  trainer.save_dir=/kaggle/working/saved \
  writer.mode=offline writer.run_name=vcr_qa_baseline
```

#### VCR QA→R (отдельная модель)

```bash
python train.py --config-name baseline_vcr_qar \
  datasets=vcr_qar \
  datasets.train.data_dir=/kaggle/input/vcr-gnn-data \
  datasets.val.data_dir=/kaggle/input/vcr-gnn-data \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  dataloader.batch_size=4 \
  trainer.n_epochs=20 trainer.epoch_len=5000 \
  trainer.save_dir=/kaggle/working/saved \
  writer.mode=offline writer.run_name=vcr_qar_baseline
```

---

### Этап 6: Инференс

```bash
# GQA:
python inference.py --config-name inference_gqa \
  datasets.val.data_dir=/kaggle/input/gqa-gnn-data \
  datasets.val.answer_vocab_path=/kaggle/input/gqa-gnn-data/gqa_answer_vocab.json \
  inferencer.from_pretrained=saved/CHECKPOINT_DIR/best_model.pth \
  inferencer.save_path=data/saved/gqa_predictions

# VCR:
python inference.py --config-name inference_vcr \
  datasets.val.data_dir=/kaggle/input/vcr-gnn-data \
  inferencer.from_pretrained=saved/CHECKPOINT_DIR/best_model.pth \
  inferencer.save_path=data/saved/vcr_predictions
```

---

## 4. Критические несогласованности (нужно знать)

**d_kg должен совпадать в трёх местах:**
1. `--d-kg` в `prepare_vcr_data.py` / `prepare_gqa_data.py`
2. `src/configs/model/vqa_gnn_{vcr,gqa}.yaml` → `d_kg`
3. `src/configs/datasets/{vcr_qa,vcr_qar,gqa}.yaml` → `d_kg`

Сейчас дефолты различаются по задаче:
- `VCR`: `d_kg: 300` (ConceptNet / Numberbatch)
- `GQA`: `d_kg: 600` (`concat(GloVe(name), GloVe(attributes))`)

Менять только вместе с preprocessing.

**batch_size=4** при `roberta-large` и `d_hidden=1024`. Значение по умолчанию в configs — не для T4/P100. На Kaggle всегда переопределяйте:
```
dataloader.batch_size=4
```

**text_encoder_name**: по умолчанию `roberta-large` (HuggingFace). На Kaggle без интернета используйте локальный путь:
```
model.text_encoder_name=/kaggle/input/roberta-large
datasets.train.text_encoder_name=/kaggle/input/roberta-large
datasets.val.text_encoder_name=/kaggle/input/roberta-large
```

**writer.mode=offline**: на Kaggle без CometML API Key. Логи сохраняются в `/kaggle/working/saved/`.

---

## 5. Чекпойнты и артефакты

Сохраняются автоматически через `src/utils/init_utils.py` в `saved/{run_name}_{timestamp}/`:

```
saved/gqa_baseline_20260413_120000/
├── config.yaml        # полная Hydra-конфигурация
├── git_commit.txt     # текущий git commit
├── git_diff.patch     # незакоммиченные изменения
├── model_best.pth     # лучший checkpoint по val_GQA_Accuracy
├── model_last.pth     # последний epoch checkpoint
└── log.txt            # лог обучения
```

---

## 6. Устаревший контур VQA v2 (не является воспроизведением статьи)

Код VQA v2 сохранён в `legacy/` и больше **не является приоритетом**.
Статья VQA-GNN оценивается на VCR и GQA — не на VQA v2.

Для справки, старые команды:
```bash
# Запустить старый demo (deprecated):
python train.py --config-name baseline_vqa  # только для тестирования legacy

# Подготовить данные VQA v2 (deprecated):
python scripts/prepare_answer_vocab.py --annotations data/vqa/annotations/train_annotations.json ...
python scripts/prepare_kg_graphs.py --questions data/vqa/questions/train_questions.json ...
```

Всё относящееся к VQA v2 (скрипты, конфиги, данные) остаётся для исторической
справки, но не используется в текущем контуре, согласованном со статьёй.

---

## 7. Шаги, которые нельзя автоматизировать

1. Принятие лицензий VCR (visualcommonsense.com) и GQA (stanford.edu)
2. Ручное скачивание датасетов и visual features
3. Установка Private visibility для VCR Kaggle Dataset
4. Итоговое сравнение с числами из статьи (нельзя — нет точных детекторных весов)

---

## 8. Дополнительные ресурсы

- [KAGGLE_GQA.md](KAGGLE_GQA.md) — детальные инструкции для GQA на Kaggle
- [KAGGLE_VCR.md](KAGGLE_VCR.md) — детальные инструкции для VCR на Kaggle
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — полный анализ расхождений с оригинальной статьёй
- [PAPER_ALIGNMENT_REPORT.md](PAPER_ALIGNMENT_REPORT.md) — чеклист компонентов и их статус
