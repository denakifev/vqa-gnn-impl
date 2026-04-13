# GQA на Kaggle: инструкция по подготовке и запуску VQA-GNN

Этот документ описывает полный путь от получения исходных данных GQA до запуска
обучения VQA-GNN на Kaggle. Прочитайте его перед запуском `stage_gqa_for_kaggle.sh`.

---

## 1. Что нужно заранее подготовить локально

### 1.1 Данные GQA

GQA распространяется с лицензией Creative Commons BY 4.0.
Скачайте balanced split с официального сайта:

```
https://cs.stanford.edu/people/dorarad/gqa/download.html
```

Нужны следующие файлы:
- `train_balanced_questions.json` (~800 MB)
- `val_balanced_questions.json` (~130 MB)
- Visual features (Visual Genome based, ~10–15 GB)

Положите в `data/gqa/`:
```
data/gqa/
├── questions/
│   ├── train_balanced_questions.json
│   └── val_balanced_questions.json
└── visual_features/
    ├── train_features.h5   # необходимо подготовить (см. 1.3)
    └── val_features.h5
```

### 1.2 Official Scene Graphs и GloVe

Для paper-like GQA graph path теперь нужны:

- official `train_sceneGraphs.json` / `val_sceneGraphs.json` или эквивалентный combined export;
- GloVe `300d` файл, который будет передан в `--glove`;
- официальный GQA object-feature release с `gqa_objects_info.json`.

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
│   └── ...
└── gqa_objects_info.json
```

### 1.3 Визуальные признаки (HDF5)

Визуальные признаки для GQA нужно подготовить из официального GQA object-features
релиза и упаковать в runtime HDF5 (`100` объектов × `2048` dim):

```bash
# Train split:
python scripts/prepare_gqa_data.py visual-features \
    --raw-features data/raw/gqa/objects \
    --info-json data/raw/gqa/gqa_objects_info.json \
    --questions-json data/gqa/questions/train_balanced_questions.json \
    --output data/gqa/visual_features/train_features.h5 \
    --num-visual-nodes 100 --feature-dim 2048

# Val split:
python scripts/prepare_gqa_data.py visual-features \
    --raw-features data/raw/gqa/objects \
    --info-json data/raw/gqa/gqa_objects_info.json \
    --questions-json data/gqa/questions/val_balanced_questions.json \
    --output data/gqa/visual_features/val_features.h5 \
    --num-visual-nodes 100 --feature-dim 2048
```

Это сохраняет official GQA object-feature semantics и split-specific `imageId` coverage.

---

## 2. Preprocessing: paper-aligned GQA pipeline

Подробное описание решений и приближений: [GQA_DATA_PROCESSING_REPORT.md](GQA_DATA_PROCESSING_REPORT.md)

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

Она:

- копирует official balanced questions в processed layout;
- строит детерминированный `gqa_answer_vocab.json`;
- строит детерминированный `gqa_relation_vocab.json`;
- пакует official GQA object features в `100 x 2048` HDF5;
- строит paper-like graph artifacts из `question node + visual SG + textual SG`;
- пишет служебные metadata JSON.

Нижележащие шаги можно запускать отдельно:

```bash
python scripts/prepare_gqa_data.py answer-vocab ...
python scripts/prepare_gqa_data.py visual-features ...
python scripts/prepare_gqa_data.py knowledge-graphs ...
```

### 2.1 Валидация (перед загрузкой на Kaggle)

```bash
python scripts/validate_gqa_data.py \
  --data-dir data/gqa \
  --answer-vocab data/gqa/gqa_answer_vocab.json \
  --text-encoder data/roberta-large
```

Все проверки должны пройти с `[OK]`. Ошибки нужно исправить до загрузки.

---

## 3. Подготовка Kaggle Dataset

### 3.1 Mini-вариант (для sanity runs, рекомендуется для первого запуска)

Mini-вариант содержит первые 500 вопросов из train и val. Позволяет проверить весь
pipeline на Kaggle без затрат GPU-квоты на обучение.

```bash
VARIANT=mini MINI_N=500 bash scripts/stage_gqa_for_kaggle.sh
```

Создаёт: `kaggle_staging/gqa-gnn-data-mini/`

### 3.2 Full-вариант (для реального обучения)

```bash
VARIANT=full bash scripts/stage_gqa_for_kaggle.sh
```

Создаёт: `kaggle_staging/gqa-gnn-data/`

Внимание: копирование полных HDF5 файлов занимает время и требует ~15–30 GB свободного места.

### 3.3 Редактирование metadata перед загрузкой

Откройте `kaggle_staging/gqa-gnn-data/dataset-metadata.json` и замените:
```json
"id": "KAGGLE_USERNAME/gqa-gnn-data"
```
на ваш реальный Kaggle username.

### 3.4 Загрузка на Kaggle

```bash
# Первый раз:
kaggle datasets create -p kaggle_staging/gqa-gnn-data

# Обновление:
kaggle datasets version -p kaggle_staging/gqa-gnn-data -m "VQA-GNN GQA data"
```

---

## 4. Настройка Kaggle Notebook

### 4.1 Подключение датасетов

В настройках Notebook подключите:
- `KAGGLE_USERNAME/gqa-gnn-data` (или `-mini` для sanity runs)
- `KAGGLE_USERNAME/roberta-large` (или другой офлайн-энкодер)

Путь к данным в Notebook: `/kaggle/input/gqa-gnn-data/`

### 4.2 Установка зависимостей

```python
# В ячейке Notebook:
!pip install -q h5py comet_ml hydra-core omegaconf
!pip install -q "transformers>=4.30.0"
```

### 4.3 Клонирование репо

```bash
!git clone https://github.com/YOUR_USERNAME/paper_model_replication.git
%cd paper_model_replication
```

---

## 5. Запуск обучения на Kaggle

### 5.1 Smoke test (mini-данные, мало эпох)

```bash
TASK=gqa \
OFFLINE=1 \
TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
EPOCH_LEN=50 \
N_EPOCHS=1 \
GQA_DATA_DIR=/kaggle/input/gqa-gnn-data-mini \
bash scripts/run_kaggle_smoke_test.sh
```

Или напрямую:
```bash
python train.py --config-name baseline_gqa \
  datasets=gqa \
  datasets.train.data_dir=/kaggle/input/gqa-gnn-data-mini \
  datasets.train.answer_vocab_path=/kaggle/input/gqa-gnn-data-mini/gqa_answer_vocab.json \
  datasets.val.data_dir=/kaggle/input/gqa-gnn-data-mini \
  datasets.val.answer_vocab_path=/kaggle/input/gqa-gnn-data-mini/gqa_answer_vocab.json \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  trainer.epoch_len=50 \
  trainer.n_epochs=1 \
  trainer.save_dir=/kaggle/working/saved \
  writer.mode=offline \
  writer.run_name=smoke_gqa
```

### 5.2 Полное обучение

```bash
TASK=gqa \
OFFLINE=1 \
TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
GQA_DATA_DIR=/kaggle/input/gqa-gnn-data \
bash scripts/run_kaggle_smoke_test.sh
```

Или напрямую:
```bash
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
  trainer.n_epochs=20 \
  trainer.epoch_len=5000 \
  trainer.save_dir=/kaggle/working/saved \
  writer.mode=offline \
  writer.run_name=gqa_baseline
```

---

## 6. Инференс на Kaggle

```bash
python inference.py --config-name inference_gqa \
  datasets.val.data_dir=/kaggle/input/gqa-gnn-data \
  datasets.val.answer_vocab_path=/kaggle/input/gqa-gnn-data/gqa_answer_vocab.json \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  inferencer.from_pretrained=/kaggle/working/saved/CHECKPOINT_DIR/best_model.pth \
  inferencer.save_path=/kaggle/working/gqa_predictions
```

---

## 7. Ожидаемый объём данных

| Артефакт | Примерный размер (сжатый HDF5) |
|---|---|
| train_features.h5 | ~6–10 GB |
| val_features.h5 | ~2–4 GB |
| train_graphs.h5 (d_kg=300) | ~2–4 GB |
| val_graphs.h5 (d_kg=300) | ~0.5–1 GB |
| train_balanced_questions.json | ~800 MB |
| val_balanced_questions.json | ~130 MB |
| gqa_answer_vocab.json | < 1 MB |
| **Итого** | **~12–20 GB** |

Mini-вариант (500 вопросов): < 100 MB.

---

## 8. Зафиксированные отклонения от статьи

| Аспект | В статье | Здесь |
|---|---|---|
| Визуальные признаки | Официальный GQA object-feature format | Пакуется в runtime HDF5 `100×2048` |
| Graph construction | Question node + visual/textual scene graphs | Офлайн, HDF5 (prepare_gqa_data.py), relation ids сохраняются отдельно |
| Entity extraction | Вероятно NER/entity linking | Детерминированный rule-based linker с longest-match |
| Answer vocab | Paper target: 1842 classes | Детерминированный vocab по official balanced train+val |

Все отклонения и приближения задокументированы в `GAP_ANALYSIS.md` и `GQA_DATA_PROCESSING_REPORT.md`.
