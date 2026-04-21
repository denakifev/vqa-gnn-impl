# VCR на Kaggle: инструкция по подготовке и запуску VQA-GNN

Этот документ описывает полный путь от получения исходных данных VCR до запуска
обучения VQA-GNN на Kaggle. Прочитайте его перед запуском `stage_vcr_for_kaggle.sh`.

---

## Предупреждение о лицензии

**VCR данные распространяются по лицензии VCR Terms of Use:**
https://visualcommonsense.com/

Основные ограничения:
- Данные можно использовать только в некоммерческих исследовательских целях.
- Вы должны явно принять условия лицензии перед скачиванием.
- **Загружать VCR данные как публичный Kaggle Dataset НЕЛЬЗЯ.**
- Используйте только **приватные** Kaggle Datasets.

Перед загрузкой на Kaggle убедитесь, что Dataset создаётся с visibility: **private**.

---

## 1. Что нужно заранее подготовить локально

### 1.1 Данные VCR

Скачайте с официального сайта (нужна регистрация и принятие лицензии):
```
https://visualcommonsense.com/
```

Скачайте:
- `train.jsonl` (~1.5 GB)
- `val.jsonl` (~250 MB)
- `test.jsonl` (~250 MB, опционально)
- Visual features: R2C Faster RCNN features (~2048-dim) из официального VCR релиза

Положите JSONL-файлы в `data/vcr/`:
```
data/vcr/
├── train.jsonl
├── val.jsonl
└── test.jsonl   (опционально)
```

### 1.2 Визуальные признаки VCR

VCR предоставляет признаки для каждого изображения (R2C Faster RCNN, 2048-dim).
Официальный VCR репозиторий содержит инструкции по их получению.

После получения конвертируйте в HDF5 (ключ = img_fn из JSONL):
```bash
# Из .npy файлов (один файл на изображение, имя файла = img_fn):
python scripts/prepare_visual_features.py \
    --format npy \
    --input data/raw/vcr_features/ \
    --output data/vcr/visual_features/train_features.h5 \
    --num-boxes 36 --feature-dim 2048
```

> **Важно**: HDF5 ключи должны совпадать с `img_fn` из JSONL
> (например, `lsmdc_3005_MARGARET_01.05.02.00-01.05.08.00@0.jpg`).

### 1.3 ConceptNet Numberbatch

Уже есть в `data/conceptnet/`. Если нет:
```bash
wget -P data/conceptnet \
  https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
```

---

## 2. Препроцессинг: KG-Подграфы

VCR не имеет единого файла с вопросами — они встроены в JSONL. Скрипт извлекает
тексты вопросов и строит KG-подграфы, используя annot_id как ключ.

### 2.1 KG-подграфы (train)

```bash
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/train.jsonl \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --output data/vcr/knowledge_graphs/train_graphs.h5 \
  --d-kg 300
```

Время: ~2–4 часа (CPU). Ключи в HDF5 = annot_id.

С ConceptNet assertions (лучшая структура рёбер):
```bash
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/train.jsonl \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --assertions data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
  --output data/vcr/knowledge_graphs/train_graphs.h5 \
  --d-kg 300
```

### 2.2 KG-подграфы (val)

```bash
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/val.jsonl \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --output data/vcr/knowledge_graphs/val_graphs.h5 \
  --d-kg 300
```

### 2.3 Проверка покрытия визуальных признаков

```bash
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/val.jsonl \
  --validate-visual data/vcr/visual_features/val_features.h5
```

### 2.4 Валидация

```bash
python scripts/validate_vcr_data.py \
  --data-dir data/vcr \
  --split train val \
  --d-kg 300
```

---

## 3. Подготовка Kaggle Dataset (Только Private)

### 3.1 Мини-Вариант Для Sanity-Проверки

Мини-вариант содержит первые 200 аннотаций из train и val. Быстро проверяет
весь pipeline без затрат GPU-квоты.

```bash
VARIANT=mini MINI_N=200 bash scripts/stage_vcr_for_kaggle.sh
```

Создаёт: `kaggle_staging/vcr-gnn-data-mini/`

### 3.2 Полный вариант (для реального обучения)

```bash
VARIANT=full bash scripts/stage_vcr_for_kaggle.sh
```

Создаёт: `kaggle_staging/vcr-gnn-data/`

### 3.3 Редактирование Metadata

В `dataset-metadata.json` замените `KAGGLE_USERNAME` на ваш username.
Не меняйте `"name": "other"` — это корректная license label для VCR.

### 3.4 Загрузка на Kaggle (ТОЛЬКО PRIVATE)

```bash
# Первый раз (убедитесь, что dataset будет приватным):
kaggle datasets create -p kaggle_staging/vcr-gnn-data

# Обновление:
kaggle datasets version -p kaggle_staging/vcr-gnn-data -m "VCR VQA-GNN data"
```

После загрузки зайдите в настройки Dataset на Kaggle и убедитесь,
что `Visibility` установлен в **Private**.

---

## 4. Настройка Kaggle Notebook

### 4.1 Подключение датасетов

В настройках notebook подключите:
- `KAGGLE_USERNAME/vcr-gnn-data` (или `-mini` для sanity-проверки)
- `KAGGLE_USERNAME/roberta-large`

Путь к данным: `/kaggle/input/vcr-gnn-data/`

### 4.2 Установка зависимостей

```python
!pip install -q h5py comet_ml hydra-core omegaconf
!pip install -q "transformers>=4.30.0"
```

### 4.3 Переменные окружения

```python
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["COMET_API_KEY"] = "offline_placeholder"
```

---

## 5. Запуск обучения на Kaggle

### 5.1 Смоук-тест (mini-данные)

```bash
TASK=vcr \
VCR_TASK_MODE=qa \
OFFLINE=1 \
TEXT_ENCODER_PATH=/kaggle/input/roberta-large \
EPOCH_LEN=50 \
N_EPOCHS=1 \
VCR_DATA_DIR=/kaggle/input/vcr-gnn-data-mini \
bash scripts/run_kaggle_smoke_test.sh
```

Или напрямую:
```bash
python train.py --config-name baseline_vcr_qa \
  datasets=vcr_qa \
  datasets.train.data_dir=/kaggle/input/vcr-gnn-data-mini \
  datasets.val.data_dir=/kaggle/input/vcr-gnn-data-mini \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  trainer.epoch_len=50 \
  trainer.n_epochs=1 \
  trainer.save_dir=/kaggle/working/saved \
  writer.mode=offline \
  writer.run_name=smoke_vcr_qa
```

### 5.2 VCR Q→A полное обучение

```bash
python train.py --config-name baseline_vcr_qa \
  datasets=vcr_qa \
  datasets.train.data_dir=/kaggle/input/vcr-gnn-data \
  datasets.val.data_dir=/kaggle/input/vcr-gnn-data \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  dataloader.batch_size=4 \
  trainer.n_epochs=20 \
  trainer.epoch_len=5000 \
  trainer.save_dir=/kaggle/working/saved \
  writer.mode=offline \
  writer.run_name=vcr_qa_baseline
```

### 5.3 VCR QA→R (отдельная модель)

```bash
python train.py --config-name baseline_vcr_qar \
  datasets=vcr_qar \
  datasets.train.data_dir=/kaggle/input/vcr-gnn-data \
  datasets.val.data_dir=/kaggle/input/vcr-gnn-data \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  dataloader.batch_size=4 \
  trainer.n_epochs=20 \
  trainer.epoch_len=5000 \
  trainer.save_dir=/kaggle/working/saved \
  writer.mode=offline \
  writer.run_name=vcr_qar_baseline
```

> **Точность Q→AR** (joint accuracy): вычисляется вне tренировочного пайплайна.
> Требует сохранить предсказания Q→A и QA→R и объединить их.

---

## 6. Инференс на Kaggle

```bash
python inference.py --config-name inference_vcr \
  datasets.val.data_dir=/kaggle/input/vcr-gnn-data \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  inferencer.from_pretrained=/kaggle/working/saved/CHECKPOINT_DIR/best_model.pth \
  inferencer.save_path=/kaggle/working/vcr_predictions
```

---

## 7. Ожидаемый объём данных VCR

| Артефакт | Примерный размер |
|---|---|
| train.jsonl | ~1.5 GB |
| val.jsonl | ~250 MB |
| train_features.h5 (visual) | ~6–10 GB |
| val_features.h5 (visual) | ~1–2 GB |
| train_graphs.h5 (d_kg=300) | ~2–4 GB |
| val_graphs.h5 (d_kg=300) | ~0.3–0.5 GB |
| **Итого** | **~12–18 GB** |

Мини-вариант (200 аннотаций): < 50 MB.

---

## 8. Зафиксированные отклонения от статьи

| Аспект | В статье | Здесь |
|---|---|---|
| Визуальные признаки | R2C Faster RCNN features | Bottom-up HDF5 (36×2048, приближение) |
| Ссылки на объекты VCR | Привязка на уровне регионов | Замена категорией объекта (текст) |
| Построение KG | On-the-fly | Офлайн HDF5 (`prepare_vcr_data.py`) |
| KG по кандидатам | Возможно, кандидато-зависимый | Один KG на вопрос (все 4 кандидата) |
| Entity extraction | Вероятно NER | Простая токенизация |

Все отклонения задокументированы в `GAP_ANALYSIS.md`.

---

## 9. Шаги, которые нельзя автоматизировать

Следующие шаги **требуют ручных действий**:

1. Принятие лицензии VCR на visualcommonsense.com
2. Ручное скачивание VCR JSONL и visual features
3. Установка visibility Kaggle Dataset в **Private**
4. Верификация KG-подграфов перед загрузкой (`validate_vcr_data.py`)
5. Итоговая проверка предсказаний на реальных данных после обучения
