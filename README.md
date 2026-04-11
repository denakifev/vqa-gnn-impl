# VQA-GNN: воспроизводимая реализация

Открытая инженерная реализация базовой версии статьи:

> **VQA-GNN: Reasoning with Multimodal Knowledge via Graph Neural Networks for Visual Question Answering**
> ICCV 2023 / arXiv:2205.11501

Репозиторий реализует VQA-GNN внутри существующего PyTorch + Hydra шаблона.
Цель — честная, воспроизводимая, инженерно аккуратная попытка воспроизвести базовый контур
статьи. Репозиторий не утверждает полного воспроизведения численных результатов — до тех
пор, пока не проведён реальный end-to-end эксперимент с правильными артефактами данных.

**Детальный анализ расхождений:** [GAP_ANALYSIS.md](GAP_ANALYSIS.md)

---

## Бенчмарки статьи (согласовано со статьёй)

Статья VQA-GNN оценивается на двух датасетах:

| Датасет | Тип задачи | Метрики | Конфиг обучения |
|---|---|---|---|
| **VCR** (Visual Commonsense Reasoning) | multiple choice из 4 вариантов | Q→A, QA→R, Q→AR | `baseline_vcr_qa.yaml`, `baseline_vcr_qar.yaml` |
| **GQA** (Generalized VQA) | open-ended (фиксированный словарь) | accuracy (точное совпадение) | `baseline_gqa.yaml` |

**Важно:** старый VQA v2-контур (`baseline_vqa.yaml`, `VQADataset`, `VQALoss`, `VQAAccuracy`)
сохранён в репозитории, но **не является воспроизведением статьи**. Он помечен как устаревший контур.

---

## Текущее состояние

### Компоненты, согласованные со статьёй (VCR + GQA)

| Компонент | Статус | Примечание |
|---|---|---|
| `VQAGNNModel` | реализован | dense GAT; для VCR используется `num_answers=1` (конфиг `vqa_gnn_vcr.yaml`) |
| `VCRDataset` / `VCRDemoDataset` | реализован | требует VCR JSONL + HDF5 артефактов |
| `GQADataset` / `GQADemoDataset` | реализован | требует GQA JSON + HDF5 артефактов |
| `VCRLoss` | реализован | CE над 4 кандидатами |
| `GQALoss` | реализован | CE с единственным GT ответом |
| `VCRQAAccuracy` | реализована | точность Q→A |
| `VCRQARAccuracy` | реализована | точность QA→R |
| `GQAAccuracy` | реализована | точность по точному совпадению |
| `prepare_vcr_data.py` | реализован | извлечение вопросов из JSONL + KG графы |
| `prepare_gqa_data.py` | реализован | словарь ответов + KG графы |
| VCR/GQA смоук-тест (demo) | работает | проверяет пайплайн без реальных данных |
| Численные результаты статьи | **не подтверждены** | требуется реальный end-to-end запуск |

### Устаревший контур VQA v2 (не является воспроизведением статьи)

| Компонент | Статус | Примечание |
|---|---|---|
| `VQADataset` / `VQADemoDataset` | реализован | VQA v2 формат, не датасет статьи |
| `VQALoss` | реализован | soft BCE — не используется в статье |
| `VQAAccuracy` | реализована | VQA v2 метрика — не используется в статье |
| `prepare_answer_vocab.py` | реализован | для VQA v2 |
| `validate_data.py` | реализован | для VQA v2 |

---

## Что ограничивает честное воспроизведение числовых результатов статьи

- Точные веса детектора объектов из статьи не опубликованы (визуальные признаки — приближение).
- Точный пайплайн построения KG-подграфов не описан полностью; здесь используется
  ConceptNet Numberbatch с простым извлечением сущностей.
- Точный HuggingFace-чекпойнт не зафиксирован в статье; здесь принят `roberta-large`.
- Числовые результаты статьи не подтверждены реальным end-to-end запуском.
- VCR: grounding ссылок на объекты (ссылки типа `[0, 2]` → "person car") реализован
  как замена категорией объекта, а не полноценным region-level grounding.

## Основные отличия от статьи

| Аспект | В статье | В этой реализации |
|---|---|---|
| **Бенчмарки** | VCR + GQA | VCR + GQA (согласовано со статьёй); VQA v2 — устаревший контур |
| Языковой энкодер | предобученная RoBERTa | `roberta-large` по умолчанию |
| Реализация GNN | недоступна публично | dense GAT (функционально эквивалентен) |
| Построение KG | on-the-fly из внешнего KG | офлайн, заранее подготовленные HDF5 |
| KG-граф | специальный пайплайн статьи | ConceptNet 5.7 + Numberbatch; структура приближённая |
| Визуальные признаки | детектор, специфичный для статьи | ожидаются готовые bottom-up features |
| VCR object references | region-level grounding | замена категорией объекта (приближение) |
| Числа из статьи | заявлены | не подтверждены в этом репозитории |

---

## Установка

```bash
pip install -r requirements.txt
```

Ключевые зависимости:

- `transformers>=4.30.0` — языковой энкодер вопроса
- `h5py>=3.8.0` — чтение HDF5 артефактов
- `comet_ml` — трекер экспериментов

---

## Быстрый старт: смоук-тест (без реальных данных)

```bash
# VCR Q→A смоук-тест (синтетические данные):
python train.py --config-name baseline_vcr_qa

# VCR QA→R смоук-тест:
python train.py --config-name baseline_vcr_qar

# GQA смоук-тест:
python train.py --config-name baseline_gqa
```

Все три команды проверяют полный пайплайн на синтетических данных.
Результаты **не воспроизводят статью** — только убеждаются, что код запускается.

---

## Запуск обучения, согласованного со статьёй (VCR + GQA)

### VCR Q→A

```bash
python train.py --config-name baseline_vcr_qa \
  datasets=vcr_qa \
  datasets.train.data_dir=data/vcr \
  datasets.val.data_dir=data/vcr
```

### VCR QA→R (отдельная модель)

```bash
python train.py --config-name baseline_vcr_qar \
  datasets=vcr_qar \
  datasets.train.data_dir=data/vcr \
  datasets.val.data_dir=data/vcr
```

**Точность Q→AR** вычисляется как совместная точность двух отдельных моделей
(Q→A и QA→R). Требует сохранить предсказания обоих запусков и объединить их.

### GQA

```bash
python train.py --config-name baseline_gqa \
  datasets=gqa \
  datasets.train.data_dir=data/gqa \
  datasets.train.answer_vocab_path=data/gqa/gqa_answer_vocab.json \
  datasets.val.data_dir=data/gqa \
  datasets.val.answer_vocab_path=data/gqa/gqa_answer_vocab.json
```

---

## Подготовка данных (контур, согласованный со статьёй: VCR и GQA)

### VCR

**Получение данных VCR** (требует согласия с лицензией):
- Скачайте аннотации с https://visualcommonsense.com
  (файлы: `train.jsonl`, `val.jsonl`, `test.jsonl`)
- Скачайте визуальные признаки (R2C Faster RCNN features, 2048-dim)
  из официального релиза VCR

Положите в `data/vcr/`:
```text
data/vcr/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── visual_features/
│   ├── train_features.h5   # HDF5: img_fn → float32[N_v, 2048]
│   └── val_features.h5
└── knowledge_graphs/       # создаётся скриптом ниже
    ├── train_graphs.h5
    └── val_graphs.h5
```

**Построение KG-подграфов для VCR:**

```bash
# Train split:
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/train.jsonl \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --output data/vcr/knowledge_graphs/train_graphs.h5 \
  --d-kg 300

# Val split:
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/val.jsonl \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --output data/vcr/knowledge_graphs/val_graphs.h5 \
  --d-kg 300

# Проверить покрытие визуальных признаков:
python scripts/prepare_vcr_data.py \
  --jsonl data/vcr/val.jsonl \
  --validate-visual data/vcr/visual_features/val_features.h5
```

### GQA

**Получение данных GQA** (требует согласия с лицензией):
- Скачайте balanced split questions с https://cs.stanford.edu/people/dorarad/gqa/download.html
  (файлы: `train_balanced_questions.json`, `val_balanced_questions.json`)
- Скачайте визуальные признаки (Visual Genome based)

Положите в `data/gqa/`:
```text
data/gqa/
├── questions/
│   ├── train_balanced_questions.json
│   └── val_balanced_questions.json
├── visual_features/
│   ├── train_features.h5   # HDF5: imageId → float32[N_v, 2048]
│   └── val_features.h5
├── knowledge_graphs/       # создаётся скриптом ниже
│   ├── train_graphs.h5
│   └── val_graphs.h5
└── gqa_answer_vocab.json   # создаётся скриптом ниже
```

**Построение артефактов GQA:**

```bash
# Шаг 1: Построить словарь ответов (top-1852 из train):
python scripts/prepare_gqa_data.py \
  --build-vocab \
  --questions data/gqa/questions/train_balanced_questions.json \
  --output-vocab data/gqa/gqa_answer_vocab.json \
  --top-n 1852

# Шаг 2: KG-подграфы (train):
python scripts/prepare_gqa_data.py \
  --build-graphs \
  --questions data/gqa/questions/train_balanced_questions.json \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --output data/gqa/knowledge_graphs/train_graphs.h5 \
  --d-kg 300

# Шаг 3: KG-подграфы (val):
python scripts/prepare_gqa_data.py \
  --build-graphs \
  --questions data/gqa/questions/val_balanced_questions.json \
  --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
  --output data/gqa/knowledge_graphs/val_graphs.h5 \
  --d-kg 300
```

### ConceptNet Numberbatch (нужен для VCR и GQA)

```bash
mkdir -p data/conceptnet
wget -P data/conceptnet \
  https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz

# Опционально, для лучшей структуры рёбер:
wget -P data/conceptnet \
  https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
```

---

## Подготовка данных (устаревший контур: VQA v2)

Для VQA v2 нужны четыре группы артефактов.
Все скрипты подготовки лежат в `scripts/`.

### Ожидаемая структура каталогов

```text
data/vqa/
├── questions/
│   ├── train_questions.json      # VQA v2 questions
│   └── val_questions.json
├── annotations/
│   ├── train_annotations.json    # VQA v2 annotations (train/val only)
│   └── val_annotations.json
├── visual_features/
│   ├── train_features.h5         # HDF5: str(image_id) → float32[36, 2048]
│   └── val_features.h5
├── knowledge_graphs/
│   ├── train_graphs.h5           # HDF5: str(question_id) → {node_features, adj_matrix, node_types}
│   └── val_graphs.h5
└── answer_vocab.json             # {"answer_to_idx": {"yes": 0, ...}}
```

### Шаг 0: Получение исходных данных VQA v2

Скачайте файлы с https://visualqa.org/download.html вручную (требует согласия с лицензией).
Переименуйте после скачивания:

```text
v2_OpenEnded_mscoco_train2014_questions.json    → train_questions.json
v2_OpenEnded_mscoco_val2014_questions.json      → val_questions.json
v2_mscoco_train2014_annotations.json           → train_annotations.json
v2_mscoco_val2014_annotations.json             → val_annotations.json
```

Положите в `data/vqa/questions/` и `data/vqa/annotations/`.

### Шаг 1: Словарь ответов

```bash
python scripts/prepare_answer_vocab.py \
    --annotations data/vqa/annotations/train_annotations.json \
    --output data/vqa/answer_vocab.json \
    --top-n 3129
```

Параметры:
- `--top-n 3129` — стандартный размер словаря для VQA v2 (по умолчанию)
- `--annotations` — можно передать несколько файлов (train + val)

Выходной формат: `{"answer_to_idx": {"yes": 0, "no": 1, ...}}`

### Шаг 2: Визуальные признаки

Репозиторий не включает извлечение признаков детектором объектов. Нужны
заранее извлечённые bottom-up attention features.

Источники, совместимые с этим репозиторием:
- [Bottom-up attention official release](https://github.com/peteanderson80/bottom-up-attention)
  (36 regions per image, ResNet-101, 2048-dim)
- Адаптации от [OSCAR](https://github.com/microsoft/Oscar) или [ViLBERT](https://github.com/facebookresearch/vilbert-multi-task)

После получения признаков — конвертируйте в HDF5:

```bash
# Из TSV (официальный формат bottom-up attention):
python scripts/prepare_visual_features.py \
    --format tsv \
    --input data/raw/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv \
    --output data/vqa/visual_features/train_features.h5

# Из директории с .npy-файлами (по одному на изображение):
python scripts/prepare_visual_features.py \
    --format npy \
    --input data/raw/train_features_npy/ \
    --output data/vqa/visual_features/train_features.h5

# Из .npz-файлов:
python scripts/prepare_visual_features.py \
    --format npz \
    --input data/raw/train_features_npz/ \
    --output data/vqa/visual_features/train_features.h5 \
    --npz-key features

# Проверить существующий HDF5:
python scripts/prepare_visual_features.py \
    --format h5 \
    --input data/vqa/visual_features/train_features.h5 \
    --output data/vqa/visual_features/train_features.h5 \
    --verify-only
```

Если ваши признаки имеют другую форму (`d_visual ≠ 2048` или `num_boxes ≠ 36`),
обновите параметры в конфигах:
- `src/configs/model/vqa_gnn.yaml` → `d_visual`
- `src/configs/datasets/vqa.yaml` → `d_visual`, `num_visual_nodes`

### Шаг 3: Подграфы графа знаний

Это наиболее трудоёмкий шаг и главное ограничение текущей реализации.

#### 3.1 Скачайте ConceptNet Numberbatch (обязательно)

```bash
mkdir -p data/conceptnet
cd data/conceptnet

# English-only embeddings (~450MB compressed, ~1.5GB uncompressed)
wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
```

#### 3.2 Скачайте ConceptNet assertions (опционально, для лучшей структуры рёбер)

```bash
# ConceptNet 5.7 assertions (~500MB compressed)
wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
```

Без assertions KG-узлы будут полностью связаны внутри — это задокументированное
приближение к оригинальной статье.

#### 3.3 Постройте графы

**Важно:** размерность `--d-kg` должна совпадать с `d_kg` в конфигах.
По умолчанию скрипт использует 300-dim (полный Numberbatch). Конфиги по умолчанию
используют 100 (для demo). Перед запуском обновите конфиги или используйте 100 с
явной usечением Numberbatch.

Вариант A — 300-dim (рекомендуется, нужно обновить конфиги):
```bash
# Обновить d_kg в конфигах перед этим:
# src/configs/model/vqa_gnn.yaml: d_kg: 300
# src/configs/datasets/vqa.yaml:  d_kg: 300
# src/configs/datasets/vqa_eval.yaml: d_kg: 300

python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/train_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --assertions data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
    --output data/vqa/knowledge_graphs/train_graphs.h5 \
    --d-kg 300

python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/val_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --assertions data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
    --output data/vqa/knowledge_graphs/val_graphs.h5 \
    --d-kg 300
```

Вариант B — 100-dim (подходит к demo-конфигам, применяется усечение):
```bash
python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/train_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --output data/vqa/knowledge_graphs/train_graphs.h5 \
    --d-kg 100

python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/val_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --output data/vqa/knowledge_graphs/val_graphs.h5 \
    --d-kg 100
```

#### Схема KG-подграфа

HDF5 ключ: `str(question_id)`

Внутри каждой группы:
```
node_features:  float32[N_kg, d_kg]         KG node embeddings
adj_matrix:     float32[N_total, N_total]    full adjacency matrix
node_types:     int32[N_total]              0=visual, 1=question, 2=kg

N_total = num_visual_nodes + 1 + N_kg  (N_kg ≤ max_kg_nodes)
```

### Шаг 4: Проверка данных

Перед запуском обучения, особенно на Kaggle. Значение `--d-kg` должно
совпадать с тем, что использовалось в шаге 3.

Вариант A (300-dim, нужно обновить конфиги):
```bash
python scripts/validate_data.py \
    --data-dir data/vqa \
    --split train val \
    --answer-vocab data/vqa/answer_vocab.json \
    --num-visual-nodes 36 \
    --feature-dim 2048 \
    --d-kg 300 \
    --num-answers 3129 \
    --max-kg-nodes 30
```

Вариант B (100-dim, подходит к demo-конфигам без изменений):
```bash
python scripts/validate_data.py \
    --data-dir data/vqa \
    --split train val \
    --answer-vocab data/vqa/answer_vocab.json \
    --num-visual-nodes 36 \
    --feature-dim 2048 \
    --d-kg 100 \
    --num-answers 3129 \
    --max-kg-nodes 30
```

Скрипт проверит:
- существование всех обязательных файлов
- формат answer_vocab.json
- покрытие `image_id` в HDF5 с визуальными признаками
- покрытие question_id в KG graphs HDF5
- shape, dtype, структуру групп

Если выход не равен 0 — исправить ошибки перед обучением.

---

## Запуск обучения

### Demo смоук-тест (без реальных данных)

Проверяет, что полный пайплайн обучения работает на синтетических данных.
Не воспроизводит результаты статьи.

```bash
python train.py --config-name baseline_vqa
```

### Реальное обучение на VQA v2

```bash
python train.py --config-name baseline_vqa \
    datasets=vqa \
    datasets.train.data_dir=data/vqa \
    datasets.train.answer_vocab_path=data/vqa/answer_vocab.json \
    datasets.val.data_dir=data/vqa \
    datasets.val.answer_vocab_path=data/vqa/answer_vocab.json
```

Если используете d_kg=300, добавьте:
```bash
    datasets.train.d_kg=300 \
    datasets.val.d_kg=300 \
    model.d_kg=300
```

### Полезные переопределения

Для `roberta-large` на Kaggle почти всегда стоит отдельно подбирать
`dataloader.batch_size`. Практически безопасная отправная точка:

```bash
python train.py --config-name baseline_vqa \
    dataloader.batch_size=4
```

```bash
# Заморозить text encoder (быстрее, меньше памяти GPU):
python train.py --config-name baseline_vqa model.freeze_text_encoder=True

# Вернуться к более лёгкому энкодеру, если не хватает памяти:
python train.py --config-name baseline_vqa \
    model.text_encoder_name=bert-base-uncased \
    model.d_hidden=768 \
    datasets.train.text_encoder_name=bert-base-uncased \
    datasets.val.text_encoder_name=bert-base-uncased

# Продолжить с чекпойнта:
python train.py --config-name baseline_vqa \
    trainer.resume_from=checkpoint-epoch10.pth \
    trainer.override=True
```

---

## Запуск инференса

### Demo смоук-тест инференса

```bash
python inference.py --config-name inference_vqa \
    inferencer.skip_model_load=True
```

### Реальный инференс

```bash
python inference.py --config-name inference_vqa \
    datasets=vqa_eval \
    datasets.val.data_dir=data/vqa \
    datasets.val.answer_vocab_path=data/vqa/answer_vocab.json \
    inferencer.from_pretrained=saved/vqa_gnn_baseline/model_best.pth
```

Предсказания сохраняются в:
```text
data/saved/vqa_preds/val/output_*.pth
```

Каждый файл: `{"pred_label": int, "label": int}`

---

## Запуск на Kaggle

### Что подготовить заранее (оффлайн)

Артефакты, которые нельзя подготовить прямо в Kaggle notebook из-за размера
или лицензионных ограничений, нужно подготовить заранее и загрузить как Kaggle Dataset:

1. Данные VQA v2 (questions + annotations)
2. HDF5 с визуальными признаками
3. KG graphs HDF5
4. answer_vocab.json
5. HuggingFace кэш модели (если интернет отключён)

Загрузите всё в один Kaggle Dataset (например, `vqa-gnn-data`) и подключите к notebook.

Ожидаемая структура в корне Kaggle Dataset (`/kaggle/input/vqa-gnn-data/`):
```text
/kaggle/input/vqa-gnn-data/
├── questions/
│   ├── train_questions.json
│   └── val_questions.json
├── annotations/
│   ├── train_annotations.json
│   └── val_annotations.json
├── visual_features/
│   ├── train_features.h5
│   └── val_features.h5
├── knowledge_graphs/
│   ├── train_graphs.h5
│   └── val_graphs.h5
└── answer_vocab.json
```

### Demo смоук-тест на Kaggle

> **Внимание:** demo смоук-тест использует `VQAGNNModel` с `roberta-large` (≈355M
> параметров). При первом запуске Hugging Face скачает `roberta-large` (~1.4 GB).
> Если интернет на Kaggle **отключён**, сначала загрузите `roberta-large` в Kaggle
> Dataset и используйте команду из раздела
> [«Kaggle без интернета»](#kaggle-без-интернета-offline-bertroberta).

```bash
python train.py --config-name baseline_vqa \
    trainer.save_dir=/kaggle/working/saved
```

### Реальное обучение на Kaggle

Если интернет в Kaggle **включён** (и RoBERTa может скачаться автоматически):

```bash
python train.py --config-name baseline_vqa \
    datasets=vqa \
    datasets.train.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.train.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    trainer.save_dir=/kaggle/working/saved
```

Если интернет **отключён** — используйте полную команду из раздела
[«Kaggle без интернета»](#kaggle-без-интернета-offline-bertroberta) ниже.

### Инференс на Kaggle

```bash
python inference.py --config-name inference_vqa \
    datasets=vqa_eval \
    datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    inferencer.from_pretrained=/kaggle/input/vqa-gnn-ckpt/model_best.pth \
    inferencer.save_path=vqa_preds
```

Предсказания будут в:
```text
<repo_root>/data/saved/vqa_preds/val/
```

Если репозиторий в `/kaggle/working/paper_model_replication`:
```text
/kaggle/working/paper_model_replication/data/saved/vqa_preds/val/
```

### Kaggle без интернета (офлайн RoBERTa / HF-энкодер)

```bash
# Один раз (при включённом интернете):
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('roberta-large').save_pretrained('/kaggle/working/roberta-large')
AutoModel.from_pretrained('roberta-large').save_pretrained('/kaggle/working/roberta-large')
"
# → загрузите /kaggle/working/roberta-large как Kaggle Dataset

# На основном запуске (интернет выключен):
TRANSFORMERS_OFFLINE=1 python train.py --config-name baseline_vqa \
    datasets=vqa \
    model.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.train.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.val.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.train.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.train.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    trainer.save_dir=/kaggle/working/saved
```

---

## CometML

CometML — основной трекер экспериментов в этом репозитории.

### Онлайн-режим

```bash
export COMET_API_KEY=your_key_here
python train.py --config-name baseline_vqa \
    writer.project_name=vqa_gnn \
    writer.run_name=my_run
```

### Оффлайн-режим (Kaggle без сети)

> **Важно:** CometML вызывает `comet_ml.login()` даже в оффлайн-режиме.
> Переменная `COMET_API_KEY` должна быть установлена, иначе обучение упадёт.
> Для оффлайн-экспериментов достаточно любого значения (фактическая аутентификация
> происходит только при последующей загрузке):

```bash
export COMET_API_KEY=your_key_here   # или любой непустой placeholder
python train.py --config-name baseline_vqa \
    writer.mode=offline \
    writer.run_name=kaggle_run_1
```

Оффлайн-логи сохраняются в директорию запуска (`~/.cometml-runs/` по умолчанию).
Загрузить в CometML после получения доступа к интернету:
```bash
comet upload ~/.cometml-runs/<experiment_key>
```

### Полезные переопределения

```bash
python train.py --config-name baseline_vqa \
    writer.project_name=vqa_gnn \
    writer.run_name=kaggle_run_1 \
    writer.workspace=my_workspace
```

---

## Артефакты воспроизводимости

Каждый запуск сохраняет:

```text
saved/<run_name>/
├── config.yaml           # полный Hydra конфиг
├── git_commit.txt        # текущий commit hash
├── git_diff.patch        # незакоммиченные изменения
├── checkpoint-epochN.pth # периодические чекпойнты
└── model_best.pth        # лучший чекпойнт по val_VQA_Accuracy
```

Для максимально честного воспроизведения сохраняйте:
- `config.yaml` из конкретного запуска
- конкретный набор артефактов данных (question files, vocab, feature version)
- версии всех зависимостей (`pip freeze`)

---

## Структура репозитория

```text
.
├── train.py                               # точка входа обучения
├── inference.py                           # точка входа инференса
├── GAP_ANALYSIS.md                        # анализ расхождений с бумагой
├── requirements.txt
├── README.md
├── scripts/
│   ├── prepare_vcr_data.py                # [PAPER] VCR KG графы из JSONL
│   ├── prepare_gqa_data.py                # [PAPER] GQA словарь ответов + KG графы
│   ├── prepare_answer_vocab.py            # [устаревший VQA v2] словарь ответов
│   ├── prepare_visual_features.py         # конвертация признаков в HDF5
│   ├── prepare_kg_graphs.py               # KG подграфы из ConceptNet
│   └── validate_data.py                   # проверка артефактов VQA v2
└── src/
    ├── model/
    │   └── vqa_gnn.py                     # VQAGNNModel (VCR + GQA + VQA v2)
    ├── datasets/
    │   ├── vcr_dataset.py                 # [PAPER] VCRDataset, VCRDemoDataset
    │   ├── vcr_collate.py                 # [PAPER] make_vcr_collate_fn (B→B×4)
    │   ├── gqa_dataset.py                 # [PAPER] GQADataset, GQADemoDataset
    │   ├── gqa_collate.py                 # [PAPER] gqa_collate_fn
    │   ├── vqa_dataset.py                 # [legacy] VQADataset, VQADemoDataset
    │   └── vqa_collate.py                 # [legacy] vqa_collate_fn
    ├── loss/
    │   ├── vcr_loss.py                    # [PAPER] VCRLoss (CE over 4 candidates)
    │   ├── gqa_loss.py                    # [PAPER] GQALoss (hard CE)
    │   └── vqa_loss.py                    # [legacy] VQALoss (soft BCE)
    ├── metrics/
    │   ├── vcr_metric.py                  # [PAPER] VCRQAAccuracy, VCRQARAccuracy
    │   ├── gqa_metric.py                  # [PAPER] GQAAccuracy (точное совпадение)
    │   └── vqa_metric.py                  # [legacy] VQAAccuracy (VQA v2 soft)
    └── configs/
        ├── baseline_vcr_qa.yaml           # [PAPER] VCR Q→A training
        ├── baseline_vcr_qar.yaml          # [PAPER] VCR QA→R training
        ├── baseline_gqa.yaml              # [PAPER] GQA training
        ├── inference_vcr.yaml             # [PAPER] VCR inference
        ├── inference_gqa.yaml             # [PAPER] GQA inference
        ├── baseline_vqa.yaml              # [legacy] VQA v2 training
        ├── inference_vqa.yaml             # [legacy] VQA v2 inference
        ├── model/
        │   ├── vqa_gnn_vcr.yaml           # [PAPER] num_answers=1 for VCR
        │   ├── vqa_gnn_gqa.yaml           # [PAPER] num_answers=1852 for GQA
        │   └── vqa_gnn.yaml               # [legacy] num_answers=3129 for VQA v2
        ├── datasets/
        │   ├── {vcr_qa,vcr_qar,vcr_demo,vcr_demo_qar}.yaml
        │   ├── {gqa,gqa_eval,gqa_demo}.yaml
        │   └── {vqa,vqa_eval,vqa_demo,vqa_demo_eval}.yaml
        ├── metrics/{vcr_qa,vcr_qar,gqa,vqa}.yaml
        ├── dataloader/{vcr,gqa,vqa}.yaml
        └── transforms/vqa.yaml
```

---

## Что стоит сделать перед попыткой честно повторить числа из статьи

1. Обеспечить использование тех же визуальных признаков, что и в статье
   (R2C Faster RCNN для VCR, VG-совместимые признаки для GQA).
2. Зафиксировать text encoder `roberta-large`, согласованный со статьёй.
3. Провести полный train/val эксперимент на VCR и GQA с правильными артефактами.
4. Сравнить метрики и задокументировать расхождения в [GAP_ANALYSIS.md](GAP_ANALYSIS.md).

---

## Цитирование

```bibtex
@inproceedings{vqagnn2023,
  title     = {VQA-GNN: Reasoning with Multimodal Knowledge via Graph Neural Networks
               for Visual Question Answering},
  booktitle = {ICCV},
  year      = {2023},
  note      = {arXiv:2205.11501}
}
```
