# Практический регламент запуска VQA-GNN

Пошаговый практический план запуска VQA-GNN с максимальной воспроизводимостью.
Целевая среда обучения: Kaggle. Подготовка данных: локально, Colab или удалённая VM.

---

## 1. Краткий аудит готовности к запуску

### Что готово

| Компонент | Статус | Примечание |
|---|---|---|
| `VQAGNNModel` | готов | dense GAT, `roberta-large` по умолчанию |
| `VQADataset` | готов | ориентирован на HDF5, lazy open |
| `VQADemoDataset` | готов | смоук-тест без реальных данных |
| `VQALoss`, `VQAAccuracy` | готовы | soft BCE, официальная метрика VQA v2 |
| `scripts/prepare_*.py` | готовы | все 4 скрипта реализованы |
| `baseline_vqa.yaml`, `inference_vqa.yaml` | готовы | покрывают полный цикл |
| CometML writer | готов | онлайн- и офлайн-режимы |
| Чекпойнты, git patch, config.yaml | сохраняются | существующая инфраструктура |

### Найденные несогласованности

**Несогласованность 1 — default config_name:**
`train.py` и `inference.py` используют `config_name="baseline"` и `config_name="inference"` соответственно.
VQA-конфиги называются `baseline_vqa` и `inference_vqa`.
**Всегда передавать `--config-name baseline_vqa` явно.**

**Несогласованность 2 — d_kg по умолчанию:**
`src/configs/model/vqa_gnn.yaml` и `src/configs/datasets/vqa.yaml` имеют `d_kg: 100` (demo).
`scripts/prepare_kg_graphs.py` имеет `--d-kg 300` по умолчанию.
Нужно явно выбрать одно значение и синхронизировать во всех трёх местах перед препроцессингом.

**Несогласованность 3 — batch_size:**
`src/configs/dataloader/vqa.yaml` — `batch_size: 32`.
При `roberta-large` (355M params) и `d_hidden=1024` это вызовет OOM на T4/P100.
Kaggle-безопасная точка: `batch_size=4`.

### Главные внешние блокеры

| Блокер | Серьёзность | Обходной путь |
|---|---|---|
| Визуальные признаки (36×2048 bottom-up) | **критический** | нет; нужны заранее извлечённые признаки |
| Точный детектор из статьи не опубликован | средний | официальный релиз Anderson et al. CVPR 2018 |
| KG-пайплайн препроцессинга статьи не полностью описан | средний | ConceptNet Numberbatch (задокументировано) |
| Численные результаты не подтверждены | информационный | честно документируем |

### Оценка объёма данных

| Артефакт | В сжатом виде | В распакованном виде |
|---|---|---|
| VQA v2 questions + annotations | ~250 MB | ~275 MB |
| Bottom-up TSV (train+val combined) | ~10–15 GB | ~27 GB |
| `train_features.h5` (36×2048, gzip) | ~6–9 GB | ~24 GB |
| `val_features.h5` (36×2048, gzip) | ~3–5 GB | ~12 GB |
| ConceptNet Numberbatch (английская версия) | ~450 MB | ~1.5 GB |
| ConceptNet Assertions | ~500 MB | ~3 GB |
| `train_graphs.h5` (d_kg=300) | ~2–4 GB | ~8 GB |
| `val_graphs.h5` (d_kg=300) | ~1–2 GB | ~4 GB |
| `roberta-large` checkpoint | ~1.4 GB | ~1.4 GB |
| **Итого в Kaggle Dataset** | **~16–25 GB** | — |

---

## 2. Рекомендуемый путь

Если хочешь минимизировать риски и максимально использовать бесплатные ресурсы:

**Этап A — Подготовка (локальная машина или удалённая VM, 1–2 дня):**
Скачай VQA v2 JSON + bottom-up TSV → конвертируй визуальные признаки в HDF5 →
собери KG-графы → провалидируй → упакуй Kaggle Datasets.

**Этап B — Обучение (Kaggle, 12h GPU run):**
Клонируй репо → подключи datasets → запусти обучение с `batch_size=4`,
`freeze_text_encoder=False` → сохрани checkpoint.

**Этап C — Инференс и результаты (Kaggle):**
Запусти inference на val → получи VQA Accuracy → задокументируй расхождение со статьёй.

**Если нет локальной машины:** Colab Pro с Google Drive для хранения артефактов → Kaggle.
Главный риск — 12h лимит на самых тяжёлых шагах (конвертация визуальных признаков).
Решение: запускать на ночь или использовать Google Drive для промежуточного хранения.

---

## 3. Полный пошаговый план

### Этап 0: Подготовка окружения

**Цель:** рабочее Python-окружение с зависимостями.
**Где:** локальная машина или Colab.

```bash
git clone <your-repo-url> paper_model_replication
cd paper_model_replication

pip install -r requirements.txt

# Если torch==2.8.0 не найдётся в pip:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics numpy==1.26.4 tqdm matplotlib pandas wandb comet_ml hydra-core \
    "transformers>=4.30.0" "h5py>=3.8.0"

# Проверка установки
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from transformers import AutoTokenizer; print('transformers OK')"
python -c "import h5py; print('h5py OK')"
python -c "import comet_ml; print('comet_ml OK')"

# Создать структуру директорий
mkdir -p data/vqa/questions
mkdir -p data/vqa/annotations
mkdir -p data/vqa/visual_features
mkdir -p data/vqa/knowledge_graphs
mkdir -p data/conceptnet
```

---

### Этап 1: Получение исходных данных VQA v2

**Цель:** question JSON и annotation JSON.
**Где:** локально (требует ручного скачивания + принятия лицензии).
**Объём:** ~250 MB суммарно.

Скачать вручную с https://visualqa.org/download.html:

```bash
# После скачивания переименовать и переложить:
mv v2_OpenEnded_mscoco_train2014_questions.json data/vqa/questions/train_questions.json
mv v2_OpenEnded_mscoco_val2014_questions.json   data/vqa/questions/val_questions.json
mv v2_mscoco_train2014_annotations.json         data/vqa/annotations/train_annotations.json
mv v2_mscoco_val2014_annotations.json           data/vqa/annotations/val_annotations.json

# Базовая проверка
python -c "
import json
for p in ['data/vqa/questions/train_questions.json', 'data/vqa/annotations/train_annotations.json']:
    with open(p) as f:
        d = json.load(f)
    key = 'questions' if 'questions' in d else 'annotations'
    print(p, '->', len(d[key]), 'entries')
"
# Ожидать: train questions → 443757, train annotations → 443757
```

---

### Этап 2: Подготовка словаря ответов

**Цель:** `answer_vocab.json` с 3129 нормализованными ответами.
**Где:** локально или Colab. ~1 минута. Только CPU.

```bash
python scripts/prepare_answer_vocab.py \
    --annotations data/vqa/annotations/train_annotations.json \
    --output data/vqa/answer_vocab.json \
    --top-n 3129

# Проверка
python -c "
import json
with open('data/vqa/answer_vocab.json') as f:
    v = json.load(f)
print('Vocab size:', len(v['answer_to_idx']))
print('First 5:', list(v['answer_to_idx'].items())[:5])
"
# Ожидать: Vocab size: 3129
```

---

### Этап 3: Подготовка визуальных признаков

**Цель:** `train_features.h5` и `val_features.h5`.
**Где:** локально или удалённая VM (требует ~15–30 GB диска).
**Время:** 30–60 минут на конвертацию train.

**Нельзя делать в репозитории:** скрипт конвертирует существующие features, но не извлекает их из изображений.
Нужны заранее извлечённые bottom-up attention features из внешнего источника.

**Источник A — официальный TSV (рекомендуется):**

Файл `trainval_resnet101_faster_rcnn_genome_36.tsv` (~27 GB) содержит features для train+val images.
Ссылки на скачивание актуальны в репозитории https://github.com/peteanderson80/bottom-up-attention.

```bash
# Конвертация всего TSV в один общий HDF5 (содержит train+val images):
python scripts/prepare_visual_features.py \
    --format tsv \
    --input data/raw/trainval_resnet101_faster_rcnn_genome_36.tsv \
    --output data/vqa/visual_features/all_features.h5 \
    --num-boxes 36 \
    --feature-dim 2048 \
    --compression gzip

# `VQADataset` читает по `image_id`, поэтому один общий HDF5 подходит для обоих split:
ln -s "$(pwd)/data/vqa/visual_features/all_features.h5" data/vqa/visual_features/train_features.h5
ln -s "$(pwd)/data/vqa/visual_features/all_features.h5" data/vqa/visual_features/val_features.h5
```

**Источник B — директория .npy файлов (один файл на изображение):**

```bash
python scripts/prepare_visual_features.py \
    --format npy \
    --input data/raw/train_npy/ \
    --output data/vqa/visual_features/train_features.h5 \
    --num-boxes 36 --feature-dim 2048

python scripts/prepare_visual_features.py \
    --format npy \
    --input data/raw/val_npy/ \
    --output data/vqa/visual_features/val_features.h5 \
    --num-boxes 36 --feature-dim 2048
```

**Верификация:**

```bash
python scripts/prepare_visual_features.py \
    --format h5 \
    --input data/vqa/visual_features/train_features.h5 \
    --output data/vqa/visual_features/train_features.h5 \
    --verify-only \
    --num-boxes 36 \
    --feature-dim 2048
```

**Если d_visual != 2048**, обновить перед обучением:

```yaml
# src/configs/model/vqa_gnn.yaml:  d_visual: <your_dim>
# src/configs/datasets/vqa.yaml:   d_visual: <your_dim>
```

---

### Этап 4: Подготовка KG-графов

**Цель:** `train_graphs.h5` и `val_graphs.h5`.
**Где:** локально или Colab. Только CPU. ~2–4 часа для train.

**Важно: выбрать `d_kg` ОДИН РАЗ до начала препроцессинга и синхронизировать с конфигами.**

**Вариант 300-dim (рекомендуется — полный Numberbatch, нужно обновить конфиги):**

```bash
# Обновить d_kg в трёх файлах конфигурации:
sed -i 's/d_kg: 100/d_kg: 300/g' src/configs/model/vqa_gnn.yaml
sed -i 's/d_kg: 100/d_kg: 300/g' src/configs/datasets/vqa.yaml
sed -i 's/d_kg: 100/d_kg: 300/g' src/configs/datasets/vqa_eval.yaml
```

**Вариант 100-dim (быстрее и меньше места, конфиги менять не нужно):**
Передать `--d-kg 100` в скриптах ниже и не трогать YAML.

**Скачать ConceptNet:**

```bash
# English Numberbatch (~450MB compressed, обязательно):
wget -P data/conceptnet \
    "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"

# ConceptNet assertions (~500MB compressed, опционально — улучшает структуру рёбер):
wget -P data/conceptnet \
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
```

**Собрать графы для train (с assertions — рекомендуется):**

```bash
python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/train_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --assertions data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
    --output data/vqa/knowledge_graphs/train_graphs.h5 \
    --d-kg 300 \
    --max-kg-nodes 30 \
    --num-visual-nodes 36
```

**Собрать графы для val:**

```bash
python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/val_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --assertions data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
    --output data/vqa/knowledge_graphs/val_graphs.h5 \
    --d-kg 300 \
    --max-kg-nodes 30 \
    --num-visual-nodes 36
```

**Без assertions (быстрее, KG subgraph будет полностью связным — задокументированное приближение):**

```bash
python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/train_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --output data/vqa/knowledge_graphs/train_graphs.h5 \
    --d-kg 300
```

**Debug-проверка на малом subset (1–2 минуты):**

```bash
python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/train_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --output /tmp/train_graphs_debug.h5 \
    --d-kg 300 \
    --limit 1000
```

**Памятка по ресурсам:**
- Загрузка Numberbatch в RAM: ~1.5 GB; с assertions: ещё ~2 GB.
- Colab Free (13 GB RAM) выдержит для d_kg=300 без assertions. С assertions — на грани.
- При прерывании сессии Colab незаписанные в HDF5 вопросы теряются. Используй nohup или Colab Pro.

---

### Этап 5: Валидация всех артефактов

**Цель:** убедиться, что все артефакты корректны до upload на Kaggle.
**Где:** там же, где этап 4. Запускать ОБЯЗАТЕЛЬНО до загрузки.

```bash
python scripts/validate_data.py \
    --data-dir data/vqa \
    --split train val \
    --answer-vocab data/vqa/answer_vocab.json \
    --num-visual-nodes 36 \
    --feature-dim 2048 \
    --d-kg 300 \
    --num-answers 3129 \
    --max-kg-nodes 30 \
    --sample-size 200

# Для d_kg=100 заменить --d-kg 300 на --d-kg 100
# Критерий: exit code 0, все строки [OK].
# Если exit code 1 — исправить ошибки до продолжения.
```

---

### Этап 6: Подготовка `roberta-large` для офлайн-режима

**Где:** в Kaggle с включённым интернетом (один раз) или локально.

```bash
# Локально:
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('roberta-large').save_pretrained('data/roberta-large')
AutoModel.from_pretrained('roberta-large').save_pretrained('data/roberta-large')
print('Saved to data/roberta-large (~1.4 GB)')
"
# Загрузить data/roberta-large как отдельный Kaggle Dataset (hf-roberta-large)
```

```python
# Или в Kaggle notebook при включённом интернете:
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("roberta-large").save_pretrained("/kaggle/working/roberta-large")
AutoModel.from_pretrained("roberta-large").save_pretrained("/kaggle/working/roberta-large")
# Затем: New Dataset → Upload /kaggle/working/roberta-large
```

---

## 4. План публикации в Kaggle

### Структура Kaggle Datasets

```
Dataset 1: <username>/vqa-gnn-data   (~10–20 GB)
    questions/
        train_questions.json
        val_questions.json
    annotations/
        train_annotations.json
        val_annotations.json
    visual_features/
        train_features.h5
        val_features.h5
    knowledge_graphs/
        train_graphs.h5
        val_graphs.h5
    answer_vocab.json

Dataset 2: <username>/hf-roberta-large   (~1.4 GB)
    config.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
    merges.txt
    pytorch_model.bin   (или model.safetensors)

Dataset 3: <username>/vqa-gnn-ckpt   (создать после первого успешного train)
    model_best.pth
    config.yaml
```

**Если визуальные признаки не вмещаются в один dataset (>20 GB):** создать отдельный `vqa-gnn-visual-features`, монтировать рядом и передавать путь через Hydra override:

```bash
datasets.train.data_dir=/kaggle/input/vqa-gnn-data  # для JSON и KG
# visual_features путь задаётся через data_dir, поэтому visual HDF5 должен быть там же.
# Альтернатива: симлинк внутри /kaggle/working/ при запуске notebook.
```

### dataset-metadata.json

```json
{
  "title": "VQA-GNN Data",
  "id": "<your_kaggle_username>/vqa-gnn-data",
  "licenses": [
    {"name": "other"}
  ]
}
```

### Команды Kaggle CLI

```bash
pip install kaggle

# Авторизация: скачать ~/.kaggle/kaggle.json с kaggle.com/settings
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Создать dataset (первый раз):
kaggle datasets create -p /path/to/vqa-gnn-data

# Обновить версию:
kaggle datasets version -p /path/to/vqa-gnn-data -m "add KG graphs 300-dim"

# Проверить статус:
kaggle datasets status <username>/vqa-gnn-data
```

### Ограничения Kaggle Datasets

- Максимум 100 GB на один dataset
- Каждый `kaggle datasets version` создаёт новую неизменяемую версию
- Удаление версий — только через UI

---

## 5. Подготовка Kaggle Notebook

Полный bootstrap для Kaggle notebook cell-by-cell.

```python
# Ячейка 1: клонировать репозиторий
!git clone https://github.com/<username>/paper_model_replication.git /kaggle/working/repo
import os
os.chdir("/kaggle/working/repo")
```

```bash
# Ячейка 2: установить зависимости
pip install -r requirements.txt --quiet

python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "from transformers import AutoTokenizer; print('transformers: OK')"
python -c "import h5py; print('h5py: OK')"
python -c "import comet_ml; print('comet_ml: OK')"
```

```bash
# Ячейка 3: проверить доступность datasets
ls /kaggle/input/vqa-gnn-data/
ls /kaggle/input/hf-roberta-large/
```

```bash
# Ячейка 4: валидация данных (ОБЯЗАТЕЛЬНО перед train)
python scripts/validate_data.py \
    --data-dir /kaggle/input/vqa-gnn-data \
    --split train val \
    --answer-vocab /kaggle/input/vqa-gnn-data/answer_vocab.json \
    --num-visual-nodes 36 \
    --feature-dim 2048 \
    --d-kg 300 \
    --num-answers 3129 \
    --max-kg-nodes 30 \
    --sample-size 200

# ОСТАНОВИТЬСЯ, если `exit code != 0`
```

```python
# Ячейка 5: настройка CometML
import os

# Через Kaggle Secrets (рекомендуется):
# from kaggle_secrets import UserSecretsClient
# os.environ["COMET_API_KEY"] = UserSecretsClient().get_secret("COMET_API_KEY")

# Или вручную (для офлайн-режима подойдёт любой непустой placeholder):
os.environ["COMET_API_KEY"] = "your_api_key_here"
```

```bash
# Ячейка 6: demo-смоук-тест (быстрая проверка пайплайна ПЕРЕД реальным run)
# Использует синтетические данные, но грузит `roberta-large` из dataset.
TRANSFORMERS_OFFLINE=1 python train.py \
    --config-name baseline_vqa \
    model.text_encoder_name=/kaggle/input/hf-roberta-large \
    trainer.save_dir=/kaggle/working/saved \
    trainer.n_epochs=1 \
    trainer.epoch_len=5 \
    writer.mode=offline
```

```bash
# Ячейка 7: реальное обучение (для стабильности рекомендуется internet=OFF)
TRANSFORMERS_OFFLINE=1 python train.py \
    --config-name baseline_vqa \
    datasets=vqa \
    model.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.train.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.val.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.train.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.train.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    model.d_kg=300 \
    datasets.train.d_kg=300 \
    datasets.val.d_kg=300 \
    dataloader.batch_size=4 \
    dataloader.num_workers=2 \
    trainer.save_dir=/kaggle/working/saved \
    trainer.n_epochs=20 \
    trainer.epoch_len=5000 \
    trainer.seed=42 \
    writer.mode=offline \
    writer.run_name=kaggle_run_1
```

```bash
# Ячейка 8: inference после завершения train
TRANSFORMERS_OFFLINE=1 python inference.py \
    --config-name inference_vqa \
    datasets=vqa_eval \
    model.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.val.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    datasets.val.d_kg=300 \
    model.d_kg=300 \
    inferencer.from_pretrained=/kaggle/working/saved/kaggle_run_1/model_best.pth \
    inferencer.save_path=vqa_preds
```

```python
# Ячейка 9: просмотр предсказаний
import glob, torch

preds_dir = "/kaggle/working/repo/data/saved/vqa_preds/val/"
files = sorted(glob.glob(f"{preds_dir}/output_*.pth"))
print(f"Total predictions: {len(files)}")
for f in files[:5]:
    p = torch.load(f, map_location="cpu")
    print(p)
```

```bash
# Ячейка 10: сохранить артефакты для скачивания
cp -r /kaggle/working/saved/kaggle_run_1 /kaggle/working/run_artifacts/
tar -czf /kaggle/working/kaggle_run_1_artifacts.tar.gz /kaggle/working/run_artifacts/

# Сохранить CometML offline logs:
cp -r ~/.cometml-runs /kaggle/working/cometml_offline_logs/

# После получения интернета — загрузить в CometML:
# comet upload ~/.cometml-runs/<experiment_key>
```

---

## 6. Сценарии

### Сценарий A: Есть локальная машина

**Recommended path:**

```bash
# 1. Установка и данные
pip install -r requirements.txt
# Скачать VQA v2 JSON вручную, переложить в data/vqa/

# 2. Vocab
python scripts/prepare_answer_vocab.py \
    --annotations data/vqa/annotations/train_annotations.json \
    --output data/vqa/answer_vocab.json

# 3. Визуальные признаки (из TSV)
python scripts/prepare_visual_features.py \
    --format tsv \
    --input /path/to/trainval_resnet101_faster_rcnn_genome_36.tsv \
    --output data/vqa/visual_features/all_features.h5
ln -s "$(pwd)/data/vqa/visual_features/all_features.h5" data/vqa/visual_features/train_features.h5
ln -s "$(pwd)/data/vqa/visual_features/all_features.h5" data/vqa/visual_features/val_features.h5

# 4. ConceptNet + KG graphs
mkdir -p data/conceptnet
wget -P data/conceptnet \
    "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"

# Обновить d_kg в конфигах:
sed -i 's/d_kg: 100/d_kg: 300/g' src/configs/model/vqa_gnn.yaml \
    src/configs/datasets/vqa.yaml \
    src/configs/datasets/vqa_eval.yaml

python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/train_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --output data/vqa/knowledge_graphs/train_graphs.h5 \
    --d-kg 300

python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/val_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --output data/vqa/knowledge_graphs/val_graphs.h5 \
    --d-kg 300

# 5. Validate
python scripts/validate_data.py \
    --data-dir data/vqa --split train val \
    --answer-vocab data/vqa/answer_vocab.json \
    --d-kg 300 --num-answers 3129

# 6. roberta-large для offline
python -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('roberta-large').save_pretrained('data/roberta-large')
AutoModel.from_pretrained('roberta-large').save_pretrained('data/roberta-large')
"

# 7. Публикация в Kaggle
pip install kaggle
# Подготовить ~/.kaggle/kaggle.json
kaggle datasets create -p data/vqa       # vqa-gnn-data
kaggle datasets create -p data/roberta-large  # hf-roberta-large
```

**Чего избегать:** запускать train локально если нет GPU с 16+ GB VRAM и roberta-large не заморожен.

---

### Сценарий B: Только Colab + Kaggle (без локальной машины)

**Рекомендуемый путь:** Colab для подготовки данных → Google Drive для хранения → Kaggle для обучения.

```python
# В Colab:
from google.colab import drive
drive.mount('/content/drive')

!git clone <repo_url> /content/repo
%cd /content/repo
!pip install -r requirements.txt --quiet

# VQA v2 JSON — загрузить вручную в Google Drive, затем:
!cp /content/drive/MyDrive/vqa_data/train_questions.json data/vqa/questions/
!cp /content/drive/MyDrive/vqa_data/val_questions.json   data/vqa/questions/
!cp /content/drive/MyDrive/vqa_data/train_annotations.json data/vqa/annotations/
!cp /content/drive/MyDrive/vqa_data/val_annotations.json   data/vqa/annotations/
```

```bash
# Answer vocab (быстро)
python scripts/prepare_answer_vocab.py \
    --annotations data/vqa/annotations/train_annotations.json \
    --output /content/drive/MyDrive/vqa_artifacts/answer_vocab.json

# Numberbatch
mkdir -p data/conceptnet
wget -q https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz \
     -O data/conceptnet/numberbatch-en-19.08.txt.gz

# KG graphs в фоне (риск: прерывание сессии Colab теряет незаписанное в HDF5)
nohup python scripts/prepare_kg_graphs.py \
    --questions data/vqa/questions/train_questions.json \
    --numberbatch data/conceptnet/numberbatch-en-19.08.txt.gz \
    --output /content/drive/MyDrive/vqa_artifacts/train_graphs.h5 \
    --d-kg 300 > /content/drive/MyDrive/kg_train_log.txt 2>&1 &

# Визуальные признаки (если TSV лежит в Drive):
python scripts/prepare_visual_features.py \
    --format tsv \
    --input /content/drive/MyDrive/trainval_36.tsv \
    --output /content/drive/MyDrive/vqa_artifacts/all_features.h5
```

**Риски Colab:**
- 12h лимит сессии прерывает долгие операции
- TSV с визуальными признаками (~10–15 GB) требует места в Colab runtime (107 GB диск)
- Используй nohup для долгих процессов; Colab Pro — для более длинных сессий

**Чего избегать:** запускать prepare_kg_graphs без nohup в Colab. При прерывании сессии частично записанный HDF5 потребует рестарта.

---

### Сценарий C: Визуальные признаки уже есть, нужно только остальное

```bash
# Проверить формат имеющихся features
python scripts/prepare_visual_features.py \
    --format h5 \
    --input /path/to/existing_features.h5 \
    --output /path/to/existing_features.h5 \
    --verify-only

# Если ключи не строки (integer image_ids) — rekey:
python scripts/prepare_visual_features.py \
    --format h5 \
    --input /path/to/existing_features.h5 \
    --output data/vqa/visual_features/train_features.h5 \
    --num-boxes 36 \
    --feature-dim 2048

# Если d_visual != 2048 — обновить в конфигах перед обучением:
# src/configs/model/vqa_gnn.yaml:  d_visual: <your_dim>
# src/configs/datasets/vqa.yaml:   d_visual: <your_dim>

# Затем только vocab + KG:
python scripts/prepare_answer_vocab.py ...
python scripts/prepare_kg_graphs.py ...
python scripts/validate_data.py ...
```

---

### Сценарий D: Internet ON в Kaggle

```bash
export COMET_API_KEY=your_real_api_key

# HuggingFace скачивает roberta-large автоматически (~1.4 GB при первом запуске)
python train.py --config-name baseline_vqa \
    datasets=vqa \
    datasets.train.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.train.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    model.d_kg=300 \
    datasets.train.d_kg=300 \
    datasets.val.d_kg=300 \
    dataloader.batch_size=4 \
    trainer.save_dir=/kaggle/working/saved \
    writer.mode=online \
    writer.project_name=vqa_gnn \
    writer.run_name=kaggle_run_1_online
```

---

### Сценарий E: Internet OFF в Kaggle (рекомендуется для стабильности)

```bash
# COMET_API_KEY нужен даже в offline режиме (любое непустое значение)
export COMET_API_KEY=offline_placeholder

TRANSFORMERS_OFFLINE=1 python train.py --config-name baseline_vqa \
    datasets=vqa \
    model.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.train.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.val.text_encoder_name=/kaggle/input/hf-roberta-large \
    datasets.train.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.train.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
    datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
    model.d_kg=300 \
    datasets.train.d_kg=300 \
    datasets.val.d_kg=300 \
    dataloader.batch_size=4 \
    trainer.save_dir=/kaggle/working/saved \
    writer.mode=offline \
    writer.run_name=kaggle_run_1

# CometML offline logs сохраняются в /root/.cometml-runs/
# Скопировать, чтобы не потерять после завершения сессии:
cp -r ~/.cometml-runs /kaggle/working/cometml_offline_logs/

# Загрузить в CometML после получения интернета:
# comet upload ~/.cometml-runs/<experiment_key>
```

---

## 7. Полезные overrides для обучения

```bash
# Заморозить text encoder (быстрее, меньше GPU памяти):
python train.py --config-name baseline_vqa model.freeze_text_encoder=True

# Вернуться к более лёгкому энкодеру (если не хватает памяти):
python train.py --config-name baseline_vqa \
    model.text_encoder_name=bert-base-uncased \
    model.d_hidden=768 \
    datasets.train.text_encoder_name=bert-base-uncased \
    datasets.val.text_encoder_name=bert-base-uncased

# Продолжить с чекпойнта:
python train.py --config-name baseline_vqa \
    trainer.resume_from=saved/kaggle_run_1/checkpoint-epoch10.pth \
    trainer.override=True

# Быстрый smoke-test конвергенции (2–3 часа вместо 12):
python train.py --config-name baseline_vqa \
    datasets=vqa \
    model.freeze_text_encoder=True \
    trainer.n_epochs=5 \
    trainer.epoch_len=2000 \
    dataloader.batch_size=8
```

---

## 8. Ожидаемые артефакты и метрики

### Артефакты после каждого запуска

```text
saved/kaggle_run_1/
├── config.yaml              # полный Hydra конфиг (воспроизводимость)
├── git_commit.txt           # hash коммита
├── git_diff.patch           # незакоммиченные изменения
├── checkpoint-epoch5.pth    # периодические чекпойнты (save_period=5)
├── checkpoint-epoch10.pth
├── ...
└── model_best.pth           # лучший val_VQA_Accuracy
```

### Метрики, которые будут считаться

- `train_loss` — VQA soft BCE loss
- `val_loss` — VQA soft BCE loss на val split
- `val_VQA_Accuracy` — официальная метрика VQA v2 (`min(count/3, 1.0)`)

### Интерпретация первого запуска

| Результат | Что означает |
|---|---|
| `val_VQA_Accuracy` > 0 после 1 эпохи | pipeline работает |
| Loss убывает за первые 5 эпох | обучение идёт |
| `val_VQA_Accuracy` растёт и стабилизируется | конвергенция |
| `val_VQA_Accuracy` ≈ 60–65% после 20 эпох | реалистичный baseline |
| Числа из статьи (>65%) | **не подтверждены в этом репозитории** |

### Критерии успеха

**"Pipeline works":**
- Demo-смоук-тест завершился без ошибок
- Реальный train запустился, loss убывает, `val_VQA_Accuracy` > 0
- `model_best.pth` и `config.yaml` сохранены
- Inference отработал, предсказания записаны

**"Reproducibility improved":**
- Все артефакты (`config.yaml`, `git_commit.txt`, `git_diff.patch`) сохранены
- Все deviations зафиксированы (d_kg выбор, entity extraction, adjacency structure)
- Полученный `val_VQA_Accuracy` задокументирован рядом с числами из статьи
- Команда воспроизведения — одна строка из README

**Что нельзя утверждать после первого run:**
- Что результаты воспроизводят числа из статьи — без совпадения детектора и полного KG pipeline
- Что отклонения несущественны — они задокументированы как approximations

---

## 9. Оставшиеся блокеры для точного воспроизведения статьи

| Блокер | Статус | Workaround в репозитории |
|---|---|---|
| Точный детектор объектов из статьи | не опубликован | Anderson et al. bottom-up features (задокументировано) |
| Точный KG pipeline статьи | не полностью описан | ConceptNet Numberbatch + simple tokenization (задокументировано) |
| Точная нормализация ответов | не задокументирована в статье | VQA v2 standard pipeline (задокументировано) |
| Точный HF checkpoint для RoBERTa | не зафиксирован в статье | `roberta-large` (задокументировано) |
| Численные результаты | не подтверждены | будут получены после честного run |
| Структура adjacency | не полностью задана в статье | visual↔q + kg↔q + kg↔kg (задокументировано) |

---

## 10. Итоговый рекомендуемый путь

**Минимально рискованный путь для данного репозитория прямо сейчас:**

1. **(30 мин, локально или Colab)** Скачать VQA v2 JSON → `prepare_answer_vocab.py` → скачать Numberbatch.

2. **(2–4 часа, Colab или local)** `prepare_kg_graphs.py` для train и val с `--d-kg 300`. Параллельно обновить d_kg в трёх YAML-файлах. Сохранить HDF5 в Google Drive или локально для upload.

3. **(30–60 мин, local)** Конвертировать TSV с визуальными признаками в HDF5. Это самый тяжёлый шаг по диску; если TSV уже есть — достаточно одного вызова `prepare_visual_features.py`.

4. **(15 мин)** `validate_data.py` — убедиться что все артефакты правильного формата.

5. **(30–60 мин upload)** `kaggle datasets create` для `vqa-gnn-data` и `hf-roberta-large`.

6. **(12h GPU, Kaggle)** Запустить train с `batch_size=4`, `TRANSFORMERS_OFFLINE=1`, `freeze_text_encoder=False` (согласовано со статьёй), `d_kg=300`.

   Если RAM не хватает → `freeze_text_encoder=True` как первый тест конвергенции, затем полный run.

7. **Записать полученный `val_VQA_Accuracy` рядом с числами из статьи.** Это и есть честный результат воспроизведения на данном этапе.

**Если хочешь сэкономить GPU time при первом запуске:**
Сначала запусти с `freeze_text_encoder=True` и `epoch_len=2000` для быстрой проверки конвергенции (5–6 часов). Если всё хорошо — следующий run с размороженным encoder (согласовано со статьёй).
