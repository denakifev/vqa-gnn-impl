# Разделение архитектуры

Этот репозиторий использует упрощённую двухконтурную архитектуру.

Текущее разделение архитектуры:

- `GQA` — основной benchmark и главный paper-inspired runtime path.
- `VQA-2` — вспомогательный extension path на том же training/inference
  framework.
- удалённый третий benchmark находится вне активного scope репозитория.

## Активные runtime-path’ы

GQA:

- модель: `src/model/gqa_model.py::GQAVQAGNNModel`
- датасет: `src/datasets/gqa_dataset.py::GQADataset`
- collate: `src/datasets/gqa_collate.py::gqa_collate_fn`
- train config: `src/configs/baseline_gqa.yaml`
- inference config: `src/configs/inference_gqa.yaml`

VQA-2:

- модель: `src/model/vqa_gnn.py::VQAGNNModel`
- датасет: `src/datasets/vqa_dataset.py::VQADataset`
- collate: `src/datasets/vqa_collate.py::vqa_collate_fn`
- train config: `src/configs/baseline_vqa.yaml`
- inference config: `src/configs/inference_vqa.yaml`

## Общее ядро

Общие примитивы остаются в:

- `src/model/gnn_core.py`
- `train.py`
- `inference.py`
- `src/trainer/`
- `src/logger/`

Общее ядро должно оставаться task-agnostic. Выбор задачи должен происходить
через wiring dataset/config/model, а не через скрытые conditionals в общих
модулях.

## Правило scope

Любое дальнейшее описание репозитория должно использовать такую рамку:

- GQA main
- VQA-2 auxiliary
- no removed benchmark in active scope
