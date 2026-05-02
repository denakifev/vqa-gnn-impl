# Architecture Split

This repository uses a simplified two-path architecture.

Current architecture split:

- `GQA` is the primary benchmark and the main paper-inspired runtime path.
- `VQA-2` is the auxiliary extension path built on the same training and
  inference framework.
- the removed third benchmark is outside active repository scope.

## Active Runtime Paths

GQA:

- model: `src/model/gqa_model.py::GQAVQAGNNModel`
- dataset: `src/datasets/gqa_dataset.py::GQADataset`
- collate: `src/datasets/gqa_collate.py::gqa_collate_fn`
- train config: `src/configs/baseline_gqa.yaml`
- inference config: `src/configs/inference_gqa.yaml`

VQA-2:

- model: `src/model/vqa_gnn.py::VQAGNNModel`
- dataset: `src/datasets/vqa_dataset.py::VQADataset`
- collate: `src/datasets/vqa_collate.py::vqa_collate_fn`
- train config: `src/configs/baseline_vqa.yaml`
- inference config: `src/configs/inference_vqa.yaml`

## Shared Core

Shared primitives remain in:

- `src/model/gnn_core.py`
- `train.py`
- `inference.py`
- `src/trainer/`
- `src/logger/`

The shared core must stay task-agnostic. Task selection should happen through
dataset/config/model wiring, not hidden conditionals in common modules.

## Scope Rule

Any future repository narrative should use this framing:

- GQA main
- VQA-2 auxiliary
- no removed benchmark in active scope
