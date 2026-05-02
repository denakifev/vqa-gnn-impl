# Project Scope

Current active repository scope:

- `GQA` = primary benchmark and primary training path.
- `VQA-2` = auxiliary validated extension path.
- the removed third benchmark is outside active repository scope.

What this means in practice:

- active docs, configs, tests, scripts, and comments must describe only GQA
  and VQA-2 workflows;
- the removed benchmark is not a supported benchmark, roadmap item,
  training target, or next workstream in this repository;
- future training and experiment work should start from GQA, then extend to
  controlled GQA / VQA-2 comparisons.

Source-of-truth runtime paths:

- GQA:
  `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel -> GQALoss -> GQAAccuracy`
- VQA-2:
  `VQADataset -> vqa_collate_fn -> VQAGNNModel -> VQALoss -> VQAAccuracy`

Status discipline:

- GQA can be described as `paper-aligned approximation`.
- VQA-2 must be described as an auxiliary validated extension path or
  coursework extension.
- the removed benchmark must not be described as active, supported, or
  pending.
