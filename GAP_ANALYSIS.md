# Gap Analysis

Дата актуализации: 2026-05-02.

Статус репозитория: `partially validated baseline`.

Новая цель: practical GQA и VQA-2 train, которые можно запускать в
Kaggle-лимитах. GQA сохраняет `paper-aligned approximation`; VQA-2 является
coursework extension и не должен подаваться как результат статьи.

Active repository mode:

- GQA main
- VQA-2 auxiliary
- no removed benchmark in active scope

## Implemented / Validated

- GQA 1842-way classification: `implemented`, `validated`.
- GQA relation-aware message passing via `graph_edge_types`: `implemented`,
  `validated`, `paper-aligned approximation`.
- GQA additive sparse graph-link module for visual<->textual/kg exchange:
  `implemented`, `runtime-valid`.
- Strict GQA preprocessing package: `validated`, uploaded to Kaggle, then
  validated end-to-end against restored real Kaggle data.
- Single Kaggle-practical GQA training config: `implemented`,
  `not validated yet` on a full real train run.
- VQA-2 coursework path (`VQADataset`, `VQAGNNModel`, `VQALoss`,
  `VQAAccuracy`, `baseline_vqa`): `implemented`, `not validated yet`.
- VQA-2 additive sparse graph-link module path: `implemented`,
  `runtime-valid`.
- Demo and engineering tests: `runtime-valid`.
- Recorded practical baseline observations:
  GQA controlled subset `val_GQA_Accuracy = 0.20`,
  VQA-2 classic baseline `val_VQA_Accuracy = 0.54`.

## Remaining GQA Gaps

| Gap | Status | Notes |
|---|---|---|
| Real GQA training/eval numbers | `not validated yet` | Requires actual train run |
| Kaggle-friendly protocol | `not validated yet` | Config defaults are practical, but need timed Kaggle run |
| Exact warmup length, batch size, dropout, max seq len, grad accumulation | `paper-aligned approximation` | Paper underspec |
| Hidden widths / activations in relation and node MLPs | `paper-aligned approximation` | `d_hidden=512` by default for Kaggle practicality |
| Visual vs textual scene-graph relation vocab split | `paper-aligned approximation` | Shared 624 relation vocab |
| Entity linking via tokenization | `paper-aligned approximation` | Practical preprocessing choice |
| Exact RoBERTa-large checkpoint provenance | `paper-aligned approximation` | Public checkpoint default |
| Graph-link quality gain over baseline | `not validated yet` | Module implemented for controlled experiments; no real metric claim yet |
| Separate external concept graph in active GQA path | `not paper-faithful yet` | Current graph-link module operates on visual nodes + existing textual/kg nodes, not a restored ConceptNet branch |
| First graph-link formulation on VQA-2 | `not validated yet` | Early runs regressed against the recorded baseline; module was later rewritten into a more conservative MAGIC-inspired variant |

## Remaining VQA-2 Gaps

| Gap | Status | Notes |
|---|---|---|
| Real VQA-2 data package | `not validated yet` | Expected layout is documented in DATA_CONTRACTS.md |
| Real VQA-2 training/eval numbers | `not validated yet` | Requires `scripts/validate_vqa_data.py`, then actual train run |
| VQA-2 relation typing | `not paper-faithful yet` | Current VQA-2 graph contract has untyped adjacency |
| VQA-2 claim relative to paper | `not validated yet` | Coursework extension only; paper did not validate this path here |

## Practical Training Risks

- Default `baseline_gqa.yaml` is now a small Kaggle-practical profile:
  `batch_size=4`, `epoch_len=1000`, `n_epochs=3`, frozen text encoder,
  `d_hidden=512`, and validation limited to 10k samples.
- `baseline_vqa.yaml` uses the same Kaggle-practical shape: batch 4,
  frozen text encoder, `d_hidden=512`, 3 epochs, 1000 steps per epoch, and
  validation limited to 10k samples.
- Trainer currently has no AMP and no gradient accumulation.
- Validation metrics are approximate when `datasets.val.limit=10000`.
- Unfreezing RoBERTa-large or raising `d_hidden` may exceed the available
  notebook GPU budget.
- `graph_link_gqa_frozen` is the recommended first controlled research mode if
  the goal is to isolate the contribution of the new module.
- For graph-link experiments, frozen runs are only meaningful when initialized
  from a trained baseline checkpoint; frozen-from-scratch runs can collapse and
  should be treated as setup failures, not architecture conclusions.

## Recommended First Run

```bash
python train.py --config-name baseline_gqa datasets=gqa \
  datasets.train.data_dir=/kaggle/input/gqa-gnn-data \
  datasets.train.answer_vocab_path=/kaggle/input/gqa-gnn-data/gqa_answer_vocab.json \
  datasets.val.data_dir=/kaggle/input/gqa-gnn-data \
  datasets.val.answer_vocab_path=/kaggle/input/gqa-gnn-data/gqa_answer_vocab.json \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  trainer.save_dir=/kaggle/working/saved \
  trainer.progress_bar=auto \
  trainer.override=True
```

VQA-2:

```bash
python train.py --config-name baseline_vqa datasets=vqa \
  datasets.train.data_dir=/kaggle/input/vqa-gnn-data \
  datasets.train.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
  datasets.val.data_dir=/kaggle/input/vqa-gnn-data \
  datasets.val.answer_vocab_path=/kaggle/input/vqa-gnn-data/answer_vocab.json \
  model.text_encoder_name=/kaggle/input/roberta-large \
  datasets.train.text_encoder_name=/kaggle/input/roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/roberta-large \
  trainer.save_dir=/kaggle/working/saved \
  trainer.progress_bar=auto \
  trainer.override=True
```

## Claim Discipline

Allowed status tokens:

- `implemented`
- `validated`
- `paper-aligned`
- `paper-aligned approximation`
- `not paper-faithful yet`
- `runtime-valid`
- `not validated yet`

Forbidden claims:

- `paper reproduced`
- `fully paper-faithful`
- `ready for full reproduction`
- `GQA real baseline validated`
