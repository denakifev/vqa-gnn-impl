# Training Protocol Parity

Reference: VQA-GNN §5.1 (arXiv:2205.11501).

Two valid states per item: `paper-aligned` or `not paper-faithful yet` (with the same blocker categories as
`BLOCKED_EXACTNESS_ITEMS.md`: MP, EA, OT, OC, EX).

## 1. VCR Protocol

### 1.1 Paper-cited facts (§5.1)

> "We joint train VQA-GNN on Q→A and QA→R, with a common LM encoder, the multimodal semantic
> graph for Q→A, a concept graph retrieved by giving question-answer pair with a rationale
> candidate for QA→R."

> "We use a pretrained RoBERTa Large model to embed the QA-context node, and finetune it with
> the multimodal GNN for 50 epoch by using learning rates 1e-5 and 1e-4 respectively."

> "We set the number of layers (L = 5) of VQA-GNN and use AdamW optimizer to minimize the loss.
> We use a linear warmup of the learning rate over the 15-th epoch, with a cosine decay thereafter
> to 0."

### 1.2 Current implementation status

| Item | Paper | Legacy path | Paper-equation path | Status |
|---|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW | `paper-aligned` |
| Text encoder | RoBERTa Large | RoBERTa Large | RoBERTa Large | `paper-aligned` |
| #GNN layers | L = 5 | `num_gnn_layers: 3` | `num_gnn_layers: 5` | `paper-aligned` (paper-equation config) |
| #Epochs | 50 | 10 | 50 | `paper-aligned` (paper-equation config) |
| LM LR | 1e-5 | 2e-5 | 1e-5 (via param group) | `paper-aligned` (paper-equation config) |
| GNN LR | 1e-4 | 2e-5 (shared) | 1e-4 (via param group) | `paper-aligned` (paper-equation config) |
| Scheduler | linear warmup over 15 epochs, cosine decay → 0 | StepLR | `PaperWarmupCosine` (warmup_epochs=15, total_epochs=50) | `paper-aligned` (paper-equation config) |
| Joint Q→A + QA→R training | **joint** with common LM encoder | **separate** (two configs / two models) | single-task until joint mechanics spec'd | `not paper-faithful yet` — MP (§C.1 of `BLOCKED_EXACTNESS_ITEMS.md`) |
| Loss | CE over 4 candidates | `VCRLoss` (CE) | `VCRLoss` (CE) | `paper-aligned` |
| Batch size | not stated | `batch_size: 8` | unspecified (inherits legacy) | `not paper-faithful yet` — MP |
| Dropout | not stated | 0.1 | 0.0 (no silent regularisation) | `not paper-faithful yet` — MP |
| Max seq len | not stated | 128 | unspecified | `not paper-faithful yet` — MP |
| Grad accumulation | not stated | none | none | `not paper-faithful yet` — MP |
| Seed / #runs | not stated | 42 / 1 run | 42 / 1 run | `not paper-faithful yet` — MP |
| Gradient clipping | not stated | `max_grad_norm: 1.0` | keeps 1.0; not cited | `not paper-faithful yet` — MP |

### 1.3 Legacy vs paper-equation path

- Legacy configs `baseline_vcr_qa.yaml`, `baseline_vcr_qar.yaml` are deliberately *kept* as
  Ablation 2 variants (single GNN on merged graph) for smoke tests and backward compatibility.
  They are **not** paper-aligned and are documented as such in `ARCHITECTURE_SPLIT.md`.
- Paper-equation configs `src/configs/paper_vqa_gnn_vcr.yaml` wire together:
  - `src.model.PaperVCRModel / PaperGQAModel` (two modality GNNs + super-node readout + paper-equation RGAT)
  - `src.lr_schedulers.PaperWarmupCosine`
  - `src.optim.paper_param_groups` (LM @ 1e-5, rest @ 1e-4)
  - `trainer.n_epochs: 50`, `model.num_gnn_layers: 5`
- The paper-equation VCR config **cannot be run end-to-end** until `BLOCKED_EXACTNESS_ITEMS.md`
  entries A.1–A.5 and B.1 are resolved.

## 2. GQA Protocol

### 2.1 Paper-cited facts (§5.1)

> "We define the question as the context node (node q) to fully connect visual and textual scene
> graphs (SG) respectively"

> "The node q is embedded with a pretrained RoBERTa large model, and we initialize object nodes'
> representations in visual SG using official object features, object nodes in textual SG by
> concatenating GloVe [31] based word embedding of the object name and attributes."

> "classify the given image-question pair out of 1,842 answer classes."

> "finetune the node q with VQA-GNN for 50 epoch by using learning rates 2e-5 and 2e-4 respectively."

### 2.2 Current implementation status

| Item | Paper | Legacy path | Paper-equation path | Status |
|---|---|---|---|---|
| Optimizer | AdamW (inferred from VCR section) | AdamW | AdamW | `paper-aligned` |
| Text encoder | RoBERTa Large | RoBERTa Large | RoBERTa Large | `paper-aligned` |
| #Epochs | 50 | 20 | 50 | `paper-aligned` (paper-equation config) |
| LM LR | 2e-5 | 1e-4 (shared) | 2e-5 (via param group) | `paper-aligned` (paper-equation config) |
| GNN LR | 2e-4 | 1e-4 (shared) | 2e-4 (via param group) | `paper-aligned` (paper-equation config) |
| Scheduler | not explicitly restated for GQA | StepLR | `PaperWarmupCosine` (15 warmup, 50 total) reused from VCR | `not paper-faithful yet` — MP §C.7 |
| Answer vocab size | 1842 | 1842 | 1842 | `paper-aligned` |
| Loss | hard CE | `GQALoss` | `GQALoss` | `paper-aligned` |
| Metric | exact match | `GQAAccuracy` | `GQAAccuracy` | `paper-aligned` |
| Visual features | official object features | local HDF5 (unvalidated path) | same, plus fail-fast on mismatch | runtime `not paper-faithful yet` — EA §A.7 |
| Textual SG features | concat(GloVe(obj), GloVe(attr)) → 600-d | preprocessing ready | preprocessing ready | runtime `not paper-faithful yet` — EA §A.6 |
| #GNN layers | not restated for GQA | 3 | 5 (default; `not paper-faithful yet` — MP otherwise) | `not paper-faithful yet` — MP |
| Batch size | not stated | 16 | unspecified | `not paper-faithful yet` — MP |
| Dropout | not stated | 0.1 | 0.0 | `not paper-faithful yet` — MP |
| Max seq len | not stated | 128 | unspecified | `not paper-faithful yet` — MP |
| Seed / #runs | not stated | 42 / 1 | 42 / 1 | `not paper-faithful yet` — MP |

## 3. Optimization layout — parameter groups

Paper specifies two LRs (LM vs GNN). Implementation: `src/optim/paper_param_groups.py` splits
parameters into:

- Group A (LM): `HFQuestionEncoder.encoder.*` — LR 1e-5 (VCR) / 2e-5 (GQA).
- Group B (rest): every other parameter — LR 1e-4 (VCR) / 2e-4 (GQA).

Tested by `tests/test_paper_scheduler_and_optim.py::test_paper_param_groups_split_lm_and_gnn_params`.

## 4. Learning rate scheduler

Paper: "linear warmup of the learning rate over the 15-th epoch, with a cosine decay thereafter
to 0." Natural reading: linear ramp from 0 → peak over the first 15 epochs, then cosine decay
from peak → 0 over the remaining 35 epochs (50 total for VCR).

Implementation: `src/lr_schedulers/paper_warmup_cosine.py::PaperWarmupCosine` — operates at
step granularity so behaviour is independent of `epoch_len`. Parameters: `warmup_steps`,
`total_steps`. Tested by `tests/test_paper_scheduler_and_optim.py`.

Known `not paper-faithful yet` items:

- C.6 (`BLOCKED_EXACTNESS_ITEMS.md`): warmup curve shape (linear is the most natural reading,
  but smoothstep or constant-then-ramp cannot be ruled out).
- C.7: GQA warmup length not restated in paper.

## 5. VCR Q→AR evaluation (post-hoc)

| Item | Paper | Implementation | Status |
|---|---|---|---|
| Q→AR joint accuracy | per-sample AND of Q→A and QA→R correctness | `VCRQARJointAccuracy` + `scripts/eval_vcr_qar.py` | `paper-aligned` |
| Pairing key | sample id | `annot_id` / `sample_id` | `paper-aligned` |
| Test regime for QA→R input (GT vs. predicted answer) | not specified | **community-standard**: use GT answer for QA→R input at both train and test time | `not paper-faithful yet` — MP (§D.1) |

## 6. Checkpoint & inference

| Item | Paper | Implementation | Status |
|---|---|---|---|
| Save best via monitor metric | implicit | `monitor` in configs | `paper-aligned` (framework detail, not a paper requirement) |
| Inference tensors | same as training batch keys | `inferencer.skip_model_load` override supported | `paper-aligned` |

## 7. Final status

Paper-equation training protocol is `implemented` for the explicit items below and
`not paper-faithful yet` for the underspecified items below:

- paper-aligned: optimizer, text encoder, per-param-group LRs, scheduler form, epochs, L,
  losses, metrics, answer vocab, Q→AR post-hoc joint accuracy.
- not paper-faithful yet: joint-task mixing mechanics, batch size, dropout, max seq length, gradient
  accumulation, seed/#runs for paper numbers, QA→R input regime, GQA warmup length.

Aggregate verdict per `BLOCKED_EXACTNESS_ITEMS.md`: `not paper-faithful yet`.
