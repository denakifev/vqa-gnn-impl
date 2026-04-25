# Blocked Exactness Items

This file is the **single source of truth** for every paper requirement that this repo cannot
honestly claim to implement exactly. Each item lists the exact blocker category, the concrete
artefact or spec gap, and the minimum action needed to unblock. No item here is an
"approximation" — any entry that could honestly be implemented has already been implemented
and moved to `paper-aligned` in `PAPER_PARITY_MATRIX.md`.

Blocker categories (exhaustive):

- **MP** = *missing paper detail* — the paper does not specify the value/formula to the precision
  required for bit-level reproduction.
- **EA** = *unavailable external artifact* — requires a dataset, checkpoint, or preprocessing
  output that is neither shipped in this repo nor released by the authors.
- **OT** = *unspecified training detail* — optimizer/data pipeline detail absent from paper.
- **OC** = *inaccessible original code* — upstream implementation not released (or only described
  in prose) and needed to close the gap.
- **EX** = *external dependency not reproducible* — third-party artifact whose output is not
  deterministic from paper text alone.

---

## A. Data and Artifact Blockers

### A.1 VCR — scene graph from Tang et al. unbiased SG generator
- **Category**: EA + EX
- **Paper ref**: §4.1, reference [38].
- **What is missing**: the unbiased SG generator checkpoint and its exact recall@20 thresholding
  on VCR images. VQA-GNN's scene-graph nodes and edges come from this model's output.
- **Impact**: all VCR scene-graph nodes, all scene-graph typed edges, and the `r(e)` image edges
  derive from this artefact.
- **Unblock**: obtain the Tang et al. 2020 CVPR checkpoint and run on VCR train/val/test images;
  serialize output to the paper-aligned data contract (see `DATA_CONTRACTS.md`).

### A.2 VCR — VinVL p-node features
- **Category**: EA
- **Paper ref**: §4.1 "mean pooled representation as a QA-concept node"; Table 2 "Node p (Vinvl)".
- **What is missing**: VinVL [53] ResNeXt-152 + OD head checkpoint and VisualGenome region
  retrieval by QA context.
- **Impact**: the p super-node — a fundamental part of Eq 8 — cannot be populated from current
  local data. Without it, the paper-aligned VCR head cannot run.
- **Unblock**: run VinVL on VG regions selected by QA-context relevance; cache `p_features`
  alongside each VCR question.

### A.3 VCR — ConceptNet retrieval with 1024-d node embeddings
- **Category**: EA + MP + OC
- **Paper ref**: §4.1 "concept entity ci is represented using a 1024-dimensional text feature
  vector as the concept entity embedding in [8]"; Fig 3 `Relev(e|a)` retrieval.
- **What is missing**:
  1. The 1024-d text embedding pipeline from QA-GNN [8] is *not* equivalent to ConceptNet
     Numberbatch (300-d). It uses a RoBERTa-based relation encoder whose exact weights and
     projection layer are tied to QA-GNN's code.
  2. The retrieval algorithm `Relev(e|a)` is shown as a figure but not formalised (no top-k,
     no scoring function).
  3. The "adjacent local entity" triple-insertion rule (e.g., `(bottle, aclocation, beverage)`)
     is illustrated but not given a deterministic spec.
- **Impact**: the whole concept-graph branch of VCR cannot be claimed paper-aligned on local
  data.
- **Unblock**: port QA-GNN embedding path from https://github.com/michiyasunaga/qagnn; specify a
  deterministic retrieval algorithm.

### A.4 VCR — object-mention grounding (p-node interaction)
- **Category**: MP + EA
- **Paper ref**: §4.1.
- **What is missing**: the paper does not specify how VCR's `[person0]`-style mention tokens are
  tied to VG regions (if at all). Current per-instance string identifiers preserve identity but
  are explicitly not region-grounded.
- **Unblock**: close A.2 first, then bind mention tokens to region indices at tokenization time.

### A.5 VCR — real local data
- **Category**: EA
- **What is missing**: `data/vcr/train.jsonl`, `data/vcr/val.jsonl`, VCR visual HDF5, VCR KG HDF5.
- **Impact**: blocks every VCR run regardless of architecture.
- **Unblock**: sign VCR ToU and download; use `scripts/prepare_vcr_data.py`.

### A.6 GQA — local graph artifacts mismatch paper-path contract
- **Category**: EA
- **Paper ref**: §5.1 GQA block.
- **What is missing**: local `data/gqa/knowledge_graphs/*.h5` carry `node_features (N, 300)` and
  no `graph_edge_types`. Paper-path expects `d_kg=600` (GloVe-concat object + attribute) and
  typed edges with `relation_count_total` attrs.
- **Impact**: `GQADataset` fail-fasts at first open.
- **Unblock**: rebuild via `scripts/prepare_gqa_data.py knowledge-graphs --d-kg 600`.

### A.7 GQA — official object features
- **Category**: EA
- **What is missing**: GQA's official `gqa_objects.h5` (BUTD-style 36×2048 per image) not shipped
  locally with paper-path attrs.
- **Unblock**: download GQA official features and package with scene-graph HDF5.

---

## B. Architecture Blockers

### B.1 Merged-graph data contract vs two-modality-subgraph architecture
- **Category**: EA
- **Paper ref**: §4.2 (two modality-specialized GNNs; Ablation Table 3 quantifies the gap).
- **What is missing**: current dataset emits a single merged graph (`graph_adj`,
  `graph_node_features`, `graph_node_types`) with no explicit S/C subgraph separation nor
  super-node ids. `PaperVCRModel / PaperGQAModel` expects `scene_subgraph_mask`, `concept_subgraph_mask`,
  and named super-node indices.
- **Impact**: runnable end-to-end path through `PaperVCRModel / PaperGQAModel` is blocked until datasets
  emit these fields.
- **Unblock**: extend `VCRDataset` and `GQADataset` to emit the paper-aligned subgraph
  contract (this requires the upstream blockers A.1–A.7 to be resolved first, since the exact
  separation depends on the paper-path preprocessing).

### B.2 Hidden dimension D of the GNN
- **Category**: MP
- **Paper ref**: §4.2 uses "D" symbolically.
- **What is missing**: explicit value of D.
- **Current decision**: `paper_vqa_gnn_vcr.yaml` sets `d_hidden = 1024` to match RoBERTa-Large
  output width (the only hint in the paper is "RoBERTa Large" for z). This is a *defensible*
  choice traceable to Eq 7 (z comes from RoBERTa, so concat requires matched dim), but it is
  not stated in the paper, so the item stays `not paper-faithful yet` under MP until the authors confirm.

### B.3 Number of attention heads in the GAT
- **Category**: MP
- **Paper ref**: Eqs 4–5 show scalar `γ_ij`; no mention of multi-head.
- **What is missing**: whether the paper uses multi-head or single-head attention.
- **Current decision**: `PaperMultiRelationGATLayer` uses single-head (literal reading of Eq 5).
  Multi-head is not introduced in §4.2 equations. This is the most literal reading; item stays
  `not paper-faithful yet` until confirmed.

### B.4 Widths and activations inside f_r, f_m, f_q, f_k, f_h, f_z, f_c MLPs
- **Category**: MP + OC
- **Paper ref**: §4.2 (Eqs 1–7), §4.3 (Eq 9).
- **What is missing**: the paper specifies only dimensional signatures and that `f_r` is a
  2-layer MLP, `f_h` is a 2-layer MLP with BatchNorm. Intermediate widths, activations, bias
  policy, and layer order are not specified.
- **Current decision**: hidden width = `d_hidden`; activation = `ReLU` (`f_r` and `f_h`); bias
  as PyTorch default. These are deterministic but not paper-cited.

### B.5 Dropout rate
- **Category**: MP
- **What is missing**: paper does not state dropout.
- **Current decision**: paper-aligned configs set `dropout = 0.0` (no silent regularisation).

### B.6 Weight initialization
- **Category**: MP
- **What is missing**: the paper does not specify init. The current `PaperVCRModel / PaperGQAModel` uses the
  same Xavier init policy as the legacy models. This item stays `not paper-faithful yet` under MP.

### B.7 Concept-graph node cap (N=60) selection procedure
- **Category**: MP
- **What is missing**: the paper says "maximum number of concept-graph nodes of 60" but does not
  define the truncation (e.g., by relevance score ties, hop distance, or random sampling).

### B.8 QA-text entity extraction algorithm
- **Category**: MP
- **What is missing**: extraction of entities from question and answer text for edges `r(q)` and
  `r(a)` is asserted but not specified.

---

## C. Training Protocol Blockers

### C.1 Joint VCR training mechanics
- **Category**: MP
- **Paper ref**: §5.1 "We joint train VQA-GNN on Q→A and QA→R, with a common LM encoder".
- **What is missing**: per-step mixing strategy (alternating batches vs. summed losses),
  per-task loss weights, whether QA→R uses GT answer (teacher-forcing) or predicted answer.
- **Current decision**: the legacy configs `baseline_vcr_qa.yaml` and `baseline_vcr_qar.yaml`
  train two separate models (non-paper-aligned). The paper-aligned config
  `paper_vqa_gnn_vcr.yaml` is a single-task config until MP is resolved; reaching joint training
  requires a new `JointVCRTrainer` which cannot be honestly written without knowing the paper's
  mixing strategy.

### C.2 Batch size
- **Category**: MP
- **What is missing**: paper does not state per-task or effective batch size.

### C.3 Max sequence length
- **Category**: MP
- **What is missing**: paper does not state text max-length; current tokenization uses 128.

### C.4 Gradient accumulation
- **Category**: MP
- **What is missing**: no mention in paper.

### C.5 Random seed / number of runs
- **Category**: MP
- **What is missing**: paper reports single-run numbers; seed not given.

### C.6 Warmup curve shape
- **Category**: MP
- **What is missing**: paper says "linear warmup of the learning rate over the 15-th epoch" —
  a linear ramp from 0 → peak is the most natural reading, but the precise curve type (step vs.
  linear vs. smoothstep) is not stated. `PaperLRScheduler` uses linear ramp on step granularity.

### C.7 GQA warmup length
- **Category**: MP
- **Paper ref**: §5.1 GQA block specifies 50 epochs and LRs but does not restate the 15-epoch
  warmup.
- **Current decision**: `paper_vqa_gnn_gqa.yaml` reuses the 15-epoch warmup by default and
  flags this as underspecified.

---

## D. Evaluation Protocol Blockers

### D.1 VCR Q→AR — paper's training regime for QA→R candidate rationale construction
- **Category**: MP
- **Paper ref**: §5.1 "a concept graph retrieved by giving question-answer pair with a
  rationale candidate for QA→R".
- **What is missing**: whether QA→R input uses predicted answer from the Q→A head (end-to-end)
  or GT answer (teacher-forcing standard for VCR).
- **Current decision**: the repo's evaluator `scripts/eval_vcr_qar.py` performs the standard VCR
  post-hoc AND of Q→A and QA→R correctness per sample, which is the community-standard Q→AR
  protocol on VCR's test server. Paper does not contradict this. The training regime for QA→R is
  still underspecified.

### D.2 GQA — inference aggregation
- **Category**: none required (matches paper)
- Paper: argmax over 1842 classes. Implemented in `GQAAccuracy`.

---

## E. Summary: what unblocks what

| Blocker | Unblocks |
|---|---|
| A.1 Tang et al. SG generator | A.6 (partly), VCR full-model runtime |
| A.2 VinVL features | A.4, VCR full-model runtime |
| A.3 ConceptNet retrieval / [8] embeddings | VCR concept-graph branch |
| A.5 VCR raw data | A.1–A.4 local runtime |
| A.6 GQA HDF5 rebuild | GQA full-model runtime |
| A.7 GQA official features | A.6 runtime |
| B.1 merged-graph vs two-subgraph data contract | end-to-end runtime for `PaperVCRModel / PaperGQAModel` |
| C.1 joint VCR mechanics | paper-aligned VCR training |

---

## F. What this repo does NOT claim

- Does not claim `paper-aligned approximation` anywhere in final docs.
- Does not claim `close to paper` or `plausible`.
- Does not claim paper numbers are reproducible — this would require resolving at least A.1–A.7
  and C.1.
- Claims only: (1) paper-equation code for the GNN equations, model head, and LR schedule is
  present and tested; (2) paper-aligned full runtime is `not paper-faithful yet` by the items above.

## G. Final Status

`not paper-faithful yet`.

Reason: a paper-aligned end-to-end run requires the data/architecture pipeline of A.1–A.7 and
B.1, plus the joint training mechanics of C.1. Architecture-only paper-equation code is shipped
and unit-tested; a full-model run remains blocked by the above.
