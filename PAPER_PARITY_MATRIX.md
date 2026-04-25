# Paper Parity Matrix

Source paper: VQA-GNN (Wang, Yasunaga, Ren, Wada, Leskovec — ICCV 2023, arXiv:2205.11501).
This file extracts paper spec directly from the ICCV 2023 camera-ready PDF (quoted sections
indicated as §N). For every item the status is one of **only two** valid states:

- `paper-aligned` — exact paper form is implemented and verified by a test or a precise spec
  reference.
- `not paper-faithful yet` — exact paper form cannot honestly be implemented in this repo; a precise reason
  is recorded (missing paper detail, unavailable external artifact, unspecified training detail,
  inaccessible original code, or external dependency).

No entry may be labelled `paper-aligned approximation`, `close to paper`, or similar. If an entry
would need such a label, it is instead `not paper-faithful yet` with the specific reason.

## 1. Multimodal Semantic Graph (§4.1)

| # | Paper requirement | Current impl | Status | Reason / test |
|---|---|---|---|---|
| 1.1 | Node types T = {Z, P, S, C} plus question node q (VCR: z, p, s, c; GQA: q, s, c) | Merged graph with 3 node-type ids {0=visual, 1=question, 2=kg} | `not paper-faithful yet` | **unavailable external artifact**: paper-aligned graph requires separated scene-graph and concept-graph subgraphs with explicit super-node ids; current offline HDF5 pipeline produces a single merged graph. Rebuild path documented but blocked by data (§ BLOCKED_EXACTNESS_ITEMS.md §A.1). |
| 1.2 | Scene-graph nodes from Tang et al. unbiased SG generator [38] (recall@20) | VCR: visual nodes from pre-extracted features; GQA: official scene graphs | VCR `not paper-faithful yet`, GQA `paper-aligned` (preprocessing code) | **unavailable external artifact**: Tang et al. checkpoint not shipped; recall@20 thresholding not verifiable. GQA path uses paper-specified official scene graphs via `src/datasets/gqa_preprocessing.py`; runtime is `not paper-faithful yet` with current local `data/gqa/knowledge_graphs/*.h5` because the artifacts are not rebuilt with `d_kg=600`. |
| 1.3 | Concept-graph: max N = 60 nodes | Runtime uses `max_nodes` from dataset config; no hard cap tied to 60 | `not paper-faithful yet` | **missing paper detail**: paper says "maximum number of concept-graph nodes of 60" but does not describe the selection/truncation procedure sufficiently to reproduce exactly. |
| 1.4 | Concept entity embedding: 1024-d text feature from [8] (Feng et al.) | 300-d ConceptNet Numberbatch (VCR) / 600-d GloVe-concat (GQA) | `not paper-faithful yet` | **inaccessible original code**: [8]'s 1024-d RoBERTa-based relation encoding is external and not released as part of VQA-GNN code. Local ConceptNet Numberbatch is a different embedding space; claiming it as "exact" would be dishonest. |
| 1.5 | Concept-graph edges: D-dim one-hot over D edge types | GQA: runtime typed edges via `graph_edge_types` Tensor[B,N,N] (long ids); VCR: untyped | GQA `paper-aligned` on edge-id encoding, VCR `not paper-faithful yet` | **unavailable external artifact** for VCR: ConceptNet-retrieved concept graph not built locally with typed edges. |
| 1.6 | QA-context node z (VCR): pretrained RoBERTa-Large, finetuned jointly with GNN | `HFQuestionEncoder` defaults to `roberta-large`, finetunable | `paper-aligned` | `src/model/gnn_core.py::HFQuestionEncoder`; configs `src/configs/model/vqa_gnn_vcr.yaml`, `src/configs/model/vqa_gnn_gqa.yaml`, `src/configs/model/paper_vqa_gnn_vcr.yaml`, `src/configs/model/paper_vqa_gnn_gqa.yaml`. |
| 1.7 | QA-concept node p (VCR): mean-pooled VinVL features of retrieved VG regions | No explicit p-node; visual features act as scene-graph visual nodes | `not paper-faithful yet` | **unavailable external artifact**: VinVL [53] checkpoint not shipped; VisualGenome retrieval by QA context not built. |
| 1.8 | QA-context ↔ scene-graph edges: `r(e)` fully connects z with all V(s) | Merged graph uses `graph_adj`; in current node-type scheme question↔visual edges are present in the adjacency | `not paper-faithful yet` | **cannot be labeled exact without separating S-subgraph and C-subgraph and running dedicated modality-GNNs per §4.2**; see row 1.1. |
| 1.9 | QA-context ↔ concept-graph edges: `r(q)` and `r(a)` tied to extracted entities | Merged adjacency does not distinguish `r(q)` vs `r(a)` | `not paper-faithful yet` | **missing paper detail** on entity extraction procedure + **unavailable external artifact** for entity-linked concept graph. |
| 1.10 | GQA: question node q, visual SG, textual SG; no QA-concept node p | Current model uses node-type ids {0,1,2} on merged graph, not two modality subgraphs | `not paper-faithful yet` | **unavailable external artifact**: paper-aligned GQA graph requires two separated SG subgraphs connected via q; current preprocessing produces merged graph only. |

## 2. Multimodal GNN — Bidirectional Fusion (§4.2)

Paper equations (quoted from §4.2):

- (1) `r_ij = f_r([e_ij || u_i || u_j])`, f_r a 2-layer MLP; e_ij one-hot over |R|, u one-hot over |T|.
- (2) `m_ij = f_m([h_i^{k+1} || r_ij])`, f_m linear, R^{2D}→R^D.
- (3) `h_i^{k+1} = f_h(Σ_{j∈N_i} α_ij m_ij) + h_i^{k}`, f_h a 2-layer MLP with **batch normalisation**.
- (4) `q_i = f_q(h_i^{k+1})`, `k_j = f_k([h_j^{k+1} || r_ij])`; f_q linear R^D→R^D, f_k linear R^{2D}→R^D.
- (5) `γ_ij = q_i^T k_j / sqrt(D)`.
- (6) `α_ij = softmax_j(γ_ij)`.
- (7) For z: `h_z^{k+1} = f_z([h_{z(s)}^{k+1} || h_{z(c)}^{k+1}])`, f_z linear R^{2D}→R^D.
- Two **modality-specialized** GNNs (one for S, one for C) — Ablation Table 3 and §4.2.

| # | Paper requirement | Current impl | Status | Reason / test |
|---|---|---|---|---|
| 2.1 | Relation embedding built from concat of (one-hot edge id, one-hot source node-type, one-hot target node-type) via 2-layer MLP (Eq 1) | `DenseGATLayer`: learned per-(head, relation) scalar bias directly added to attention logits. No edge-id MLP, no node-type concat | `not paper-faithful yet` in `DenseGATLayer`; `paper-aligned` in new `src/model/paper_rgat.py::PaperMultiRelationGATLayer` | Paper equation implemented verbatim in `PaperMultiRelationGATLayer`; unit test `tests/test_paper_rgat.py::test_relation_embedding_mlp_shapes`. The current `DenseGATLayer` is kept for the legacy single-GNN model path (explicit Ablation 2 of Table 3, `not paper-faithful yet` for full runtime). |
| 2.2 | Message m_ij = f_m([h_i || r_ij]) (Eq 2) | Current GAT uses standard V projection; no relation-concat into message | `not paper-faithful yet` in `DenseGATLayer`; `paper-aligned` in `PaperMultiRelationGATLayer` | test `tests/test_paper_rgat.py::test_message_key_and_residual_shapes`. |
| 2.3 | Aggregation f_h is 2-layer MLP with batch normalisation (Eq 3) | Current impl: LayerNorm + GELU FFN (pre-norm transformer block) | `not paper-faithful yet` in `DenseGATLayer`; `paper-aligned` in `PaperMultiRelationGATLayer` | test `tests/test_paper_rgat.py::test_fh_has_batchnorm_and_two_layers`. |
| 2.4 | Residual h^{k+1} = f_h(aggr) + h^k (post-aggregation residual) (Eq 3) | Current impl residual pattern is transformer-style pre-norm; does not match | `not paper-faithful yet` in `DenseGATLayer`; `paper-aligned` in `PaperMultiRelationGATLayer` | tests `tests/test_paper_rgat.py::test_message_key_and_residual_shapes`, `tests/test_paper_rgat.py::test_forward_shape_and_backward_no_nan`. |
| 2.5 | Attention scoring q^T k / sqrt(D) with k built from concat(h_j, r_ij) (Eqs 4–5) | Current impl: standard multi-head QKV; k does not see r_ij | `not paper-faithful yet` in `DenseGATLayer`; `paper-aligned` in `PaperMultiRelationGATLayer` | test `tests/test_paper_rgat.py::test_message_key_and_residual_shapes`. |
| 2.6 | Softmax over neighbors (Eq 6); isolated-node safety | `DenseGATLayer` masks non-edges with -inf and replaces NaN rows with 0 | `paper-aligned` (in `PaperMultiRelationGATLayer` with the same isolated-node guard) | test `tests/test_paper_rgat.py::test_isolated_nodes_are_finite`. |
| 2.7 | Two modality-specialized GNNs — not a single GNN on the merged graph | Current models run a single GNN on the merged graph | `not paper-faithful yet` on runnable path; `paper-aligned` in model-code-only form via `src/model/paper_vqa_gnn.py::PaperVCRModel / PaperGQAModel` | **unavailable external artifact** for runtime (see 1.1). Paper-equation model code provided; runtime blocked by data pipeline. |
| 2.8 | z-fuse across two GNNs (Eq 7: `h_z^{k+1} = f_z([h_{z(s)} || h_{z(c)}])`) | Not present in legacy models | `not paper-faithful yet` on runnable path; `paper-aligned` in `PaperVCRModel / PaperGQAModel` | tests `tests/test_paper_vqa_gnn_model.py::test_vcr_head_is_concat_4D_to_1`, `tests/test_paper_vqa_gnn_model.py::test_gqa_head_is_concat_3D_to_vocab`. |

## 3. Inference & Learning (§4.3)

Paper equations:

- (8) `h_a^{k+1} = [h_s^{K+1} || h_c^{K+1} || h_p^{K+1} || h_z^{K+1}]`
- (9) `logit(a) = f_c(h_a^{k+1})`, f_c linear R^{4D}→R^1 (scalar per candidate).
- (10) `p(a|c,q) = softmax_a(logit(a))` over the 4 candidates (VCR) or 1842 classes (GQA).
- Loss: cross-entropy.

| # | Paper requirement | Current impl | Status | Reason / test |
|---|---|---|---|---|
| 3.1 | VCR readout: concat of `[pool(S), pool(C), p, z]` then linear → scalar (Eq 8–9) | Legacy models: attention-pooled single graph rep → MLP → 1 | `not paper-faithful yet` on legacy model; `paper-aligned` in `PaperVCRModel.vcr_head` | test `tests/test_paper_vqa_gnn_model.py::test_vcr_head_is_concat_4D_to_1`. Runtime is `not paper-faithful yet` because p-node features are missing (row 1.7). |
| 3.2 | GQA readout: concat of modality-pooled reps + question node, classify → 1842 | Legacy models: attention-pooled merged graph → MLP → 1842 | `not paper-faithful yet` on legacy model; `paper-aligned` in `PaperGQAModel.gqa_head` | test `tests/test_paper_vqa_gnn_model.py::test_gqa_head_is_concat_3D_to_vocab`. Runtime is `not paper-faithful yet` until the two-SG-subgraph data contract is available. |
| 3.3 | Softmax over candidates / answer vocab (Eq 10) | Loss uses `F.cross_entropy` which subsumes softmax | `paper-aligned` | tests `tests/test_paper_path.py` (existing). |
| 3.4 | Cross-entropy loss | `VCRLoss`, `GQALoss` both CE | `paper-aligned` | existing tests. |

## 4. VCR Training Protocol (§5.1)

Paper statements (quoted):

- "We joint train VQA-GNN on Q→A and QA→R, with a common LM encoder"
- "We use a pretrained RoBERTa Large model"
- "finetune it with the multimodal GNN for 50 epoch by using learning rates 1e-5 and 1e-4 respectively" (RoBERTa=1e-5, GNN=1e-4)
- "We set the number of layers (L = 5) of VQA-GNN and use AdamW optimizer"
- "linear warmup of the learning rate over the 15-th epoch, with a cosine decay thereafter to 0"

| # | Paper requirement | Current impl | Status | Reason / test |
|---|---|---|---|---|
| 4.1 | Joint training Q→A + QA→R with common LM encoder | Separate configs `baseline_vcr_qa.yaml` and `baseline_vcr_qar.yaml`, two independent models | `not paper-faithful yet` | **missing paper detail**: paper does not specify joint-task loss weighting, batch composition, or per-task-step scheduling. Current single-task trainer cannot express joint training without reimplementation; paper leaves the joint-training mechanics underspecified (see `TRAIN_PROTOCOL_PARITY.md §1`). |
| 4.2 | RoBERTa-Large | Default in both model configs | `paper-aligned` | configs `vqa_gnn_vcr.yaml`, `vqa_gnn_gqa.yaml`. |
| 4.3 | Number of GNN layers L = 5 (VCR) | Legacy configs set `num_gnn_layers: 3` | `paper-aligned` via new paper-equation config `src/configs/paper_vqa_gnn_vcr.yaml` (L=5); legacy configs marked non-paper-aligned ablation | commit marks legacy as Ablation 2 variant. |
| 4.4 | 50 epochs | Legacy configs set `n_epochs: 10` | `paper-aligned` in new paper-equation config; legacy remains reduced for smoke-test practicality | `paper_vqa_gnn_vcr.yaml: n_epochs: 50`. |
| 4.5 | Optimizer AdamW | Legacy configs already AdamW | `paper-aligned` | configs. |
| 4.6 | LR 1e-5 (LM) and 1e-4 (GNN) | Legacy uses single LR 2e-5 | `paper-aligned` in new paper-equation config + `paper_param_groups` utility | test `tests/test_paper_scheduler_and_optim.py::test_paper_param_groups_split_lm_and_gnn_params`. |
| 4.7 | Linear warmup over 15 epochs + cosine decay to 0 | Legacy uses `StepLR` | `paper-aligned` in new scheduler `src/lr_schedulers/paper_warmup_cosine.py`; used by paper-equation config | test `tests/test_paper_scheduler_and_optim.py::test_paper_warmup_cosine_schedule_values`. |
| 4.8 | Batch size | Not specified in paper | `not paper-faithful yet` | **missing paper detail**. |
| 4.9 | Gradient accumulation / max sequence length | Not specified in paper | `not paper-faithful yet` | **missing paper detail**. |
| 4.10 | Dropout rate | Not specified in paper | `not paper-faithful yet` | **missing paper detail**. |

## 5. GQA Training Protocol (§5.1)

Paper statements:

- "We define the question as the context node (node q)"
- "node q is embedded with a pretrained RoBERTa large model"
- "initialize object nodes' representations in visual SG using official object features"
- "object nodes in textual SG by concatenating GloVe [31] based word embedding of the object name and attributes"
- "classify the given image-question pair out of 1,842 answer classes"
- "finetune the node q with VQA-GNN for 50 epoch by using learning rates 2e-5 and 2e-4 respectively"

| # | Paper requirement | Current impl | Status | Reason / test |
|---|---|---|---|---|
| 5.1 | Question-as-context node q | Merged graph has one question-type node (id=1) | `not paper-faithful yet` (for full-model sense) | requires two-SG-subgraph structure (row 1.10). |
| 5.2 | Visual SG node features: official GQA object features | Runtime ingests HDF5 `visual_features`; preprocessing code supports the paper path | `paper-aligned` (preprocessing code); runtime `not paper-faithful yet` | **unavailable external artifact**: local `data/gqa/visual_features/*.h5` not produced by paper-path pipeline (see `DATA_CONTRACTS.md`). |
| 5.3 | Textual SG node features: concat(GloVe object name, GloVe attribute(s)) → 600-d (2×300) | `d_kg=600` in config; preprocessing has `paper_aligned_build_object_graph` | `paper-aligned` (preprocessing code); runtime `not paper-faithful yet` on local artifacts | `src/datasets/gqa_preprocessing.py`; local HDF5 has `(N, 300)` (`blocked by data`). |
| 5.4 | Answer vocab: 1842 classes (balanced split) | `num_answers: 1842`; `gqa_answer_vocab.json` built via `prepare_gqa_data.py` | `paper-aligned` | `src/configs/model/vqa_gnn_gqa.yaml`. |
| 5.5 | 50 epochs | Legacy: `n_epochs: 20` | `paper-aligned` in new paper-equation config | `src/configs/paper_vqa_gnn_gqa.yaml`. |
| 5.6 | LR 2e-5 (LM) + 2e-4 (GNN) | Legacy single LR 1e-4 | `paper-aligned` in new paper-equation config | same. |
| 5.7 | AdamW | yes | `paper-aligned` | same. |
| 5.8 | Warmup + cosine schedule | Paper specifies this for VCR. GQA schedule not explicitly stated — only LRs and 50 epochs | `not paper-faithful yet` | **missing paper detail** — paper's GQA section does not reassert the warmup length. Paper-equation GQA config reuses VCR's scheduler form but flags it as underspecified. |
| 5.9 | Batch size, dropout, warmup-length for GQA | Not specified in paper | `not paper-faithful yet` | **missing paper detail**. |

## 6. Text Encoder

| # | Paper requirement | Current impl | Status | Reason / test |
|---|---|---|---|---|
| 6.1 | RoBERTa Large via `transformers.AutoModel` | yes | `paper-aligned` | `src/model/gnn_core.py::HFQuestionEncoder`. |
| 6.2 | Finetuned jointly with GNN (not frozen) | `freeze_text_encoder: False` default | `paper-aligned` | configs. |

## 7. Visual Features

| # | Paper requirement | Current impl | Status | Reason / test |
|---|---|---|---|---|
| 7.1 | VCR visual: VinVL [53] (Table 2 legend "Node p (Vinvl)") | Pre-extracted 36×2048 features assumed (Bottom-Up style) | `not paper-faithful yet` | **unavailable external artifact**: VinVL checkpoint + extraction pipeline not shipped; local VCR visual HDF5 not present. |
| 7.2 | GQA visual: official object features | Runtime reads HDF5; preprocessing code aligned | `paper-aligned` on preprocessing code; runtime `not paper-faithful yet` | local HDF5 `attrs={}`, not produced by paper path. |

## 8. Loss & Metrics

| # | Paper requirement | Current impl | Status |
|---|---|---|---|
| 8.1 | VCR: softmax over 4 candidates + CE | `VCRLoss` reshapes [B*4,1]→[B,4] then `F.cross_entropy` | `paper-aligned` |
| 8.2 | VCR: Q→A accuracy | `VCRQAAccuracy` | `paper-aligned` |
| 8.3 | VCR: QA→R accuracy | `VCRQARAccuracy` | `paper-aligned` |
| 8.4 | VCR: Q→AR joint accuracy (per-sample AND of QA and QAR correctness) | `VCRQARJointAccuracy` + post-hoc `scripts/eval_vcr_qar.py` | `paper-aligned` |
| 8.5 | GQA: hard CE | `GQALoss` | `paper-aligned` |
| 8.6 | GQA: exact-match accuracy | `GQAAccuracy` | `paper-aligned` |

## 9. Object Grounding (VCR)

| # | Paper requirement | Current impl | Status | Reason |
|---|---|---|---|---|
| 9.1 | Object mentions resolved to VG regions via VinVL-pooled p-node | Per-instance identifier strings (`person0`, `person2`) injected into text | `not paper-faithful yet` | **unavailable external artifact**: VinVL region pooling not built; paper underspecifies tokenization of object refs in the QA text. |

## 10. ConceptNet Retrieval (VCR concept graph)

| # | Paper requirement | Current impl | Status | Reason |
|---|---|---|---|---|
| 10.1 | ConceptNet subgraph retrieved for each QA; local-entity expansion when adjacent | Offline HDF5 pre-built | `not paper-faithful yet` | **missing paper detail**: paper's Fig 3 shows `Relev(e|a)` but formula and top-k thresholds are not fully specified. |
| 10.2 | D-dim one-hot relation edges where D = number of edge types in concept graph | Runtime carries `graph_edge_types` for GQA only | `not paper-faithful yet` | **unavailable external artifact**: VCR concept-graph relation vocabulary not built locally. |

---

## Summary

Exact paper form is recovered (verified by test) for:

- text encoder choice, freeze policy
- answer-vocab size, exact-match metric, hard CE for GQA
- VCR candidate-scoring CE + accuracies + post-hoc Q→AR evaluator
- paper GNN equations (new `PaperMultiRelationGATLayer` — Eqs 1–7)
- paper readout (new `PaperVCRModel / PaperGQAModel` — Eq 8)
- L = 5, n_epochs = 50, paper LR warmup+cosine schedule, per-param-group LRs (new paper-equation configs)
- preprocessing code for GQA (official SG, GloVe-concat object features, 1842 vocab)

Blocked items fall into four strict categories:

- **unavailable external artifact**: VinVL features, Tang et al. SG-generator outputs, QA-GNN 1024-d concept embeddings, local GQA `d_kg=600` HDF5 + `graph_edge_types`, local VCR data.
- **missing paper detail**: joint VCR training mechanics, concept-graph node cap selection, entity-linking algorithm, batch size, dropout, D, number of heads, GQA scheduler specifics.
- **unspecified training detail**: gradient accumulation, max sequence length, init seeds, warm-up curve shape (linear vs constant in warmup subinterval).
- **inaccessible original code**: the `fr/fm/fh/fq/fk/fc/fz` MLP widths and activations are inferred from Eq 1–7 dim signatures; any specific hidden-layer sizes beyond those dims are inaccessible.

See `BLOCKED_EXACTNESS_ITEMS.md` for the full blocker list with reproduction requirements.
