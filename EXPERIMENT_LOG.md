# Experiment Log

Дата актуализации: 2026-05-03.

Этот файл фиксирует практические Kaggle baseline runs и наблюдённые метрики.
Это не paper reproduction log и не full-val leaderboard. Если metric получена на
subset, это указывается явно.

## Recorded Baselines

### GQA — Controlled Frozen Subset Baseline

Observed status:

- path: `baseline_gqa.yaml`
- setup: frozen text encoder, relation-aware GQA model, reproducible random subsets
- train subset: `100000`
- val subset: `7000`
- observed `val_GQA_Accuracy`: `0.20`

Reference command family:

```bash
python train.py --config-name baseline_gqa \
  datasets=gqa \
  datasets.train.data_dir=/kaggle/input/datasets/daakifev/gqa-gnn-data \
  datasets.train.answer_vocab_path=/kaggle/input/datasets/daakifev/gqa-gnn-data/gqa_answer_vocab.json \
  datasets.val.data_dir=/kaggle/input/datasets/daakifev/gqa-gnn-data \
  datasets.val.answer_vocab_path=/kaggle/input/datasets/daakifev/gqa-gnn-data/gqa_answer_vocab.json \
  datasets.train.text_encoder_name=/kaggle/input/datasets/daakifev/hf-roberta-large \
  datasets.val.text_encoder_name=/kaggle/input/datasets/daakifev/hf-roberta-large \
  model.text_encoder_name=/kaggle/input/datasets/daakifev/hf-roberta-large \
  dataloader.batch_size=32 \
  datasets.train.limit=100000 \
  datasets.train.shuffle_index=true \
  datasets.train.shuffle_seed=42 \
  datasets.val.limit=7000 \
  datasets.val.shuffle_index=true \
  datasets.val.shuffle_seed=42 \
  trainer.epoch_len=null \
  trainer.n_epochs=15 \
  trainer.save_period=100 \
  writer.run_name=gqa_subset100k_frozen_run \
  writer.mode=online \
  trainer.save_dir=/kaggle/working/saved \
  trainer.override=true
```

Interpretation:

- this is a controlled-subset metric, not a full-val claim;
- use the same subsets when comparing architecture changes.

### VQA-2 — Classic Baseline

Observed status:

- path: `baseline_vqa.yaml`
- setup: classic frozen-text VQA baseline
- observed `val_VQA_Accuracy`: `0.54`

Notes:

- this metric is recorded as an observed practical baseline number;
- exact subset/full-val scope should be kept aligned with the run logs when
  comparing later experiments.

## Graph-Link Negative Results And Rewrite Notes

### VQA-2 — First Graph-Link Variant Regressed

Observed status:

- path family: `graph_link_vqa.yaml`, `graph_link_vqa_frozen.yaml`
- comparison reference: `baseline_vqa.yaml`
- matched practical baseline on VQA-2 remained stronger:
  observed `val_VQA_Accuracy = 0.54`
- early graph-link runs underperformed this baseline and, in one frozen-from-
  scratch setup, produced near-zero train accuracy

What was learned:

- the first graph-link implementation was too aggressive as a pre-GNN
  intervention;
- it directly modified visual and kg node states before the baseline GNN stack,
  which made it easy to disturb a working baseline path;
- frozen runs launched from random initialization were methodologically weak,
  because a tiny trainable graph-link block was being optimized on top of an
  otherwise frozen random baseline;
- full graph-link runs showed that the module was learnable, but the resulting
  extra capacity did not translate into better validation performance.

Most likely causes recorded for future reporting:

1. The first graph-link variant started as a non-identity perturbation and
   could harm baseline node geometry from the first steps.
2. Cross-graph interaction happened too early and too strongly, before the
   existing backbone had a chance to stabilize around it.
3. The first design used top-k learned links without enough filtering or
   relevance calibration, so noisy cross-graph edges could survive.
4. The frozen graph-link experiment is only meaningful when initialized from a
   trained baseline checkpoint; frozen-from-scratch results should not be used
   as evidence against the hypothesis.

Follow-up action taken in the repo:

- `src/model/graph_link.py` was rewritten into a more conservative,
  MAGIC-inspired implicit augmentation block:
  - cosine-based scoring;
  - top-k filtering;
  - dynamic relevance thresholds;
  - bidirectional sparse cross-attention;
  - pooled auxiliary branch representation instead of direct node overwrite.
- after that, the integration strategy was strengthened again:
  - baseline answer head kept intact;
  - graph-link branch now has a separate auxiliary answer head;
  - final prediction uses weighted logit fusion instead of representation
    overwrite.

Interpretation discipline:

- these failed runs should be treated as valid negative experimental evidence
  about the *first* graph-link formulation, not as a final verdict on the
  broader hypothesis about controlled inter-graph integration.

## Experimental Difficulties Recorded

The following difficulties should be kept in the protocol for later reporting.

### Methodological difficulties

1. Controlled comparison required exact matching of:
   - train subset size
   - val subset size
   - shuffle behavior / seed
   - batch size
   - epoch protocol (`epoch_len=null` vs fixed steps)
   - clipping settings

   Early graph-link runs were not always matched perfectly to the baseline,
   which made raw metric comparisons less reliable until the protocol was
   tightened.

2. Frozen graph-link experiments are only interpretable when the frozen
   backbone is initialized from a trained baseline checkpoint. Frozen runs from
   random initialization can collapse to near-zero training accuracy and should
   be treated as setup failures, not model conclusions.

3. The first graph-link formulation intervened too early in the pipeline
   (pre-GNN node rewriting), which made it difficult to separate:
   - useful cross-graph integration
   - harmful perturbation of an already working baseline

4. Later rewrites showed that making the module safer can also make it too weak
   to provide an immediate measurable gain. This created a recurring tradeoff
   between stability and effect size.

### Engineering difficulties

1. Kaggle checkpoint saving for large runs was unstable under the default
   PyTorch zip serialization and required a hardened save path.

2. Resume / checkpoint loading required explicit compatibility handling for
   PyTorch 2.6 (`weights_only=False`) and config serialization cleanup.

3. Hydra-based optimizer/scheduler wiring needed extra normalization and
   conversion so custom param groups and epoch-based schedulers worked in both
   GQA and VQA-2 paths.

4. Controlled subset experiments required explicit config support for:
   - `limit`
   - `shuffle_index`
   - `shuffle_seed`
   so that repeated runs used the same data slices.

### Current protocol lesson

For future architectural experiments, the safest reporting order is:

1. matched baseline control
2. frozen-baseline + new module
3. full baseline + new module

This ordering gives the clearest interpretation and catches setup problems
before expensive full-train comparison runs.
