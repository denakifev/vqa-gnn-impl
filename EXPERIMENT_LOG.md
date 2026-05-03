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
