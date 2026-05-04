# GQA: итоговый отчёт по инженерной фазе

Дата: 2026-05-04.

Статус: GQA engineering-path находится в состоянии `implemented`,
`validated` и готов к practical Kaggle training experiments. Numeric
training-results всё ещё `not validated yet`.

Примечание по scope репозитория:

- GQA — основной benchmark.
- VQA-2 — вспомогательный extension path.
- удалённый третий benchmark находится вне активного scope репозитория.

## Запускаемый путь

| Layer | Class / config |
|---|---|
| Model | `src/model/gqa_model.py::GQAVQAGNNModel` |
| Dataset | `src/datasets/gqa_dataset.py::GQADataset` |
| Collate | `src/datasets/gqa_collate.py::gqa_collate_fn` |
| Hydra train | `src/configs/baseline_gqa.yaml` |
| Hydra inference | `src/configs/inference_gqa.yaml` |

Статус: `paper-aligned approximation`. Это единственный поддерживаемый
Kaggle baseline path.

## Проверенный data contract

- `answer_to_idx` size = 1842, contiguous, coverage 100%.
- `relation_count_total` = 624, contiguous.
- train: 943,000 questions, 72,140 images, full image coverage.
- val: 132,062 questions, 10,234 images, full image coverage.
- visual feature shape: `float32[100, 2048]`.
- graph `node_features.shape[-1] = 600`.
- graph attrs include `d_kg=600`, `num_visual_nodes=100`,
  `max_kg_nodes=100`, `graph_edge_type_count=624`,
  `graph_mode=official_scene_graph`, `conceptnet_used=False`,
  `fully_connected_fallback_used=False`.
- metadata package exists; no-match rate about `9.5%` train / `9.6%` val.

## Runtime-проверка

- `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel(num_relations=624) -> GQALoss -> GQAAccuracy`: `validated`.
- `GQAVQAGNNModel` consumes `graph_edge_types` through DenseGAT relation bias:
  `validated`.
- `baseline_gqa.yaml` targets `GQAVQAGNNModel`: `validated`.

Runtime sanity-check на восстановленных real Kaggle data:

- logits shape: `(1, 1842)`;
- loss finite: `7.326373`;
- random-init loss close to `ln(1842) = 7.519`.

## Зафиксированные run’ы

### 2026-05-03 — controlled GQA subset baseline

Замысел:

- fixed random `train` subset: `100000`
- fixed random `val` subset: `7000`
- frozen text encoder baseline
- full subset epochs via `trainer.epoch_len=null`
- Kaggle online logging

Команда, использованная на Kaggle:

```bash
cd /kaggle/working/repo/ && COMET_API_KEY="<set via Kaggle Secrets or env>" python train.py --config-name baseline_gqa \
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

Примечания:

- `model_best.pth` is intended to be overwritten in-place, not accumulated.
- For Kaggle stability, checkpoint saving was later hardened with atomic temp-file
  writes and legacy PyTorch serialization.
- Наблюдённая controlled-subset validation accuracy для этого семейства run’ов:
  `val_GQA_Accuracy = 0.20`.

### 2026-05-04 — controlled GQA improvement с Graph-Link

Замысел:

- keep the same controlled subset regime;
- preserve the baseline answer path;
- use the graph-link module as an auxiliary branch instead of rewriting the
  baseline graph states;
- fuse the branch only at the answer-logit level.

Использованная graph-link стратегия:

- question-conditioned sparse inter-graph links;
- bidirectional sparse cross-attention;
- pooled `link_repr`;
- separate graph-link answer head;
- weighted fusion:
  `final_logits = baseline_logits + alpha * graph_link_logits`.

Наблюдённый результат:

- controlled-subset `val_GQA_Accuracy`: `0.30+`
  (approximately `0.31` on the recorded Kaggle run).

Интерпретация:

- это самый сильный controlled-subset результат GQA, который сейчас
  зафиксирован в репозитории;
- относительно более раннего controlled baseline (`0.20`) это существенный
  положительный сигнал в пользу late-fusion graph-link идеи;
- это всё ещё controlled-subset result, и его нельзя подавать как full
  paper reproduction.

## Что остаётся открытым

- Real GQA train/eval numbers: `not validated yet`.
- Kaggle budget fit: `not validated yet`.
- Exact paper training hyperparameters: `paper-aligned approximation`.
- Текущие Kaggle defaults намеренно замораживают text encoder, используют
  `d_hidden=512`, and limit validation to 10k samples.
- Possible future engineering: AMP, gradient accumulation, and timed Kaggle
  profiling.

## Вне текущего scope

VQA-2 has been restored as a separate coursework extension via
`baseline_vqa.yaml`, but it is not part of this GQA closure claim and remains
`not validated yet` on real data.
