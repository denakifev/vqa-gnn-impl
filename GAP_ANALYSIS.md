# Анализ разрывов

Дата актуализации: 2026-05-04.

Статус репозитория: `partially validated baseline`.

Новая цель: practical GQA и VQA-2 train, которые можно запускать в
Kaggle-лимитах. GQA сохраняет `paper-aligned approximation`; VQA-2 является
coursework extension и не должен подаваться как результат статьи.

Активный режим репозитория:

- GQA main
- VQA-2 auxiliary
- no removed benchmark in active scope

## Что реализовано / проверено

- GQA 1842-way classification: `implemented`, `validated`.
- GQA relation-aware message passing via `graph_edge_types`: `implemented`,
  `validated`, `paper-aligned approximation`.
- GQA additive sparse graph-link module for visual<->textual/kg exchange with
  separate auxiliary answer head and weighted logit fusion:
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
- Зафиксированные practical baseline-результаты:
  GQA controlled subset `val_GQA_Accuracy = 0.20`,
  GQA graph-link controlled subset `val_GQA_Accuracy = 0.30+`,
  VQA-2 classic baseline `val_VQA_Accuracy = 0.54`,
  VQA-2 graph-link late-fusion `val_VQA_Accuracy = 0.55`.

## Оставшиеся GQA-разрывы

| Разрыв | Статус | Примечания |
|---|---|---|
| Real GQA training/eval numbers | `not validated yet` | Требуется фактический train run |
| Kaggle-friendly protocol | `not validated yet` | Config defaults практичны, но нужен timed Kaggle run |
| Точная длина warmup, batch size, dropout, max seq len, grad accumulation | `paper-aligned approximation` | В статье это недоописано |
| Hidden widths / activations в relation и node MLP | `paper-aligned approximation` | `d_hidden=512` по умолчанию выбран ради Kaggle practicality |
| Разделение relation vocab между visual и textual scene graph | `paper-aligned approximation` | Используется shared vocab на 624 relation |
| Entity linking через tokenization | `paper-aligned approximation` | Практический preprocessing choice |
| Точное происхождение checkpoint `RoBERTa-large` | `paper-aligned approximation` | Используется публичный checkpoint по умолчанию |
| Прирост graph-link относительно baseline | `paper-aligned approximation` | На controlled GQA subset уже есть сильный сигнал (`0.20 -> 0.30+`), а на VQA-2 зафиксирован небольшой положительный сдвиг (`0.54 -> 0.55`); более широкая валидация вне этих протоколов ещё открыта |
| Отдельный внешний concept graph в активном GQA path | `not paper-faithful yet` | Текущий graph-link module работает с visual nodes и уже существующими textual/kg nodes, а не с восстановленной отдельной ConceptNet branch |
| Первая formulation graph-link на VQA-2 | `not validated yet` | Ранние run’ы уступали зафиксированному baseline; позже модуль был переписан в более conservative MAGIC-inspired variant |

## Оставшиеся VQA-2-разрывы

| Разрыв | Статус | Примечания |
|---|---|---|
| Real VQA-2 data package | `not validated yet` | Ожидаемая структура описана в `DATA_CONTRACTS.md` |
| Real VQA-2 training/eval numbers | `not validated yet` | Сначала нужен `scripts/validate_vqa_data.py`, затем фактический train run |
| Типизация relations в VQA-2 | `not paper-faithful yet` | Текущий VQA-2 graph contract использует untyped adjacency |
| Claim по VQA-2 относительно статьи | `not validated yet` | Это только coursework extension; в статье этот путь не валидировался |

## Практические риски обучения

- `baseline_gqa.yaml` сейчас представляет собой небольшой Kaggle-practical profile:
  `batch_size=4`, `epoch_len=1000`, `n_epochs=3`, frozen text encoder,
  `d_hidden=512`, and validation limited to 10k samples.
- `baseline_vqa.yaml` использует тот же Kaggle-practical shape: batch 4,
  frozen text encoder, `d_hidden=512`, 3 epochs, 1000 steps per epoch, and
  validation limited to 10k samples.
- Trainer currently has no AMP and no gradient accumulation.
- Валидационные метрики являются приближёнными, когда `datasets.val.limit=10000`.
- Unfreezing RoBERTa-large or raising `d_hidden` may exceed the available
  notebook GPU budget.
- `graph_link_gqa_frozen` is the recommended first controlled research mode if
  the goal is to isolate the contribution of the new module.
- For graph-link experiments, frozen runs are only meaningful when initialized
  from a trained baseline checkpoint; frozen-from-scratch runs can collapse and
  should be treated as setup failures, not architecture conclusions.
- `graph_link_gqa_frozen_warm` is now the preferred causal-attribution regime:
  it isolates the graph-link branch while keeping the baseline backbone fixed
  at the trained `0.20` GQA checkpoint.
- Comparing graph-link variants to baseline requires fully matched protocol:
  subset sizes, shuffle regime, batch size, clipping, and epoch definition all
  materially affect the interpretation.

## Рекомендуемый первый запуск

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

## Дисциплина claim’ов

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
