# Область проекта

Текущий активный scope репозитория:

- `GQA` = основной benchmark и основной training path.
- `VQA-2` = вспомогательный validated extension path.
- удалённый третий benchmark находится вне активного scope репозитория.

Что это значит на практике:

- активные docs, configs, tests, scripts и comments должны описывать только
  GQA и VQA-2 workflows;
- удалённый benchmark не является ни supported benchmark, ни roadmap item,
  ни training target, ни следующим workstream в этом репозитории;
- дальнейшая training и experiment work должна стартовать от GQA и затем
  расширяться до controlled GQA / VQA-2 comparisons.

Базовые source-of-truth runtime-path’ы:

- GQA:
  `GQADataset -> gqa_collate_fn -> GQAVQAGNNModel -> GQALoss -> GQAAccuracy`
- VQA-2:
  `VQADataset -> vqa_collate_fn -> VQAGNNModel -> VQALoss -> VQAAccuracy`

Дисциплина статусов:

- GQA можно описывать как `paper-aligned approximation`.
- VQA-2 нужно описывать как auxiliary validated extension path или
  coursework extension.
- удалённый benchmark нельзя описывать как active, supported или pending.
