# Журнал экспериментов

Дата актуализации: 2026-05-04.

Этот файл фиксирует практические Kaggle baseline runs и наблюдённые метрики.
Это не paper reproduction log и не full-val leaderboard. Если metric получена на
subset, это указывается явно.

## Исторические вехи

### GQA — ранние практические вехи до controlled R&D

Наблюдавшиеся практические вехи из ранних Kaggle-ориентированных GQA run’ов:

- frozen baseline family: best observed `val_GQA_Accuracy ≈ 0.176`
- unfrozen / paper-closer practical family: best observed
  `val_GQA_Accuracy ≈ 0.190`

Эти run’ы были полезны как диагностика оптимизации:

- unfreezing the text path helped relative to the earliest frozen baseline;
- however, those runs were still noisy and not ideal for controlled module
  comparison;
- they motivated the later switch to fixed subset protocol.

## Зафиксированные baseline’ы

### GQA — controlled frozen subset baseline

Наблюдённый статус:

- path: `baseline_gqa.yaml`
- setup: frozen text encoder, relation-aware GQA model, reproducible random subsets
- train subset: `100000`
- val subset: `7000`
- observed `val_GQA_Accuracy`: `0.20`

Опорное семейство команд:

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

Интерпретация:

- this is a controlled-subset metric, not a full-val claim;
- use the same subsets when comparing architecture changes.

### GQA — run с Graph-Link и logit fusion

Наблюдённый статус:

- path: `graph_link_gqa.yaml`
- setup: same controlled subset family as the recorded GQA baseline
- train subset: `100000`
- val subset: `7000`
- module regime:
  - sparse question-conditioned inter-graph links
  - bidirectional sparse cross-attention
  - separate auxiliary graph-link answer head
  - weighted logit fusion with the untouched baseline head
- observed `val_GQA_Accuracy`: `0.30+` (approximately `0.31` on the recorded
  run chart)

Интерпретация:

- this is the strongest controlled GQA signal observed so far in the repo;
- relative to the recorded controlled baseline (`0.20`), this is a large
  positive gain on the same experiment family;
- this should still be described as a controlled-subset result, not as a full
  paper reproduction claim.

### GQA — stratified analysis by question type

Наблюдённый статус:

- analysis artifact directory: `exp2`
- comparison: `baseline_gqa` vs `graph_link_gqa`
- common questions: `7000`
- overall baseline accuracy in this analysis run: `0.1796`
- overall graph-link accuracy in this analysis run: `0.3066`
- overall delta: `+0.1270`
- bootstrap 95% CI: `[+0.1156, +0.1386]`

Краткие результаты по bucket'ам:

- `compare`: `0.4112 -> 0.5514`, delta `+0.1402`
- `logical`: `0.5334 -> 0.5043`, delta `-0.0292`
- `choose`: `0.0000 -> 0.3391`, delta `+0.3391`
- `verify`: `0.5038 -> 0.5031`, delta `-0.0007`
- `query`: `0.0000 -> 0.1592`, delta `+0.1592`

Интерпретация:

- stratified analysis подтверждает, что новый модуль даёт не просто общий
  прирост, а выраженный прирост на части question-type bucket'ов;
- заметный gain на `compare` хорошо согласуется с основной гипотезой о пользе
  controlled cross-graph bridging;
- `verify` почти не меняется, а `logical` даже слегка проседает, так что
  улучшение не является равномерным по всем типам вопросов;
- крупные сдвиги на `choose` и `query` выглядят многообещающе, но требуют
  дополнительной sanity-check интерпретации, потому что baseline в этих
  bucket'ах в текущем анализе оказался аномально слабым;
- в целом это удачный и важный интерпретационный эксперимент: он делает
  успех на GQA содержательно объяснимым, а не только численным.

### GQA — следующий обязательный каузальный эксперимент

Следующий приоритетный R&D шаг:

- path: `graph_link_gqa_frozen_warm.yaml`
- baseline warm-start: trained GQA baseline checkpoint with observed
  `val_GQA_Accuracy = 0.20`
- freeze regime: full baseline frozen
- trainable components:
  - link scorer;
  - sparse cross-attention;
  - pooled `link_repr`;
  - `graph_link_classifier`;
  - scalar `graph_link_alpha`

Зачем это нужно:

- текущий `0.20 -> 0.30+` результат получен при совместном обучении baseline и
  graph-link branch;
- warm-start frozen regime позволит отделить вклад самого модуля от
  co-adaptation baseline backbone;
- если результат останется высоким (`~0.27-0.30`), это станет главным
  аргументом в пользу модуля как plug-in graft поверх фиксированного baseline;
- если результат откатится ближе к baseline, это тоже останется сильным
  исследовательским выводом о существенной роли joint adaptation.

### VQA-2 — классический baseline

Наблюдённый статус:

- path: `baseline_vqa.yaml`
- setup: classic frozen-text VQA baseline
- observed `val_VQA_Accuracy`: `0.54`

Примечания:

- this metric is recorded as an observed practical baseline number;
- exact subset/full-val scope should be kept aligned with the run logs when
  comparing later experiments.

### VQA-2 — Graph-Link с late logit fusion

Наблюдённый статус:

- path: `graph_link_vqa.yaml`
- comparison reference: `baseline_vqa.yaml`
- module regime:
  - sparse question-conditioned inter-graph links;
  - bidirectional sparse cross-attention;
  - separate auxiliary graph-link answer head;
  - weighted logit fusion with the untouched baseline head
- observed `val_VQA_Accuracy`: `0.55`
  (примерно на `+0.01` выше зафиксированного baseline `0.54`)

Интерпретация:

- это уже не регрессия, а небольшой положительный сигнал;
- эффект на VQA-2 заметно слабее, чем на GQA;
- правдоподобная рабочая интерпретация состоит в том, что для VQA-2
  controlled cross-graph reasoning может быть менее критичен, чем для GQA;
- этот вывод стоит подавать как практическую гипотезу по текущему evidence,
  а не как окончательное утверждение о свойствах датасета.
- в сумме это означает, что текущее late-fusion / logit-fusion изменение
  можно считать удачным: оно дало сильный сигнал на GQA и хотя бы небольшой
  положительный сигнал на VQA-2.

## Негативные результаты Graph-Link и заметки по переписыванию

### VQA-2 — первая версия Graph-Link ухудшила результат

Наблюдённый статус:

- path family: `graph_link_vqa.yaml`, `graph_link_vqa_frozen.yaml`
- comparison reference: `baseline_vqa.yaml`
- matched practical baseline on VQA-2 remained stronger:
  observed `val_VQA_Accuracy = 0.54`
- early graph-link runs underperformed this baseline and, in one frozen-from-
  scratch setup, produced near-zero train accuracy

Что стало понятно:

- the first graph-link implementation was too aggressive as a pre-GNN
  intervention;
- it directly modified visual and kg node states before the baseline GNN stack,
  which made it easy to disturb a working baseline path;
- frozen runs launched from random initialization were methodologically weak,
  because a tiny trainable graph-link block was being optimized on top of an
  otherwise frozen random baseline;
- full graph-link runs showed that the module was learnable, but the resulting
  extra capacity did not translate into better validation performance.

Наиболее вероятные причины, зафиксированные для будущего отчёта:

1. The first graph-link variant started as a non-identity perturbation and
   could harm baseline node geometry from the first steps.
2. Cross-graph interaction happened too early and too strongly, before the
   existing backbone had a chance to stabilize around it.
3. The first design used top-k learned links without enough filtering or
   relevance calibration, so noisy cross-graph edges could survive.
4. The frozen graph-link experiment is only meaningful when initialized from a
   trained baseline checkpoint; frozen-from-scratch results should not be used
   as evidence against the hypothesis.

Какие follow-up действия были сделаны в репозитории:

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

Дисциплина интерпретации:

- these failed runs should be treated as valid negative experimental evidence
  about the *first* graph-link formulation, not as a final verdict on the
  broader hypothesis about controlled inter-graph integration.

## Нарратив успехов и неудач

Это сжатая экспериментальная история, которую стоит нести дальше в отчёт.

### Что явно помогло

1. Moving from ad hoc large-budget runs to a fixed subset protocol made
   comparisons much cleaner.
2. Treating GQA as the main benchmark and VQA-2 as auxiliary reduced
   evaluation ambiguity.
3. Preserving the baseline answer path and moving the new module into an
   auxiliary branch improved stability.
4. Replacing representation overwrite with:
   - sparse cross-graph linking
   - late auxiliary answer head
   - weighted logit fusion
   produced the first strong positive GQA signal (`0.30+` on the controlled
   subset family).
5. Та же late-fusion стратегия дала и положительный VQA-2 сигнал, хотя и
   заметно меньшего масштаба (`0.54 -> 0.55`).

### Что явно мешало

1. Early pre-GNN node rewriting was too destructive.
2. Frozen-from-scratch graph-link runs were methodologically invalid.
3. Unmatched protocols between baseline and module runs created confusing
   comparisons.
4. VQA-2 was much less forgiving as a rapid proving ground for the first
   graph-link formulation.

### Текущая лучшая practical-картина

- GQA controlled subset baseline: `0.20`
- GQA graph-link controlled subset: `0.30+`
- VQA-2 classic practical baseline: `0.54`
- VQA-2 current graph-link late-fusion run: `0.55`
- early VQA-2 graph-link formulations: negative / unstable relative to the
  baseline, but the current late-fusion/logit-fusion variant recovered a
  small positive gain

## Зафиксированные экспериментальные трудности

Следующие трудности стоит сохранить в протоколе для будущего отчёта.

### Методологические трудности

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

### Инженерные трудности

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

### Текущий урок по протоколу

For future architectural experiments, the safest reporting order is:

1. matched baseline control
2. frozen-baseline + new module
3. full baseline + new module

This ordering gives the clearest interpretation and catches setup problems
before expensive full-train comparison runs.
