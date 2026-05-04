# Дизайн Graph-Link Module

Статус: `implemented`, `runtime-valid`.

## 1. Мотивация

Текущие GQA и VQA-2 baselines уже содержат общий runtime graph:

- visual scene nodes
- question node
- textual/kg nodes

Но взаимодействие между visual и textual/kg ветками в baseline в основном
происходит через общий GNN stack и существующую adjacency structure.

Исследовательская гипотеза этого этапа: более явная и контролируемая
межграфовая связь между visual scene nodes и textual/kg nodes может дать
дополнительный полезный обмен сигналом без полного architectural redesign.

## 2. Что делает модуль

Новый модуль: `SparseGraphLinkModule` в
[`src/model/graph_link.py`](src/model/graph_link.py).

Он выполняет:

1. question-conditioned cosine scoring между visual nodes и textual/kg nodes;
2. sparse top-k selection межграфовых связей;
3. dynamic relevance filtering по схеме, вдохновлённой MAGIC-VQA:
   `mean ± std * scale`;
4. bidirectional sparse multi-head cross-attention:
   - scene attends to kg
   - kg attends to scene
5. pooled `link_repr`
6. отдельный graph-link answer head поверх
   `concat([baseline_repr, alpha * link_repr])`
7. late logit fusion:
   `final_logits = baseline_logits + alpha * graph_link_logits`

Модуль не переписывает baseline adjacency matrix и не меняет dataset contract.
Интеграция теперь поздняя и более безопасная: baseline graph path строит
`baseline_repr` и `baseline_logits` как раньше, а graph-link branch добавляет
отдельный auxiliary answer signal только на этапе финального logit fusion.

## 3. Где он встроен

Точка встройки для обоих active paths:

1. visual features проецируются в hidden space;
2. question кодируется в hidden space;
3. textual/kg node features проецируются в hidden space;
4. baseline собирает `visual | question | kg` и запускает обычный
   relation-aware `DenseGATLayer` stack;
5. отдельно graph-link branch строит sparse cross-graph attention и
   производит `link_repr`;
6. baseline classifier даёт `baseline_logits`;
7. graph-link classifier даёт `graph_link_logits`;
8. итоговый ответ получается через
   `baseline_logits + alpha * graph_link_logits`.

То есть модуль больше не стоит как pre-GNN overwrite. Это теперь
**late-fusion auxiliary branch**.

## 4. Поддерживаемые режимы

### GQA baseline

- config: `baseline_gqa`
- `model.enable_graph_link_module = False`
- baseline behavior сохраняется

### GQA с включённым graph-link

- config: `graph_link_gqa`
- `model.enable_graph_link_module = True`
- baseline backbone остаётся trainable

### GQA: frozen baseline + graph-link

- config: `graph_link_gqa_frozen`
- `model.enable_graph_link_module = True`
- `freeze_policy.freeze_all_baseline = True`
- trainable остаётся в основном graph-link module

### GQA: frozen warm-start baseline + graph-link

- config: `graph_link_gqa_frozen_warm`
- `model.enable_graph_link_module = True`
- `freeze_policy.freeze_all_baseline = True`
- `trainer.from_pretrained` должен указывать на обученный baseline checkpoint
- `trainer.pretrained_strict = False`, чтобы baseline checkpoint можно было
  безопасно загрузить в graph-link model variant
- это основной каузальный режим для проверки, даёт ли модуль прирост как
  plug-in graft поверх фиксированного baseline

### VQA-2 baseline

- config: `baseline_vqa`
- `model.enable_graph_link_module = False`

### VQA-2 с включённым graph-link

- config: `graph_link_vqa`
- `model.enable_graph_link_module = True`

### VQA-2: frozen baseline + graph-link

- config: `graph_link_vqa_frozen`
- `model.enable_graph_link_module = True`
- `freeze_policy.freeze_all_baseline = True`

## 5. Сравнение baseline vs module-enabled

Контролируемое сравнение задаётся так:

- одинаковый dataset path
- одинаковый trainer path
- одинаковый loss / metrics / collate
- одинаковый baseline GNN stack
- меняется только:
  - включение graph-link module
  - при необходимости freeze policy

Это делает сравнение:

- `baseline`
- `baseline + graph links`
- `frozen baseline + graph links`

явным и воспроизводимым.

## 6. Freeze-режим

Freeze policy реализован в
[`src/utils/model_freeze.py`](src/utils/model_freeze.py)
и применяется из [`train.py`](train.py) после instantiation модели и до
создания optimizer param groups.

Поддерживаются флаги:

- `freeze_all_baseline`
- `freeze_visual_proj`
- `freeze_question_encoder`
- `freeze_kg_node_proj`
- `freeze_node_type_emb`
- `freeze_gnn_layers`
- `freeze_readout`
- `freeze_classifier`
- `freeze_graph_link_module`

Основной controlled-experiment режим:

- `freeze_all_baseline = True`
- `freeze_graph_link_module = False`

Но для методологически корректного эксперимента этого недостаточно: frozen
graph-link runs должны стартовать не с random initialization baseline, а с
обученного baseline checkpoint. Поэтому приоритетный режим теперь:

- `graph_link_gqa_frozen_warm`
- trained baseline checkpoint
- frozen baseline backbone
- trainable graph-link branch only

## 7. Оставшиеся ограничения и риски

- Текущий GQA path не содержит отдельный внешний ConceptNet graph:
  модуль связывает visual nodes с текущими textual/kg nodes из active runtime
  contract, а не с новым preprocessing artifact.
- top-k sparsification вопрос-обусловлена, но пока не использует отдельный
  threshold schedule или auxiliary sparsity loss.
- baseline adjacency matrix не расширяется новыми typed edges; cross-graph
  branch работает как auxiliary late-fusion path, а не как полная graph
  rewiring.
- На controlled GQA subset уже наблюдался сильный положительный сигнал
  (`0.20 -> 0.30+`), но broader validation beyond this protocol is still
  `not validated yet`.
- На VQA-2 текущая late-fusion / logit-fusion версия тоже дала положительный,
  но существенно более слабый сигнал (`0.54 -> 0.55`), что делает GQA
  главным носителем исследовательского эффекта на данном этапе.

## 8. Зафиксированные трудности

- Первая graph-link реализация оказалась слишком агрессивной как pre-GNN
  intervention и регрессировала на VQA-2 относительно baseline.
- Frozen graph-link runs без pretrained baseline checkpoint методологически
  слабы и могут коллапсировать почти в нулевую accuracy.
- Matching experimental protocol оказался критичным: small differences in
  subset/seed/batch/clipping made comparisons noisy.
- Более conservative rewrites улучшают stability, но могут уменьшать эффект
  модуля. Это остаётся активной исследовательской дилеммой.
