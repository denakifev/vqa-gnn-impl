# Graph Link Module Design

Статус: `implemented`, `runtime-valid`.

## 1. Motivation

Текущие GQA и VQA-2 baselines уже содержат общий runtime graph:

- visual scene nodes
- question node
- textual/kg nodes

Но взаимодействие между visual и textual/kg ветками в baseline в основном
происходит через общий GNN stack и существующую adjacency structure.

Исследовательская гипотеза этого этапа: более явная и контролируемая
межграфовая связь между visual scene nodes и textual/kg nodes может дать
дополнительный полезный обмен сигналом без полного architectural redesign.

## 2. What The Module Does

Новый модуль: `SparseGraphLinkModule` в
[`src/model/graph_link.py`](src/model/graph_link.py).

Он выполняет:

1. question-conditioned scoring между visual nodes и textual/kg nodes;
2. sparse top-k selection межграфовых связей;
3. weighted message aggregation:
   - visual <- top-k kg
   - kg <- top-k visual
4. gated residual fusion назад в baseline node states.

Модуль не переписывает baseline adjacency matrix и не меняет dataset contract.

## 3. Where It Is Inserted

Точка встройки для обоих active paths:

1. visual features проецируются в hidden space;
2. question кодируется в hidden space;
3. textual/kg node features проецируются в hidden space;
4. **graph-link module** связывает visual и kg node states;
5. затем baseline собирает `visual | question | kg` и запускает обычный
   relation-aware `DenseGATLayer` stack.

То есть модуль стоит **до baseline GNN stack** как additive pre-GNN linker.

## 4. Supported Modes

### GQA Baseline

- config: `baseline_gqa`
- `model.enable_graph_link_module = False`
- baseline behavior сохраняется

### GQA Graph-link enabled

- config: `graph_link_gqa`
- `model.enable_graph_link_module = True`
- baseline backbone остаётся trainable

### GQA Frozen baseline + graph-link

- config: `graph_link_gqa_frozen`
- `model.enable_graph_link_module = True`
- `freeze_policy.freeze_all_baseline = True`
- trainable остаётся в основном graph-link module

### VQA-2 Baseline

- config: `baseline_vqa`
- `model.enable_graph_link_module = False`

### VQA-2 Graph-link enabled

- config: `graph_link_vqa`
- `model.enable_graph_link_module = True`

### VQA-2 Frozen baseline + graph-link

- config: `graph_link_vqa_frozen`
- `model.enable_graph_link_module = True`
- `freeze_policy.freeze_all_baseline = True`

## 5. Baseline vs Module-Enabled Comparison

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

## 6. Freeze Regime

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

## 7. Remaining Limits / Risks

- Текущий GQA path не содержит отдельный внешний ConceptNet graph:
  модуль связывает visual nodes с текущими textual/kg nodes из active runtime
  contract, а не с новым preprocessing artifact.
- top-k sparsification вопрос-обусловлена, но пока не использует отдельный
  threshold schedule или auxiliary sparsity loss.
- baseline adjacency matrix не расширяется новыми typed edges; cross-graph
  linking сейчас реализован как pre-GNN residual exchange, а не как полная
  graph rewiring.
- Реальный gain по quality пока `not validated yet`; эта сессия готовит
  чистую experimental surface, а не claim о metric improvement.
