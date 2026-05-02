"""
GQA-specific VQA-GNN model (arXiv:2205.11501).

GQA (Generalized Visual Question Answering) is an open-ended VQA task with
a fixed answer vocabulary. The model produces a probability distribution
over 1842 answer classes (paper target for the balanced split).

This is an open-ended classification model over the fixed GQA answer
vocabulary.

Key GQA-specific characteristics:
  - Direct classification: output [B, num_answers].
  - Answer vocabulary: 1842 classes (paper-aligned balanced split).
  - Scene graph: GQA batch includes graph_edge_types (relation type matrix),
    representing typed edges from the Visual Genome scene graph.

GQA scene-graph relation awareness:
  - The batch includes graph_edge_types (Tensor[B, N_total, N_total], long)
    produced by the paper-like preprocessing pipeline
    (src/datasets/gqa_preprocessing.py).
  - GQAVQAGNNModel wires graph_edge_types into each dense GAT layer via
    num_relations > 0, which adds a learned per-(head, relation) scalar
    bias to the pre-softmax attention logits. This is the minimal
    paper-aligned injection of typed scene-graph edges into message
    passing; it is not numerically equivalent to an arbitrary relational
    GNN variant, but it is the straightforward adaptation documented in
    the GAT literature.
  - num_relations in the model config must be >= the size of the runtime
    relation vocabulary (gqa_relation_vocab.json → relation_count_total).
    The dense GAT layer checks this at forward time and fails loudly on
    out-of-range edge ids, so a silent truncation is not possible.

General deviation from paper:
  - Same GNN approximations as described in gnn_core.py.
  - KG / scene-graph subgraphs are pre-built offline.
  - Entity extraction via tokenization (paper: likely NER/entity linking).
  - Answer vocabulary is inferred from the released balanced splits rather
    than from an official paper artifact.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.gnn_core import DenseGATLayer, HFQuestionEncoder

logger = logging.getLogger(__name__)


class GQAVQAGNNModel(nn.Module):
    """
    GQA-specific VQA-GNN model from arXiv:2205.11501.

    Architecture:
      1. Project visual region features to d_hidden.
      2. Encode question via HFQuestionEncoder → d_hidden.
      3. Project scene-graph node features to d_hidden.
      4. Add node-type embeddings (visual / question / scene-graph).
      5. Concatenate all nodes into a single graph node matrix.
      6. Run L layers of DenseGATLayer on the full graph.
      7. Attention-pool the graph nodes → graph representation.
      8. Classify over the answer vocabulary.

    GQA task contract:
      - Input batch: standard (B items, no candidate expansion).
      - Output: logits [B, num_answers].
      - GQALoss: hard cross-entropy with single GT answer index.
      - GQAAccuracy: argmax(logits) == labels.

    Relation-aware message passing:
      - graph_edge_types is consumed by every dense GAT layer when
        num_relations > 0 (the paper-aligned default).
      - Setting num_relations = 0 reverts to the earlier untyped variant
        (useful only for ablation; less paper-aligned).

    Batch keys consumed:
        visual_features (Tensor[B, N_v, d_visual]): region features.
        question_input_ids (Tensor[B, L]): tokenized question.
        question_attention_mask (Tensor[B, L]): text encoder mask.
        graph_node_features (Tensor[B, N_kg, d_kg]): scene-graph node feats.
        graph_adj (Tensor[B, N_total, N_total]): adjacency matrix.
        graph_node_types (Tensor[B, N_total]): 0=visual, 1=question, 2=kg.
        graph_edge_types (Tensor[B, N_total, N_total]): relation ids.

    Batch keys produced:
        logits (Tensor[B, num_answers]): unnormalized answer logits.
    """

    # Node type indices — must match GQADataset
    NODE_TYPE_VISUAL = 0
    NODE_TYPE_QUESTION = 1
    NODE_TYPE_KG = 2
    NUM_NODE_TYPES = 3

    def __init__(
        self,
        d_visual: int = 2048,
        d_kg: int = 600,
        d_hidden: int = 1024,
        num_gnn_layers: int = 3,
        num_heads: int = 8,
        num_answers: int = 1842,
        num_relations: int = 512,
        dropout: float = 0.1,
        text_encoder_name: str = "roberta-large",
        freeze_text_encoder: bool = False,
    ):
        """
        Args:
            d_visual (int): visual feature dimension (2048 for bottom-up/GQA).
            d_kg (int): scene-graph node feature dimension (600 for GQA SG).
                        Must match d_kg in the dataset config and preprocessing.
            d_hidden (int): hidden dim throughout model (1024 for roberta-large).
            num_gnn_layers (int): number of dense GAT layers (paper: 3).
            num_heads (int): attention heads per GAT layer.
            num_answers (int): answer vocabulary size.
                        Paper target for GQA balanced split: 1842.
                        Build vocabulary with: scripts/prepare_gqa_data.py answer-vocab
            num_relations (int): size of the typed-edge vocabulary accepted by
                        the relation-aware dense GAT layers. Must be >=
                        runtime relation vocabulary produced by
                        scripts/prepare_gqa_data.py relation-vocab
                        (see gqa_relation_vocab.json → relation_count_total).
                        Setting 0 disables relation awareness (ablation only).
            dropout (float): dropout probability.
            text_encoder_name (str): HF identifier for text encoder.
            freeze_text_encoder (bool): if True, freeze text encoder weights.
        """
        super().__init__()

        self.num_answers = num_answers
        self.num_relations = int(num_relations)

        # Visual projection
        self.visual_proj = nn.Sequential(
            nn.Linear(d_visual, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
        )

        # Question encoder (HF backbone + projection)
        self.question_encoder = HFQuestionEncoder(
            d_hidden=d_hidden,
            model_name=text_encoder_name,
            freeze=freeze_text_encoder,
            dropout=dropout,
        )

        # Scene-graph node projection
        self.kg_node_proj = nn.Sequential(
            nn.Linear(d_kg, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
        )

        # Node type bias embeddings
        self.node_type_emb = nn.Embedding(self.NUM_NODE_TYPES, d_hidden)

        # Graph attention layers (relation-aware when num_relations > 0)
        self.gnn_layers = nn.ModuleList(
            [
                DenseGATLayer(
                    d_hidden,
                    num_heads,
                    dropout,
                    num_relations=self.num_relations,
                )
                for _ in range(num_gnn_layers)
            ]
        )

        # Attention-pooling readout
        self.readout_attn = nn.Linear(d_hidden, 1)

        # Answer classifier: d_hidden → num_answers
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_answers),
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all non-pretrained linear layers."""
        encoder_param_ids = {id(p) for p in self.question_encoder.encoder.parameters()}
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if all(id(p) in encoder_param_ids for p in module.parameters()):
                continue
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        graph_node_features: torch.Tensor,
        graph_adj: torch.Tensor,
        graph_node_types: torch.Tensor,
        graph_edge_types: torch.Tensor | None = None,
        **batch,
    ) -> dict:
        """
        Args:
            visual_features (Tensor): region features [B, N_v, d_visual].
            question_input_ids (Tensor): tokenized question [B, L].
            question_attention_mask (Tensor): text encoder mask [B, L].
            graph_node_features (Tensor): scene-graph node feats [B, N_kg, d_kg].
            graph_adj (Tensor): adjacency matrix [B, N_total, N_total].
                                N_total = N_v + 1 + N_kg.
            graph_node_types (Tensor): node type ids [B, N_total].
            graph_edge_types (Tensor | None): typed-edge ids
                [B, N_total, N_total]. Consumed by the dense GAT layers
                when num_relations > 0 to produce relation-aware attention.
                Required when num_relations > 0.
            **batch: additional batch keys (ignored).

        Returns:
            dict with key 'logits' (Tensor[B, num_answers]).
        """
        if self.num_relations > 0 and graph_edge_types is None:
            raise ValueError(
                "GQAVQAGNNModel was built with num_relations > 0 "
                f"(num_relations={self.num_relations}) but the batch does not "
                "contain `graph_edge_types`. Add `graph_edge_types` to the "
                "trainer/inferencer `device_tensors` list and ensure the "
                "dataset emits typed edges "
                "(GQADataset.expect_graph_edge_types=True)."
            )

        # 1. Project visual features → [B, N_v, d_hidden]
        vis_emb = self.visual_proj(visual_features)

        # 2. Encode question → [B, d_hidden] → [B, 1, d_hidden]
        q_emb = self.question_encoder(question_input_ids, question_attention_mask)
        q_emb = q_emb.unsqueeze(1)

        # 3. Project scene-graph node features → [B, N_kg, d_hidden]
        kg_emb = self.kg_node_proj(graph_node_features)

        # 4. Build full node feature matrix [B, N_total, d_hidden]
        #    Order: visual | question | scene-graph  (matches graph_node_types)
        x = torch.cat([vis_emb, q_emb, kg_emb], dim=1)

        # 5. Add node-type embeddings
        type_emb = self.node_type_emb(graph_node_types)
        x = x + type_emb

        # 6. Relation-aware GNN message passing
        for layer in self.gnn_layers:
            x = layer(x, graph_adj, edge_types=graph_edge_types)

        # 7. Attention-weighted readout → [B, d_hidden]
        attn_scores = self.readout_attn(x).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        graph_repr = (x * attn_weights).sum(dim=1)

        # 8. Classify over answer vocabulary → [B, num_answers]
        logits = self.classifier(self.dropout(graph_repr))

        return {"logits": logits}

    def __str__(self) -> str:
        all_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = super().__str__()
        return (
            f"{base}\n"
            f"Task: GQA answer classification (vocab size: {self.num_answers})\n"
            f"All parameters: {all_params:,}\n"
            f"Trainable: {trainable:,}"
        )
