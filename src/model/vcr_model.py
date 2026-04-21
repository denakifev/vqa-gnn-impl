"""
VCR-specific VQA-GNN model (arXiv:2205.11501).

VCR (Visual Commonsense Reasoning) is a 4-way multiple-choice task.
The model does NOT produce a vocabulary distribution — it produces
a single scalar score per candidate item.

Candidate scoring contract:
  - Input batch is expanded B → B×4 by vcr_collate_fn before the
    model forward pass.
  - Each expanded item encodes one [image, QA-context, KG] triple
    independently.
  - The model outputs logits [B×4, 1]: one score per candidate.
  - VCRLoss reshapes [B×4, 1] → [B, 4] and applies cross-entropy.
  - VCRQAAccuracy / VCRQARAccuracy do the same reshape for prediction.

This is architecturally different from GQAVQAGNNModel, which produces
a vocabulary distribution [B, 1842] for direct classification.

Deviation from paper:
  - Same GNN approximations as described in gnn_core.py.
  - KG subgraphs are pre-built offline (paper: on-the-fly ConceptNet).
  - Entity extraction via tokenization (paper: likely NER/entity linking).
  - VCR object references resolved to category name strings (not region
    grounding).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.gnn_core import DenseGATLayer, HFQuestionEncoder

logger = logging.getLogger(__name__)


class VCRVQAGNNModel(nn.Module):
    """
    VCR-specific VQA-GNN model from arXiv:2205.11501.

    Architecture:
      1. Project visual region features to d_hidden.
      2. Encode [question; candidate] context via HFQuestionEncoder → d_hidden.
      3. Project KG node features to d_hidden.
      4. Add node-type embeddings (visual / question / KG).
      5. Concatenate all nodes into a single graph node matrix.
      6. Run L layers of DenseGATLayer on the full graph.
      7. Attention-pool the graph nodes → graph representation.
      8. Score the candidate: Linear(d_hidden → 1).

    VCR task contract:
      - Input batch: expanded B→B×4 by vcr_collate_fn.
        Each item encodes [question; one answer/rationale candidate].
      - Output: logits [B×4, 1] — scalar score per candidate.
      - VCRLoss: reshape [B×4, 1] → [B, 4], cross-entropy with answer_label.
      - The output dimension is always 1 (not configurable). Using a
        vocabulary-sized output for VCR would conflate candidate scoring
        with answer classification — they are different output regimes.

    Batch keys consumed:
        visual_features (Tensor[B*4, N_v, d_visual]): region features.
        question_input_ids (Tensor[B*4, L]): [question; candidate] tokens.
        question_attention_mask (Tensor[B*4, L]): text encoder mask.
        graph_node_features (Tensor[B*4, N_kg, d_kg]): KG node features.
        graph_adj (Tensor[B*4, N_total, N_total]): adjacency matrix.
        graph_node_types (Tensor[B*4, N_total]): 0=visual, 1=question, 2=kg.

    Batch keys produced:
        logits (Tensor[B*4, 1]): candidate scores.
    """

    # Node type indices — must match VCRDataset
    NODE_TYPE_VISUAL = 0
    NODE_TYPE_QUESTION = 1
    NODE_TYPE_KG = 2
    NUM_NODE_TYPES = 3

    # VCR always scores one value per candidate (binary scorer, not vocab dist)
    _N_SCORES_PER_CANDIDATE = 1

    def __init__(
        self,
        d_visual: int = 2048,
        d_kg: int = 300,
        d_hidden: int = 1024,
        num_gnn_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        text_encoder_name: str = "roberta-large",
        freeze_text_encoder: bool = False,
    ):
        """
        Args:
            d_visual (int): visual feature dimension (2048 for R2C/bottom-up).
            d_kg (int): KG node feature dimension (300 for ConceptNet Numberbatch).
            d_hidden (int): hidden dim throughout model (1024 for roberta-large).
            num_gnn_layers (int): number of dense GAT layers (paper: 3).
            num_heads (int): attention heads per GAT layer.
            dropout (float): dropout probability.
            text_encoder_name (str): HF identifier for text encoder.
            freeze_text_encoder (bool): if True, freeze text encoder weights.

        Note on VCR output dimension:
            The classifier produces a scalar score per candidate item
            (self._N_SCORES_PER_CANDIDATE = 1). This is NOT configurable
            because VCR candidate scoring and GQA vocabulary classification
            are different output regimes — not the same architecture with
            a different size parameter.
        """
        super().__init__()

        # Visual projection
        self.visual_proj = nn.Sequential(
            nn.Linear(d_visual, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
        )

        # Question / QA-context encoder (HF backbone + projection)
        self.question_encoder = HFQuestionEncoder(
            d_hidden=d_hidden,
            model_name=text_encoder_name,
            freeze=freeze_text_encoder,
            dropout=dropout,
        )

        # KG node projection
        self.kg_node_proj = nn.Sequential(
            nn.Linear(d_kg, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
        )

        # Node type bias embeddings
        self.node_type_emb = nn.Embedding(self.NUM_NODE_TYPES, d_hidden)

        # Graph attention layers
        self.gnn_layers = nn.ModuleList(
            [DenseGATLayer(d_hidden, num_heads, dropout) for _ in range(num_gnn_layers)]
        )

        # Attention-pooling readout
        self.readout_attn = nn.Linear(d_hidden, 1)

        # VCR candidate scorer: scalar score per candidate
        self.candidate_scorer = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, self._N_SCORES_PER_CANDIDATE),
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
        **batch,
    ) -> dict:
        """
        Args:
            visual_features (Tensor): region features [B*4, N_v, d_visual].
            question_input_ids (Tensor): [question; candidate] tokens [B*4, L].
            question_attention_mask (Tensor): text encoder mask [B*4, L].
            graph_node_features (Tensor): KG node features [B*4, N_kg, d_kg].
            graph_adj (Tensor): adjacency matrix [B*4, N_total, N_total].
                                N_total = N_v + 1 + N_kg.
            graph_node_types (Tensor): node type ids [B*4, N_total].

        Returns:
            dict with key 'logits' (Tensor[B*4, 1]): candidate scores.
            VCRLoss will reshape to [B, 4] internally.
        """
        # 1. Project visual features → [B*4, N_v, d_hidden]
        vis_emb = self.visual_proj(visual_features)

        # 2. Encode [question; candidate] → [B*4, d_hidden] → [B*4, 1, d_hidden]
        q_emb = self.question_encoder(question_input_ids, question_attention_mask)
        q_emb = q_emb.unsqueeze(1)

        # 3. Project KG node features → [B*4, N_kg, d_hidden]
        kg_emb = self.kg_node_proj(graph_node_features)

        # 4. Build full node feature matrix [B*4, N_total, d_hidden]
        #    Order: visual | question/candidate | kg  (matches graph_node_types)
        x = torch.cat([vis_emb, q_emb, kg_emb], dim=1)

        # 5. Add node-type embeddings
        type_emb = self.node_type_emb(graph_node_types)
        x = x + type_emb

        # 6. GNN message passing
        for layer in self.gnn_layers:
            x = layer(x, graph_adj)

        # 7. Attention-weighted readout → [B*4, d_hidden]
        attn_scores = self.readout_attn(x).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        graph_repr = (x * attn_weights).sum(dim=1)

        # 8. Candidate scoring → [B*4, 1]
        logits = self.candidate_scorer(self.dropout(graph_repr))

        return {"logits": logits}

    def __str__(self) -> str:
        all_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = super().__str__()
        return (
            f"{base}\n"
            f"Task: VCR candidate scoring (output dim: {self._N_SCORES_PER_CANDIDATE})\n"
            f"All parameters: {all_params:,}\n"
            f"Trainable: {trainable:,}"
        )
