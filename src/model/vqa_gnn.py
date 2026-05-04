"""
VQA-GNN: Reasoning with Multimodal Knowledge via Graph Neural Networks
for Visual Question Answering.

Reference: arXiv:2205.11501 / ICCV 2023

This module keeps the original monolithic VQAGNNModel for the restored
VQA-2 coursework path and re-exports the shared GNN primitives that tests
import from this path.

Active configs use task-specific classes:
  - GQA: src.model.GQAVQAGNNModel (gqa_model.py)
  - VQA-2 coursework path: src.model.VQAGNNModel (this module)

Shared primitives (DenseGATLayer, HFQuestionEncoder) live in gnn_core.py
and are re-exported here for backward compatibility with existing imports
(e.g. `from src.model.vqa_gnn import DenseGATLayer`).

VQAGNNModel is not a GQA paper-result reproduction target. It is the VQA-2
coursework baseline class.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.graph_link import SparseGraphLinkModule
# Re-export shared primitives so existing imports like
#   from src.model.vqa_gnn import DenseGATLayer
# continue to work.
from src.model.gnn_core import DenseGATLayer, HFQuestionEncoder  # noqa: F401

logger = logging.getLogger(__name__)


class VQAGNNModel(nn.Module):
    """
    VQA-GNN model from arXiv:2205.11501.

    This is the original unified implementation used by the VQA-2 coursework
    baseline. GQA paper-aligned configs use src.model.GQAVQAGNNModel.

    Architecture:
      1. Project visual region features to d_hidden.
      2. Encode question via the text encoder's first token representation
         → project to d_hidden.
      3. Project KG node features to d_hidden.
      4. Add node-type embeddings (visual / question / KG).
      5. Concatenate all nodes into a single graph node matrix.
      6. Run L layers of DenseGATLayer on the full graph.
      7. Attention-pool the graph nodes → graph representation.
      8. Classify over the answer vocabulary.

    Batch keys consumed:
        visual_features (Tensor[B, N_v, d_visual]): region features.
        question_input_ids (Tensor[B, L]): tokenized question.
        question_attention_mask (Tensor[B, L]): text encoder attention mask.
        graph_node_features (Tensor[B, N_kg, d_kg]): KG node features.
        graph_adj (Tensor[B, N_total, N_total]): adjacency matrix.
        graph_node_types (Tensor[B, N_total]): 0=visual, 1=question, 2=kg.

    Batch keys produced:
        logits (Tensor[B, num_answers]): unnormalized answer logits.
    """

    # Node type indices — must match dataset
    NODE_TYPE_VISUAL = 0
    NODE_TYPE_QUESTION = 1
    NODE_TYPE_KG = 2
    NUM_NODE_TYPES = 3

    def __init__(
        self,
        d_visual: int = 2048,
        d_kg: int = 100,
        d_hidden: int = 1024,
        num_gnn_layers: int = 3,
        num_heads: int = 8,
        num_answers: int = 3129,
        dropout: float = 0.1,
        text_encoder_name: str = "roberta-large",
        freeze_text_encoder: bool = False,
        bert_model_name: Optional[str] = None,
        freeze_bert: Optional[bool] = None,
        enable_graph_link_module: bool = False,
        graph_link_top_k: int = 4,
        graph_link_num_heads: int = 8,
        graph_link_dropout: Optional[float] = None,
        graph_link_threshold_std_scale: float = 0.5,
    ):
        """
        Args:
            d_visual (int): visual feature dimension (e.g. 2048 for ResNet).
            d_kg (int): KG node feature dimension (e.g. 100 for ConceptNet).
            d_hidden (int): hidden dimension throughout the model.
            num_gnn_layers (int): number of graph attention layers.
            num_heads (int): attention heads per GNN layer.
            num_answers (int): size of the answer vocabulary.
            dropout (float): dropout probability.
            text_encoder_name (str): Hugging Face text encoder identifier.
            freeze_text_encoder (bool): whether to freeze text encoder weights.
            bert_model_name (str | None): deprecated alias for
                `text_encoder_name`, kept for backward compatibility.
            freeze_bert (bool | None): deprecated alias for
                `freeze_text_encoder`, kept for backward compatibility.
            enable_graph_link_module (bool): enable additive sparse
                visual<->kg linking as an auxiliary late-fusion branch.
            graph_link_top_k (int): sparse top-k links per node direction.
            graph_link_num_heads (int): number of heads in the sparse
                cross-attention branch.
            graph_link_dropout (float | None): dropout inside the graph-link
                module; defaults to `dropout`.
            graph_link_threshold_std_scale (float): relevance-threshold width
                used by the sparse graph-link branch.

        Deviations from paper:
            - Paper uses pretrained RoBERTa for QA-context encoding. This repo
              defaults to `roberta-large` as the closest standard checkpoint.
            - Paper uses torch_geometric / custom GNN; here dense GAT
              (engineering approximation; paper's GNN variant not fully specified).
            - Default num_answers=3129 targets the VQA-2 coursework path;
              GQA configs use GQAVQAGNNModel (num_answers=1842).
        """
        super().__init__()

        if bert_model_name is not None:
            text_encoder_name = bert_model_name
        if freeze_bert is not None:
            freeze_text_encoder = freeze_bert

        self.enable_graph_link_module = bool(enable_graph_link_module)

        # Visual projection
        self.visual_proj = nn.Sequential(
            nn.Linear(d_visual, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
        )

        # Question encoder (HF backbone + projection) — from gnn_core
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

        # Node type bias embeddings (learned, added to node features)
        self.node_type_emb = nn.Embedding(self.NUM_NODE_TYPES, d_hidden)

        if self.enable_graph_link_module:
            self.graph_link_module = SparseGraphLinkModule(
                d_hidden=d_hidden,
                top_k=graph_link_top_k,
                num_heads=graph_link_num_heads,
                dropout=dropout if graph_link_dropout is None else graph_link_dropout,
                threshold_std_scale=graph_link_threshold_std_scale,
            )
            self.graph_link_alpha = nn.Parameter(torch.tensor(0.0))
            self.graph_link_classifier = nn.Sequential(
                nn.Linear(d_hidden * 2, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, num_answers),
            )
        else:
            self.graph_link_module = None

        # Graph attention layers — DenseGATLayer from gnn_core
        self.gnn_layers = nn.ModuleList(
            [DenseGATLayer(d_hidden, num_heads, dropout) for _ in range(num_gnn_layers)]
        )

        # Readout: learnable attention pooling over all graph nodes
        self.readout_attn = nn.Linear(d_hidden, 1)

        # Answer classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_answers),
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for non-pretrained linear layers."""
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
            visual_features (Tensor): region features [B, N_v, d_visual].
            question_input_ids (Tensor): tokenized question [B, L].
            question_attention_mask (Tensor): text encoder mask [B, L].
            graph_node_features (Tensor): KG-only node features [B, N_kg, d_kg].
            graph_adj (Tensor): full adjacency matrix [B, N_total, N_total].
                                N_total = N_v + 1 + N_kg.
            graph_node_types (Tensor): node type ids [B, N_total].
        Returns:
            dict with key 'logits' (Tensor[B, num_answers]).
        """
        # 1. Project visual features → [B, N_v, d_hidden]
        vis_emb = self.visual_proj(visual_features)

        # 2. Encode question → [B, d_hidden], then reshape to [B, 1, d_hidden]
        q_emb = self.question_encoder(question_input_ids, question_attention_mask)
        q_emb = q_emb.unsqueeze(1)  # [B, 1, d_hidden]

        # 3. Project KG node features → [B, N_kg, d_hidden]
        kg_emb = self.kg_node_proj(graph_node_features)

        graph_link_stats = None
        graph_link_repr = None
        if self.graph_link_module is not None:
            visual_mask = visual_features.abs().sum(dim=-1) > 0
            kg_mask = graph_node_features.abs().sum(dim=-1) > 0
            graph_link_repr, graph_link_stats = self.graph_link_module(
                visual_nodes=vis_emb,
                kg_nodes=kg_emb,
                question_node=q_emb.squeeze(1),
                visual_mask=visual_mask,
                kg_mask=kg_mask,
            )

        # 4. Build full node feature matrix [B, N_total, d_hidden]
        #    Order: visual | question | kg (must match graph_node_types)
        x = torch.cat([vis_emb, q_emb, kg_emb], dim=1)

        # 5. Add node-type embeddings
        type_emb = self.node_type_emb(graph_node_types)  # [B, N_total, d_hidden]
        x = x + type_emb

        # 6. GNN message passing
        for layer in self.gnn_layers:
            x = layer(x, graph_adj)

        # 7. Attention-weighted graph readout → [B, d_hidden]
        attn_scores = self.readout_attn(x).squeeze(-1)  # [B, N_total]
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, N_total, 1]
        graph_repr = (x * attn_weights).sum(dim=1)  # [B, d_hidden]

        # 8. Classify
        baseline_logits = self.classifier(self.dropout(graph_repr))  # [B, num_answers]
        logits = baseline_logits
        outputs = {"logits": logits, "baseline_logits": baseline_logits}
        if graph_link_stats is not None:
            link_scale = torch.tanh(self.graph_link_alpha)
            graph_link_logits = self.graph_link_classifier(
                self.dropout(torch.cat([graph_repr, link_scale * graph_link_repr], dim=-1))
            )
            logits = baseline_logits + link_scale * graph_link_logits
            outputs["logits"] = logits
            outputs["graph_link_logits"] = graph_link_logits
            outputs["graph_link_stats"] = {
                **graph_link_stats,
                "link_alpha": float(link_scale.detach().cpu().item()),
            }
        return outputs

    def __str__(self) -> str:
        """Print model with parameter counts."""
        all_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = super().__str__()
        return f"{base}\nAll parameters: {all_params:,}\nTrainable: {trainable:,}"
