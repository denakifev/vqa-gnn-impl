"""
VQA-GNN: Reasoning with Multimodal Knowledge via Graph Neural Networks
for Visual Question Answering.

Reference: arXiv:2205.11501 / ICCV 2023

This module implements the baseline VQA-GNN model from the paper.

Deviation from paper:
- The paper builds the KG subgraph on-the-fly; here the graph is pre-built
  and passed as part of the batch (graph_node_features, graph_adj,
  graph_node_types).
- The paper uses pretrained RoBERTa for the QA-context node; this repo now
  defaults to `roberta-large` as the paper-aligned text encoder, while still
  supporting overrides for smaller backbones when resources are limited.
- The GNN is implemented with dense adjacency matrices (no torch_geometric)
  to minimize dependencies. This may be memory-intensive for very large graphs
  but is functionally equivalent for the graph sizes used in this paper
  (N_total = N_visual + 1 + N_kg ≈ 67 nodes).
- Readout uses learnable attention pooling over all graph nodes.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DenseGATLayer(nn.Module):
    """
    Dense multi-head Graph Attention Layer.

    Operates on a dense [B, N, N] adjacency matrix. Suitable for
    small-to-medium graphs in batched settings (N ≲ 200).

    Reference: Veličković et al., Graph Attention Networks, ICLR 2018.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model (int): input and output feature dimension.
            num_heads (int): number of attention heads.
            dropout (float): dropout probability.
        """
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): node features [B, N, d_model].
            adj (Tensor): adjacency matrix [B, N, N] (0/1 or weighted).
                          A value of 1 means an edge exists. Self-loops
                          should be included by the caller.
        Returns:
            out (Tensor): updated node features [B, N, d_model].
        """
        B, N, _ = x.shape
        residual = x
        x = self.norm1(x)

        # Multi-head projections: [B, N, d_model] → [B, H, N, head_dim]
        def reshape(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape(self.q_proj(x))
        k = reshape(self.k_proj(x))
        v = reshape(self.v_proj(x))

        # Attention scores [B, H, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Mask non-edges with -inf so they get zero weight after softmax
        adj_mask = adj.unsqueeze(1).expand_as(attn)  # [B, H, N, N]
        attn = attn.masked_fill(adj_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        # Replace NaN rows (isolated nodes, all-masked) with zero
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # Aggregate: [B, H, N, head_dim] → [B, N, d_model]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        out = self.dropout(out) + residual

        # Feed-forward with residual
        out = out + self.ffn(self.norm2(out))
        return out


class HFQuestionEncoder(nn.Module):
    """
    Question encoder backed by Hugging Face `AutoModel`.

    Outputs the [CLS] token representation projected to d_hidden.
    """

    def __init__(
        self,
        d_hidden: int,
        model_name: str = "roberta-large",
        freeze: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_hidden (int): output dimension after projection.
            model_name (str): HuggingFace model name.
            freeze (bool): if True, encoder weights are frozen.
            dropout (float): dropout after projection.

        For maximal paper alignment this repo defaults to `roberta-large`.
        The encoder class itself is selected automatically from the Hugging Face
        checkpoint via `AutoModel`.
        """
        super().__init__()
        try:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(model_name)
            self.encoder_hidden = self.encoder.config.hidden_size
        except ImportError as e:
            raise ImportError(
                "transformers is required for question encoding. "
                "Install it with: pip install transformers"
            ) from e

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Question encoder weights frozen.")

        self.proj = nn.Sequential(
            nn.Linear(self.encoder_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids (Tensor): token IDs [B, L].
            attention_mask (Tensor): attention mask [B, L].
        Returns:
            cls_repr (Tensor): question representation [B, d_hidden].
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls_repr)


class VQAGNNModel(nn.Module):
    """
    VQA-GNN model from arXiv:2205.11501.

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

        Deviations from paper:
            - Paper uses pretrained RoBERTa for QA-context encoding. This repo
              defaults to `roberta-large` as the closest standard checkpoint.
            - Paper uses torch_geometric / custom GNN; here dense GAT.
            - num_answers=3129 follows VQA v2 open-ended standard.
        """
        super().__init__()

        if bert_model_name is not None:
            text_encoder_name = bert_model_name
        if freeze_bert is not None:
            freeze_text_encoder = freeze_bert

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

        # KG node projection
        self.kg_node_proj = nn.Sequential(
            nn.Linear(d_kg, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
        )

        # Node type bias embeddings (learned, added to node features)
        self.node_type_emb = nn.Embedding(self.NUM_NODE_TYPES, d_hidden)

        # Graph attention layers
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
        # Collect parameter ids belonging to the pretrained encoder to avoid
        # reinitializing downloaded Hugging Face weights.
        # pretrained weights.
        encoder_param_ids = {id(p) for p in self.question_encoder.encoder.parameters()}

        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            # Skip any linear layer whose parameters are entirely inside the
            # pretrained text encoder.
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
        B = visual_features.size(0)

        # 1. Project visual features → [B, N_v, d_hidden]
        vis_emb = self.visual_proj(visual_features)

        # 2. Encode question → [B, d_hidden], then reshape to [B, 1, d_hidden]
        q_emb = self.question_encoder(question_input_ids, question_attention_mask)
        q_emb = q_emb.unsqueeze(1)  # [B, 1, d_hidden]

        # 3. Project KG node features → [B, N_kg, d_hidden]
        kg_emb = self.kg_node_proj(graph_node_features)

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
        logits = self.classifier(self.dropout(graph_repr))  # [B, num_answers]

        return {"logits": logits}

    def __str__(self) -> str:
        """Print model with parameter counts."""
        all_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = super().__str__()
        return f"{base}\nAll parameters: {all_params:,}\nTrainable: {trainable:,}"
