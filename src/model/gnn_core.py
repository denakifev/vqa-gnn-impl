"""
Shared GNN primitives for VQA-GNN (arXiv:2205.11501).

This module contains the graph and text encoding building blocks used by
the GQA stack:

  DenseGATLayer       Dense multi-head Graph Attention Layer.
  HFQuestionEncoder   HuggingFace-backed question/context encoder.

GQAVQAGNNModel imports from here. Nothing in this module is task-specific.

Deviation from paper:
  - DenseGATLayer uses dense adjacency matrices (no torch_geometric).
    The paper does not fully specify the GNN variant; dense GAT is an
    engineering approximation. Graph sizes here are small (N_total ≲ 200),
    so the dense path is acceptable memory-wise, but numerical equivalence
    with the paper's GNN has not been verified.
  - Relation awareness: DenseGATLayer accepts an optional
    `num_relations` > 0 constructor flag. When enabled and an
    `edge_types` tensor is passed to forward(), the layer adds a learned
    per-(head, relation) scalar bias to the pre-softmax attention logits.
    This is the minimal paper-aligned injection of typed scene-graph
    relations into message passing. When `num_relations=0` the layer is
    bit-for-bit identical to the untyped implementation.
  - HFQuestionEncoder defaults to 'roberta-large' as the closest public
    checkpoint to what the paper describes.
"""

import logging

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

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_relations: int = 0,
    ):
        """
        Args:
            d_model (int): input and output feature dimension.
            num_heads (int): number of attention heads.
            dropout (float): dropout probability.
            num_relations (int): size of the typed-edge vocabulary. If 0,
                the layer ignores edge types entirely and behaves exactly
                like the baseline dense GAT. If > 0, a
                learned per-(head, relation) scalar bias is added to the
                attention logits whenever an `edge_types` tensor is passed
                to forward(). The embedding is zero-initialized so relation
                awareness starts as a no-op and gets learned during training.
        """
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.num_relations = int(num_relations)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if self.num_relations > 0:
            self.relation_bias = nn.Embedding(self.num_relations, num_heads)
            nn.init.zeros_(self.relation_bias.weight)
        else:
            self.relation_bias = None

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): node features [B, N, d_model].
            adj (Tensor): adjacency matrix [B, N, N] (0/1 or weighted).
                          A value of 1 means an edge exists. Self-loops
                          should be included by the caller.
            edge_types (Tensor | None): optional typed-edge ids
                [B, N, N] (long). Required when the layer was built with
                `num_relations > 0` and the caller wants relation-aware
                message passing. Ignored when `num_relations == 0`.
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

        # Relation-aware additive bias.
        if self.relation_bias is not None and edge_types is not None:
            if int(edge_types.max().item()) >= self.num_relations:
                raise ValueError(
                    f"edge_types contains id {int(edge_types.max().item())} "
                    f">= num_relations={self.num_relations}. Increase "
                    f"num_relations in the model config or rebuild graphs "
                    f"with a smaller relation vocabulary."
                )
            # [B, N, N, H] → [B, H, N, N]
            rel_bias = self.relation_bias(edge_types).permute(0, 3, 1, 2)
            attn = attn + rel_bias

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

    Used by the GQA model to encode the question.
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
            cls_repr (Tensor): [CLS] representation [B, d_hidden].
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls_repr)
