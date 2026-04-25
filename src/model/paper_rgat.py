"""
Paper-equation relation-aware GAT layer for VQA-GNN (arXiv:2205.11501, §4.2).

This module implements the equations printed in the paper verbatim. It is kept
separate from `src.model.gnn_core.DenseGATLayer` deliberately: the legacy dense
GAT is an engineering variant that corresponds to the paper's Ablation 2 / 1
(single GNN, no explicit multi-relation message concatenation), whereas
`PaperMultiRelationGATLayer` matches §4.2 Eqs 1-6 and is the paper-equation
RGAT layer in this repository (see `DATA_CONTRACTS.md` and
`GQA_CLOSURE_REPORT.md` for current status).

Paper equations (verbatim symbol mapping):

    (1) r_ij = f_r([e_ij || u_i || u_j])
    (2) m_ij = f_m([h_i^{k+1} || r_ij])
    (3) h_i^{k+1} = f_h( sum_{j in N_i} alpha_ij * m_ij ) + h_i^{k}
    (4) q_i = f_q(h_i^{k+1}),    k_j = f_k([h_j^{k+1} || r_ij])
    (5) gamma_ij = q_i^T k_j / sqrt(D)
    (6) alpha_ij = softmax_j(gamma_ij)

Known paper underspecifications (tracked in `GAP_ANALYSIS.md`):

    * number of attention heads (paper Eq 5 is written with scalar gamma_ij, so
      this layer uses **single-head** attention as the most literal reading);
    * widths of intermediate hidden layers inside f_r and f_h;
    * activations inside f_r and f_h (this implementation uses ReLU, the most
      common GNN choice when unspecified);
    * dropout (defaults to 0.0 so the layer never silently regularises beyond
      what the caller configures).

All structural requirements the paper *does* state are tested by
`tests/test_paper_rgat.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperMultiRelationGATLayer(nn.Module):
    """
    Paper-aligned relation-aware GAT layer (VQA-GNN §4.2, Eqs 1-6).

    Dense-adjacency implementation. Operates on batches of small graphs
    (N <= 200) typical of VQA-GNN's multimodal semantic graph.

    Args:
        d_model (int): node feature / hidden dim (D in the paper).
        num_relations (int): size of the typed-edge vocabulary |R|. The
            relation embedding in Eq 1 uses a one-hot over |R|; we store it
            as an `nn.Embedding(num_relations, num_relations)` with identity
            weights (frozen) so that the downstream f_r MLP sees exactly the
            paper's one-hot input. Reserve id 0 for "no edge"; callers set
            non-edge positions to 0 and mask them via `adj`.
        num_node_types (int): |T| — number of node types. Paper states
            {Z, P, S, C} plus question node q; |T| defaults to 5.
        hidden_mult (int): internal width multiplier inside the 2-layer MLPs
            (f_r and f_h). Defaults to 1 so MLP hidden = d_model.
        dropout (float): dropout applied to attention weights. Default 0.0
            because the paper does not state a dropout rate.

    Inputs to forward():
        x (Tensor[B, N, d_model]): node features h^{k}.
        adj (Tensor[B, N, N]): 0/1 adjacency (symmetric, self-loops allowed).
            A value of 0 masks the (i, j) pair; non-zero means an edge exists.
        edge_types (Tensor[B, N, N] long): per-(i, j) relation id in
            [0, num_relations). Positions where `adj == 0` are ignored.
        node_types_src (Tensor[B, N] long): per-node type id u_i.
        node_types_dst (Tensor[B, N] long | None): per-node type id u_j. If
            None, `node_types_src` is used for both (the graph is homogeneous
            in node types under Eq 1 and paper's setup: u_i / u_j come from
            the same type vocabulary T).

    Returns:
        h_next (Tensor[B, N, d_model]): node features h^{k+1}.
    """

    def __init__(
        self,
        d_model: int,
        num_relations: int,
        num_node_types: int = 5,
        hidden_mult: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_relations <= 0:
            raise ValueError(
                f"num_relations must be >= 1 (paper Eq 1 requires a non-empty "
                f"relation vocabulary |R|). Got {num_relations}."
            )
        if num_node_types <= 0:
            raise ValueError(
                f"num_node_types must be >= 1 (paper Eq 1 uses one-hot u_i). "
                f"Got {num_node_types}."
            )

        self.d_model = d_model
        self.num_relations = num_relations
        self.num_node_types = num_node_types

        # One-hot encoders (Eq 1 inputs). Kept as frozen identity embeddings
        # so gradient signal flows only through f_r. This is a literal reading
        # of Eq 1 where e_ij and u_i are one-hot vectors.
        self.register_buffer(
            "_edge_onehot",
            torch.eye(num_relations, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_node_onehot",
            torch.eye(num_node_types, dtype=torch.float32),
            persistent=False,
        )

        # f_r: R^{|R| + 2|T|} -> R^D, 2-layer MLP (paper Eq 1 statement).
        fr_hidden = d_model * hidden_mult
        self.f_r = nn.Sequential(
            nn.Linear(num_relations + 2 * num_node_types, fr_hidden),
            nn.ReLU(),
            nn.Linear(fr_hidden, d_model),
        )

        # f_m: R^{2D} -> R^D, linear (paper Eq 2 states "linear transformation").
        self.f_m = nn.Linear(2 * d_model, d_model)

        # f_q: R^D -> R^D, linear (paper Eq 4).
        self.f_q = nn.Linear(d_model, d_model)
        # f_k: R^{2D} -> R^D, linear (paper Eq 4).
        self.f_k = nn.Linear(2 * d_model, d_model)

        # f_h: R^D -> R^D, 2-layer MLP with batch normalisation (paper Eq 3).
        fh_hidden = d_model * hidden_mult
        self.f_h = nn.Sequential(
            nn.Linear(d_model, fh_hidden),
            nn.BatchNorm1d(fh_hidden),
            nn.ReLU(),
            nn.Linear(fh_hidden, d_model),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self._inv_sqrt_d = 1.0 / math.sqrt(d_model)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_types: torch.Tensor,
        node_types_src: torch.Tensor,
        node_types_dst: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        if D != self.d_model:
            raise ValueError(
                f"x has d_model={D} but layer was built with d_model={self.d_model}"
            )
        if adj.shape != (B, N, N):
            raise ValueError(f"adj shape {tuple(adj.shape)} != (B, N, N) = ({B}, {N}, {N})")
        if edge_types.shape != (B, N, N):
            raise ValueError(
                f"edge_types shape {tuple(edge_types.shape)} != (B, N, N)"
            )
        if edge_types.dtype != torch.long:
            raise ValueError(
                f"edge_types must be long (relation ids), got dtype={edge_types.dtype}"
            )
        if int(edge_types.max().item()) >= self.num_relations:
            raise ValueError(
                f"edge_types contains id {int(edge_types.max().item())} "
                f">= num_relations={self.num_relations}"
            )
        if node_types_dst is None:
            node_types_dst = node_types_src

        # ---- Eq 1: r_ij = f_r([e_ij || u_i || u_j]) ---------------------------------
        # Build one-hot encodings then stack into [B, N, N, |R| + 2|T|].
        e_oh = self._edge_onehot[edge_types]  # [B, N, N, |R|]
        u_src_oh = self._node_onehot[node_types_src]  # [B, N, |T|]
        u_dst_oh = self._node_onehot[node_types_dst]  # [B, N, |T|]
        # Broadcast source (per-row) and destination (per-col) to [B, N, N, |T|].
        u_i = u_src_oh.unsqueeze(2).expand(B, N, N, self.num_node_types)
        u_j = u_dst_oh.unsqueeze(1).expand(B, N, N, self.num_node_types)
        relation_input = torch.cat([e_oh, u_i, u_j], dim=-1)
        r_ij = self.f_r(relation_input)  # [B, N, N, D]

        # ---- Eq 2: m_ij = f_m([h_i || r_ij]) -----------------------------------------
        # In paper's notation h_i is the feature of the *source* node i. For each
        # pair (i, j) we concatenate along feature dim: [h_i || r_ij].
        h_src = x.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D], repeated along j
        message_input = torch.cat([h_src, r_ij], dim=-1)  # [B, N, N, 2D]
        m_ij = self.f_m(message_input)  # [B, N, N, D]

        # ---- Eq 4: q_i = f_q(h_i^{k+1}),  k_j = f_k([h_j || r_ij]) -------------------
        # Following paper, h^{k+1} inside the attention step means "feature at this
        # layer's input". We use x for both q and k (pre-update), which is the
        # standard self-attention convention.
        q = self.f_q(x)  # [B, N, D]
        h_dst = x.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D], repeated along i
        k_input = torch.cat([h_dst, r_ij], dim=-1)
        k = self.f_k(k_input)  # [B, N, N, D]

        # ---- Eq 5: gamma_ij = q_i^T k_j / sqrt(D) ------------------------------------
        # q is per-row (source), k is per-(row, col). Inner product over D.
        # q: [B, N, 1, D];  k: [B, N, N, D] -> gamma: [B, N, N]
        gamma = (q.unsqueeze(2) * k).sum(dim=-1) * self._inv_sqrt_d

        # ---- Eq 6: alpha_ij = softmax_j(gamma_ij), masked by adj ---------------------
        # The paper sums over N_i: j in neighbourhood of i. With dense adjacency we
        # mask non-edges with -inf before softmax, then zero-out NaN rows caused by
        # isolated nodes.
        mask = adj == 0
        gamma = gamma.masked_fill(mask, float("-inf"))
        alpha = F.softmax(gamma, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.attn_dropout(alpha)

        # ---- Aggregation: sum_{j in N_i} alpha_ij * m_ij -----------------------------
        # alpha: [B, N, N], m_ij: [B, N, N, D] -> [B, N, D]
        # In paper's indexing (i is target, j is source neighbour); here source is
        # already encoded in h_src via the i-axis. So sum over the j-axis (dim=2).
        agg = (alpha.unsqueeze(-1) * m_ij).sum(dim=2)  # [B, N, D]

        # ---- Eq 3: h^{k+1} = f_h(agg) + h^{k} ----------------------------------------
        # BatchNorm1d needs [B*N, D].
        flat = agg.reshape(B * N, D)
        flat = self.f_h[0](flat)  # Linear
        flat = self.f_h[1](flat)  # BatchNorm1d
        flat = self.f_h[2](flat)  # ReLU
        flat = self.f_h[3](flat)  # Linear
        update = flat.reshape(B, N, D)

        return update + x
