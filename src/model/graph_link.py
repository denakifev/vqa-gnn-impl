import math

import torch
import torch.nn as nn


class SparseGraphLinkModule(nn.Module):
    """
    Question-conditioned sparse cross-graph linker.

    The module connects visual scene nodes with textual/knowledge nodes without
    rewriting the baseline adjacency matrix. It performs a lightweight,
    top-k-limited cross-graph exchange before the baseline GNN stack:

      visual -> top-k knowledge -> gated residual update
      knowledge -> top-k visual -> gated residual update

    This keeps the baseline path intact while making the extra interaction
    explicit, sparse, and easy to freeze/unfreeze in controlled experiments.
    """

    def __init__(
        self,
        d_hidden: int,
        top_k: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self.top_k = int(top_k)
        self.scale = math.sqrt(float(d_hidden))

        self.visual_score_proj = nn.Linear(d_hidden, d_hidden)
        self.kg_score_proj = nn.Linear(d_hidden, d_hidden)
        self.question_to_visual = nn.Linear(d_hidden, d_hidden)
        self.question_to_kg = nn.Linear(d_hidden, d_hidden)

        self.kg_value_proj = nn.Linear(d_hidden, d_hidden)
        self.visual_value_proj = nn.Linear(d_hidden, d_hidden)

        self.visual_gate = nn.Linear(d_hidden * 3, d_hidden)
        self.kg_gate = nn.Linear(d_hidden * 3, d_hidden)

        self.visual_norm = nn.LayerNorm(d_hidden)
        self.kg_norm = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)

        self.last_stats = {}
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _batched_gather(nodes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Gather node states with per-batch indices.

        Args:
            nodes:   [B, N, D]
            indices: [B, Q, K]
        Returns:
            gathered: [B, Q, K, D]
        """
        bsz, num_nodes, dim = nodes.shape
        flat_nodes = nodes.reshape(bsz * num_nodes, dim)
        offsets = torch.arange(bsz, device=nodes.device).view(bsz, 1, 1) * num_nodes
        flat_indices = (indices + offsets).reshape(-1)
        gathered = flat_nodes.index_select(0, flat_indices)
        return gathered.view(bsz, indices.shape[1], indices.shape[2], dim)

    def _masked_topk(
        self,
        scores: torch.Tensor,
        target_mask: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a target-node mask and select top-k links along the last axis.
        """
        masked_scores = scores.masked_fill(~target_mask.unsqueeze(1), float("-inf"))
        topk_scores, topk_indices = torch.topk(masked_scores, k=k, dim=-1)
        topk_weights = torch.softmax(topk_scores, dim=-1)
        topk_weights = torch.nan_to_num(topk_weights, nan=0.0, posinf=0.0, neginf=0.0)
        return topk_weights, topk_indices

    def _record_stats(
        self,
        visual_weights: torch.Tensor,
        kg_weights: torch.Tensor,
        num_visual: int,
        num_kg: int,
    ):
        self.last_stats = {
            "visual_link_density": float(self.top_k) / float(max(num_kg, 1)),
            "kg_link_density": float(self.top_k) / float(max(num_visual, 1)),
            "mean_visual_link_weight": float(visual_weights.mean().detach().cpu().item()),
            "mean_kg_link_weight": float(kg_weights.mean().detach().cpu().item()),
        }

    def forward(
        self,
        visual_nodes: torch.Tensor,
        kg_nodes: torch.Tensor,
        question_node: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        kg_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_nodes: [B, N_v, D]
            kg_nodes: [B, N_kg, D]
            question_node: [B, D]
            visual_mask: [B, N_v] valid visual nodes
            kg_mask: [B, N_kg] valid kg/textual nodes
        """
        bsz, num_visual, d_hidden = visual_nodes.shape
        _, num_kg, _ = kg_nodes.shape

        if visual_mask is None:
            visual_mask = torch.ones(
                bsz,
                num_visual,
                dtype=torch.bool,
                device=visual_nodes.device,
            )
        if kg_mask is None:
            kg_mask = torch.ones(
                bsz,
                num_kg,
                dtype=torch.bool,
                device=kg_nodes.device,
            )

        k_visual = min(self.top_k, num_kg)
        k_kg = min(self.top_k, num_visual)

        q_vis = self.question_to_visual(question_node).unsqueeze(1)
        q_kg = self.question_to_kg(question_node).unsqueeze(1)

        visual_query = self.visual_norm(self.visual_score_proj(visual_nodes) + q_vis)
        kg_query = self.kg_norm(self.kg_score_proj(kg_nodes) + q_kg)

        scores = torch.matmul(visual_query, kg_query.transpose(1, 2)) / self.scale

        visual_weights, visual_indices = self._masked_topk(scores, kg_mask, k_visual)
        visual_messages = self._batched_gather(self.kg_value_proj(kg_nodes), visual_indices)
        visual_messages = (visual_messages * visual_weights.unsqueeze(-1)).sum(dim=2)

        kg_weights, kg_indices = self._masked_topk(scores.transpose(1, 2), visual_mask, k_kg)
        kg_messages = self._batched_gather(self.visual_value_proj(visual_nodes), kg_indices)
        kg_messages = (kg_messages * kg_weights.unsqueeze(-1)).sum(dim=2)

        q_vis_full = question_node.unsqueeze(1).expand(-1, num_visual, -1)
        q_kg_full = question_node.unsqueeze(1).expand(-1, num_kg, -1)

        visual_gate = torch.sigmoid(
            self.visual_gate(torch.cat([visual_nodes, visual_messages, q_vis_full], dim=-1))
        )
        kg_gate = torch.sigmoid(self.kg_gate(torch.cat([kg_nodes, kg_messages, q_kg_full], dim=-1)))

        updated_visual = visual_nodes + self.dropout(visual_gate * visual_messages)
        updated_kg = kg_nodes + self.dropout(kg_gate * kg_messages)

        updated_visual = torch.where(visual_mask.unsqueeze(-1), updated_visual, visual_nodes)
        updated_kg = torch.where(kg_mask.unsqueeze(-1), updated_kg, kg_nodes)

        self._record_stats(visual_weights, kg_weights, num_visual=num_visual, num_kg=num_kg)
        return updated_visual, updated_kg
