import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGraphLinkModule(nn.Module):
    """
    MAGIC-inspired implicit graph augmentation block.

    Compared with the earlier aggressive pre-GNN linker, this variant is much
    more conservative:

    - cross-graph links are scored by cosine similarity in a shared space
    - links are filtered by top-k and dynamic relevance thresholds
    - the resulting heterogeneous graph is processed with a lightweight
      2-layer graph-convolution style propagation
    - outputs are injected through a learnable residual scale initialized at 0

    The module therefore starts as an exact identity map and only deviates from
    the baseline when training discovers useful cross-graph structure.
    """

    RELEVANCE_HIGH = 1.0
    RELEVANCE_MEDIUM = 0.5
    RELEVANCE_LOW = 0.0

    def __init__(
        self,
        d_hidden: int,
        top_k: int = 4,
        dropout: float = 0.1,
        threshold_std_scale: float = 0.5,
    ):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self.top_k = int(top_k)
        self.threshold_std_scale = float(threshold_std_scale)

        self.visual_score_proj = nn.Linear(d_hidden, d_hidden)
        self.kg_score_proj = nn.Linear(d_hidden, d_hidden)
        self.question_score_proj = nn.Linear(d_hidden, d_hidden)

        self.gcn_linear1 = nn.Linear(d_hidden, d_hidden)
        self.gcn_linear2 = nn.Linear(d_hidden, d_hidden)

        self.visual_out_proj = nn.Linear(d_hidden, d_hidden)
        self.kg_out_proj = nn.Linear(d_hidden, d_hidden)

        self.visual_norm = nn.LayerNorm(d_hidden)
        self.kg_norm = nn.LayerNorm(d_hidden)
        self.graph_norm1 = nn.LayerNorm(d_hidden)
        self.graph_norm2 = nn.LayerNorm(d_hidden)

        # Conservative start: the module is initially an identity mapping.
        self.visual_residual_scale = nn.Parameter(torch.tensor(0.0))
        self.kg_residual_scale = nn.Parameter(torch.tensor(0.0))

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
    def _masked_topk(
        scores: torch.Tensor,
        target_mask: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_scores = scores.masked_fill(~target_mask.unsqueeze(1), float("-inf"))
        topk_scores, topk_indices = torch.topk(masked_scores, k=k, dim=-1)
        return topk_scores, topk_indices

    @staticmethod
    def _batched_scatter(
        values: torch.Tensor,
        indices: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        """
        Scatter top-k row values back into a dense [B, Q, size] matrix.
        """
        dense = torch.zeros(
            values.shape[0],
            values.shape[1],
            size,
            dtype=values.dtype,
            device=values.device,
        )
        return dense.scatter(dim=-1, index=indices, src=values)

    def _compute_thresholds(self, scores: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MAGIC-inspired dynamic thresholds using mean ± std * scale over the
        valid candidate pool for each sample.
        """
        masked_scores = scores.masked_fill(~valid_mask, 0.0)
        counts = valid_mask.sum(dim=(1, 2)).clamp_min(1).to(scores.dtype)
        mean = masked_scores.sum(dim=(1, 2)) / counts
        centered = torch.where(valid_mask, scores - mean.view(-1, 1, 1), torch.zeros_like(scores))
        var = (centered.pow(2).sum(dim=(1, 2)) / counts).clamp_min(0.0)
        std = torch.sqrt(var)
        low_thr = mean - self.threshold_std_scale * std
        high_thr = mean + self.threshold_std_scale * std
        return low_thr, high_thr

    def _relevance_weights(
        self,
        selected_scores: torch.Tensor,
        low_thr: torch.Tensor,
        high_thr: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        low_thr = low_thr.view(-1, 1, 1)
        high_thr = high_thr.view(-1, 1, 1)

        high = selected_scores >= high_thr
        medium = (selected_scores >= low_thr) & (selected_scores < high_thr)
        low = selected_scores < low_thr

        relevance = torch.where(
            high,
            torch.full_like(selected_scores, self.RELEVANCE_HIGH),
            torch.where(
                medium,
                torch.full_like(selected_scores, self.RELEVANCE_MEDIUM),
                torch.full_like(selected_scores, self.RELEVANCE_LOW),
            ),
        )
        stats = {
            "high_frac": float(high.float().mean().detach().cpu().item()),
            "medium_frac": float(medium.float().mean().detach().cpu().item()),
            "low_frac": float(low.float().mean().detach().cpu().item()),
        }
        return relevance, stats

    def _build_cross_weights(
        self,
        scores: torch.Tensor,
        visual_mask: torch.Tensor,
        kg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        bsz, num_visual, num_kg = scores.shape
        valid_pairs = visual_mask.unsqueeze(2) & kg_mask.unsqueeze(1)
        low_thr, high_thr = self._compute_thresholds(scores, valid_pairs)

        k_visual = min(self.top_k, num_kg)
        k_kg = min(self.top_k, num_visual)

        vis_scores, vis_indices = self._masked_topk(scores, kg_mask, k_visual)
        vis_relevance, vis_stats = self._relevance_weights(vis_scores, low_thr, high_thr)
        vis_weights = torch.softmax(vis_scores.masked_fill(vis_relevance == 0, float("-inf")), dim=-1)
        vis_weights = torch.nan_to_num(vis_weights, nan=0.0, posinf=0.0, neginf=0.0) * vis_relevance
        vis_weights = vis_weights / vis_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        vis_dense = self._batched_scatter(vis_weights, vis_indices, size=num_kg)

        kg_scores, kg_indices = self._masked_topk(scores.transpose(1, 2), visual_mask, k_kg)
        kg_relevance, kg_stats = self._relevance_weights(kg_scores, low_thr, high_thr)
        kg_weights = torch.softmax(kg_scores.masked_fill(kg_relevance == 0, float("-inf")), dim=-1)
        kg_weights = torch.nan_to_num(kg_weights, nan=0.0, posinf=0.0, neginf=0.0) * kg_relevance
        kg_weights = kg_weights / kg_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        kg_dense = self._batched_scatter(kg_weights, kg_indices, size=num_visual)

        cross_weights = torch.maximum(vis_dense, kg_dense.transpose(1, 2))
        cross_weights = cross_weights * valid_pairs.to(cross_weights.dtype)

        stats = {
            "mean_cross_weight": float(cross_weights.mean().detach().cpu().item()),
            "visual_high_frac": vis_stats["high_frac"],
            "visual_medium_frac": vis_stats["medium_frac"],
            "kg_high_frac": kg_stats["high_frac"],
            "kg_medium_frac": kg_stats["medium_frac"],
        }
        return cross_weights, stats

    @staticmethod
    def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adj / degree

    def _graph_conv(self, adj: torch.Tensor, x: torch.Tensor, linear: nn.Linear, norm: nn.LayerNorm) -> torch.Tensor:
        propagated = torch.matmul(adj, x)
        out = linear(propagated)
        out = F.gelu(out)
        out = self.dropout(out)
        return norm(out + x)

    def _record_stats(
        self,
        cross_weights: torch.Tensor,
        num_visual: int,
        num_kg: int,
        extra_stats: dict[str, float],
    ):
        active_links = (cross_weights > 0).float()
        self.last_stats = {
            "visual_link_density": float(active_links.sum().detach().cpu().item())
            / float(max(cross_weights.shape[0] * num_visual * max(num_kg, 1), 1)),
            "kg_link_density": float(active_links.sum().detach().cpu().item())
            / float(max(cross_weights.shape[0] * num_kg * max(num_visual, 1), 1)),
            "visual_residual_scale": float(torch.tanh(self.visual_residual_scale).detach().cpu().item()),
            "kg_residual_scale": float(torch.tanh(self.kg_residual_scale).detach().cpu().item()),
            **extra_stats,
        }

    def forward(
        self,
        visual_nodes: torch.Tensor,
        kg_nodes: torch.Tensor,
        question_node: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        kg_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_visual, _ = visual_nodes.shape
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

        q_bias = self.question_score_proj(question_node).unsqueeze(1)
        visual_query = F.normalize(
            self.visual_norm(self.visual_score_proj(visual_nodes) + q_bias),
            dim=-1,
        )
        kg_query = F.normalize(
            self.kg_norm(self.kg_score_proj(kg_nodes) + q_bias),
            dim=-1,
        )
        cosine_scores = torch.matmul(visual_query, kg_query.transpose(1, 2))

        cross_weights, extra_stats = self._build_cross_weights(
            cosine_scores,
            visual_mask=visual_mask,
            kg_mask=kg_mask,
        )

        n_total = num_visual + 1 + num_kg
        adjacency = torch.zeros(
            bsz,
            n_total,
            n_total,
            dtype=visual_nodes.dtype,
            device=visual_nodes.device,
        )

        question_idx = num_visual
        kg_start = num_visual + 1

        adjacency[:, :num_visual, kg_start:] = cross_weights
        adjacency[:, kg_start:, :num_visual] = cross_weights.transpose(1, 2)

        adjacency[:, :num_visual, question_idx] = visual_mask.to(adjacency.dtype)
        adjacency[:, question_idx, :num_visual] = visual_mask.to(adjacency.dtype)
        adjacency[:, kg_start:, question_idx] = kg_mask.to(adjacency.dtype)
        adjacency[:, question_idx, kg_start:] = kg_mask.to(adjacency.dtype)

        adjacency = adjacency + torch.eye(n_total, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        adjacency = self._normalize_adjacency(adjacency)

        graph_states = torch.cat([visual_nodes, question_node.unsqueeze(1), kg_nodes], dim=1)
        graph_states = self._graph_conv(adjacency, graph_states, self.gcn_linear1, self.graph_norm1)
        graph_states = self._graph_conv(adjacency, graph_states, self.gcn_linear2, self.graph_norm2)

        visual_delta = self.visual_out_proj(graph_states[:, :num_visual])
        kg_delta = self.kg_out_proj(graph_states[:, kg_start:])

        visual_scale = torch.tanh(self.visual_residual_scale)
        kg_scale = torch.tanh(self.kg_residual_scale)

        updated_visual = visual_nodes + visual_scale * self.dropout(visual_delta)
        updated_kg = kg_nodes + kg_scale * self.dropout(kg_delta)

        updated_visual = torch.where(visual_mask.unsqueeze(-1), updated_visual, visual_nodes)
        updated_kg = torch.where(kg_mask.unsqueeze(-1), updated_kg, kg_nodes)

        self._record_stats(
            cross_weights,
            num_visual=num_visual,
            num_kg=num_kg,
            extra_stats=extra_stats,
        )
        return updated_visual, updated_kg
