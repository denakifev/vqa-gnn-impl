import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGraphLinkModule(nn.Module):
    """
    Sparse cross-graph attention branch for late fusion.

    The module builds question-conditioned sparse links between visual scene
    nodes and textual/kg nodes, applies bidirectional multi-head cross-attention
    restricted to those links, and produces a pooled `link_repr`.

    This branch does not rewrite baseline node states. It is meant to be fused
    with the baseline graph representation later in the model.
    """

    def __init__(
        self,
        d_hidden: int,
        top_k: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        threshold_std_scale: float = 0.5,
    ):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if d_hidden % num_heads != 0:
            raise ValueError(
                f"d_hidden ({d_hidden}) must be divisible by num_heads ({num_heads})"
            )

        self.top_k = int(top_k)
        self.num_heads = int(num_heads)
        self.head_dim = d_hidden // num_heads
        self.scale = self.head_dim**-0.5
        self.threshold_std_scale = float(threshold_std_scale)

        self.visual_score_proj = nn.Linear(d_hidden, d_hidden)
        self.kg_score_proj = nn.Linear(d_hidden, d_hidden)
        self.question_score_proj = nn.Linear(d_hidden, d_hidden)

        self.scene_q = nn.Linear(d_hidden, d_hidden)
        self.scene_k = nn.Linear(d_hidden, d_hidden)
        self.scene_v = nn.Linear(d_hidden, d_hidden)

        self.kg_q = nn.Linear(d_hidden, d_hidden)
        self.kg_k = nn.Linear(d_hidden, d_hidden)
        self.kg_v = nn.Linear(d_hidden, d_hidden)

        self.scene_out = nn.Linear(d_hidden, d_hidden)
        self.kg_out = nn.Linear(d_hidden, d_hidden)

        self.scene_norm = nn.LayerNorm(d_hidden)
        self.kg_norm = nn.LayerNorm(d_hidden)
        self.question_norm = nn.LayerNorm(d_hidden)

        self.scene_pool = nn.Linear(d_hidden, 1)
        self.kg_pool = nn.Linear(d_hidden, 1)
        self.link_proj = nn.Sequential(
            nn.Linear(d_hidden * 3, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
        )

        self.dropout = nn.Dropout(dropout)
        self.last_stats = {}
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_nodes, _ = x.shape
        return x.view(bsz, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

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
    def _batched_scatter(values: torch.Tensor, indices: torch.Tensor, size: int) -> torch.Tensor:
        dense = torch.zeros(
            values.shape[0],
            values.shape[1],
            size,
            dtype=values.dtype,
            device=values.device,
        )
        return dense.scatter(dim=-1, index=indices, src=values)

    def _compute_thresholds(
        self,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            torch.ones_like(selected_scores),
            torch.where(medium, torch.full_like(selected_scores, 0.5), torch.zeros_like(selected_scores)),
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

    def _sparse_cross_attention(
        self,
        query_nodes: torch.Tensor,
        key_nodes: torch.Tensor,
        edge_weights: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        q = self._reshape_heads(q_proj(query_nodes))
        k = self._reshape_heads(k_proj(key_nodes))
        v = self._reshape_heads(v_proj(key_nodes))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        pair_mask = query_mask.unsqueeze(-1) & key_mask.unsqueeze(1)
        sparse_mask = edge_weights > 0
        full_mask = pair_mask & sparse_mask

        attn = attn.masked_fill(~full_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = attn * edge_weights.unsqueeze(1)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(query_nodes.shape[0], query_nodes.shape[1], -1)
        out = out_proj(out)
        out = self.dropout(out)
        out = norm(query_nodes + out)
        out = torch.where(query_mask.unsqueeze(-1), out, query_nodes)
        return out

    @staticmethod
    def _masked_pool(nodes: torch.Tensor, mask: torch.Tensor, score_proj: nn.Linear) -> torch.Tensor:
        scores = score_proj(nodes).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)
        return (nodes * weights).sum(dim=1)

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
            **extra_stats,
        }

    def forward(
        self,
        visual_nodes: torch.Tensor,
        kg_nodes: torch.Tensor,
        question_node: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        kg_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
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
            self.scene_norm(self.visual_score_proj(visual_nodes) + q_bias),
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

        scene_ctx = self._sparse_cross_attention(
            query_nodes=visual_nodes,
            key_nodes=kg_nodes,
            edge_weights=cross_weights,
            query_mask=visual_mask,
            key_mask=kg_mask,
            q_proj=self.scene_q,
            k_proj=self.scene_k,
            v_proj=self.scene_v,
            out_proj=self.scene_out,
            norm=self.scene_norm,
        )
        kg_ctx = self._sparse_cross_attention(
            query_nodes=kg_nodes,
            key_nodes=visual_nodes,
            edge_weights=cross_weights.transpose(1, 2),
            query_mask=kg_mask,
            key_mask=visual_mask,
            q_proj=self.kg_q,
            k_proj=self.kg_k,
            v_proj=self.kg_v,
            out_proj=self.kg_out,
            norm=self.kg_norm,
        )

        scene_repr = self._masked_pool(scene_ctx, visual_mask, self.scene_pool)
        kg_repr = self._masked_pool(kg_ctx, kg_mask, self.kg_pool)
        question_repr = self.question_norm(question_node)

        link_repr = self.link_proj(torch.cat([scene_repr, question_repr, kg_repr], dim=-1))

        self._record_stats(
            cross_weights,
            num_visual=num_visual,
            num_kg=num_kg,
            extra_stats=extra_stats,
        )
        return link_repr, self.last_stats
