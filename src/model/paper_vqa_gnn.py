"""
Paper-equation VQA-GNN model (arXiv:2205.11501, §4.1-§4.3).

Implements the bidirectional-fusion architecture with two modality-specialized
GNNs, explicit super nodes (z, p for VCR; z = q for GQA), and the paper's
concat-based readout (Eq 8). For GQA, the two-subgraph batch contract is
runnable via `GQADataset(emit_two_subgraphs=True)`. For VCR, the paper-equation
path still requires split scene/concept graph artifacts and is not the current
runtime path (see `GAP_ANALYSIS.md`).

Paper equations mapped to code:

    Eq 1-6  -> src.model.paper_rgat.PaperMultiRelationGATLayer (one per modality)
    Eq 7    -> PaperVCRModel / PaperGQAModel._fuse_z_across_modalities
    Eq 8    -> PaperVCRModel / PaperGQAModel._readout_concat
    Eq 9    -> PaperVCRModel.vcr_head / PaperGQAModel.gqa_head
    Eq 10   -> softmax applied implicitly by F.cross_entropy

Two public classes:

    PaperVCRModel: VCR candidate-scoring head. Expects super nodes z (QA-context
        from RoBERTa) and p (QA-concept from VinVL). Final head is R^{4D} -> R^1
        per candidate (paper Eq 9).

    PaperGQAModel: GQA classification head. Expects single super node q
        (context from RoBERTa). Final head is R^{3D} -> R^{num_answers}.

Items the caller must satisfy before running:

    * the dataset must emit `scene_subgraph_mask`, `concept_subgraph_mask`,
      and super-node indices (z, p, q). GQA emits this via
      `emit_two_subgraphs=True`; VCR does not yet emit the paper-equation
      split graph contract.
    * for VCR, p-node features require VinVL-style assets.
    * for both tasks, proper relation vocabularies are required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.gnn_core import HFQuestionEncoder
from src.model.paper_rgat import PaperMultiRelationGATLayer


class _ModalityGNNStack(nn.Module):
    """Stack of L paper-equation RGAT layers for a single modality."""

    def __init__(
        self,
        d_model: int,
        num_relations: int,
        num_node_types: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PaperMultiRelationGATLayer(
                    d_model=d_model,
                    num_relations=num_relations,
                    num_node_types=num_node_types,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_types: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                adj,
                edge_types=edge_types,
                node_types_src=node_types,
                node_types_dst=node_types,
            )
        return x


class _PaperFeatureProjections(nn.Module):
    """Linear projections for visual / scene-graph / concept-graph nodes."""

    def __init__(self, d_visual: int, d_kg: int, d_hidden: int):
        super().__init__()
        self.visual_proj = nn.Linear(d_visual, d_hidden)
        self.kg_proj = nn.Linear(d_kg, d_hidden)


class PaperVCRModel(nn.Module):
    """
    Paper-equation VCR candidate-scoring model (§4.1-§4.3).

    Architecture (one candidate pass, expanded B -> B*4 upstream):

        z = RoBERTa([question; candidate])[CLS]                       # §4.1
        p = mean_pool(VinVL(retrieved VG regions))                    # §4.1
        scene-graph subgraph  = {v_i}_{i=1..Ns} connected to z via r(e)
        concept-graph subgraph = {c_i}_{i=1..Nc} connected to z via r(q), r(a)

        for k in 0..L-1:
            h_{S}   = PaperMultiRelationGATLayer_S(h_{S}, adj_S, rel_S, types_S)
            h_{C}   = PaperMultiRelationGATLayer_C(h_{C}, adj_C, rel_C, types_C)
            z      <- f_z([z_from_S || z_from_C])                     # Eq 7

        pool_S, pool_C = masked_mean(h_S), masked_mean(h_C)
        h_a = [pool_S || pool_C || p || z]                            # Eq 8
        logit = f_c(h_a)                                              # Eq 9

    Per-candidate output: [B*4, 1]. Loss reshape handled by VCRLoss.
    """

    # Match legacy node-type ids for backward compat with existing tests.
    NODE_TYPE_Z = 0          # QA-context
    NODE_TYPE_P = 1          # QA-concept
    NODE_TYPE_SCENE = 2      # scene-graph node
    NODE_TYPE_CONCEPT = 3    # concept-graph node
    NODE_TYPE_QUESTION = 4   # (for GQA, unused here)
    NUM_NODE_TYPES = 5

    def __init__(
        self,
        d_visual: int = 2048,
        d_kg: int = 1024,
        d_hidden: int = 1024,
        num_gnn_layers: int = 5,
        num_relations_scene: int = 64,
        num_relations_concept: int = 64,
        dropout: float = 0.0,
        text_encoder_name: str = "roberta-large",
        freeze_text_encoder: bool = False,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_gnn_layers = num_gnn_layers

        self.proj = _PaperFeatureProjections(d_visual, d_kg, d_hidden)

        self.question_encoder = HFQuestionEncoder(
            d_hidden=d_hidden,
            model_name=text_encoder_name,
            freeze=freeze_text_encoder,
            dropout=dropout,
        )

        self.scene_gnn = _ModalityGNNStack(
            d_model=d_hidden,
            num_relations=num_relations_scene,
            num_node_types=self.NUM_NODE_TYPES,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        self.concept_gnn = _ModalityGNNStack(
            d_model=d_hidden,
            num_relations=num_relations_concept,
            num_node_types=self.NUM_NODE_TYPES,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # Eq 7: f_z: R^{2D} -> R^D
        self.f_z = nn.Linear(2 * d_hidden, d_hidden)

        # Eq 9: f_c: R^{4D} -> R^1 for VCR
        self.vcr_head = nn.Linear(4 * d_hidden, 1)

    def _mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pooling over N. mask: [B, N] bool."""
        if mask.dtype != torch.bool:
            mask = mask != 0
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (x * mask_f).sum(dim=1) / denom

    def forward(
        self,
        *,
        # z super-node features [B, d_hidden] or text inputs to encode z
        question_input_ids: torch.Tensor | None = None,
        question_attention_mask: torch.Tensor | None = None,
        z_features: torch.Tensor | None = None,
        # p super-node features [B, d_hidden] (from VinVL mean-pool upstream)
        p_features: torch.Tensor,
        # scene-graph subgraph
        scene_node_features: torch.Tensor,
        scene_adj: torch.Tensor,
        scene_edge_types: torch.Tensor,
        scene_node_types: torch.Tensor,
        scene_node_is_z_index: torch.Tensor,  # [B] long — column index of z-node within scene graph
        # concept-graph subgraph
        concept_node_features: torch.Tensor,
        concept_adj: torch.Tensor,
        concept_edge_types: torch.Tensor,
        concept_node_types: torch.Tensor,
        concept_node_is_z_index: torch.Tensor,  # [B] long
        **_batch,
    ) -> dict:
        if z_features is None:
            if question_input_ids is None or question_attention_mask is None:
                raise ValueError(
                    "PaperVCRModel.forward requires either `z_features` or both "
                    "`question_input_ids` and `question_attention_mask`."
                )
            z = self.question_encoder(question_input_ids, question_attention_mask)
        else:
            z = z_features

        # Project non-text node features to d_hidden.
        h_s = self.proj.visual_proj(scene_node_features)  # [B, Ns, D]
        h_c = self.proj.kg_proj(concept_node_features)    # [B, Nc, D]

        B, Ns, D = h_s.shape
        _, Nc, _ = h_c.shape

        # Inject z into scene and concept subgraphs at the stored index.
        batch_idx = torch.arange(B, device=h_s.device)
        h_s = h_s.clone()
        h_c = h_c.clone()
        h_s[batch_idx, scene_node_is_z_index] = z
        h_c[batch_idx, concept_node_is_z_index] = z

        # L-layer bidirectional fusion: update each subgraph in parallel, then
        # concatenate the z-views (Eq 7).
        for k in range(self.num_gnn_layers):
            h_s_new = self.scene_gnn.layers[k](
                h_s,
                scene_adj,
                edge_types=scene_edge_types,
                node_types_src=scene_node_types,
                node_types_dst=scene_node_types,
            )
            h_c_new = self.concept_gnn.layers[k](
                h_c,
                concept_adj,
                edge_types=concept_edge_types,
                node_types_src=concept_node_types,
                node_types_dst=concept_node_types,
            )
            z_from_s = h_s_new[batch_idx, scene_node_is_z_index]
            z_from_c = h_c_new[batch_idx, concept_node_is_z_index]
            z_next = self.f_z(torch.cat([z_from_s, z_from_c], dim=-1))

            # Write back the fused z to both subgraphs before the next layer.
            h_s_new[batch_idx, scene_node_is_z_index] = z_next
            h_c_new[batch_idx, concept_node_is_z_index] = z_next
            h_s, h_c, z = h_s_new, h_c_new, z_next

        # Pool modality subgraphs excluding the z position (paper Eq 8 pools S
        # and C separately; z is added explicitly).
        scene_pool_mask = torch.ones(B, Ns, dtype=torch.bool, device=h_s.device)
        scene_pool_mask[batch_idx, scene_node_is_z_index] = False
        concept_pool_mask = torch.ones(B, Nc, dtype=torch.bool, device=h_c.device)
        concept_pool_mask[batch_idx, concept_node_is_z_index] = False

        pool_s = self._mean_pool(h_s, scene_pool_mask)
        pool_c = self._mean_pool(h_c, concept_pool_mask)

        # Eq 8-9: concat and score.
        h_a = torch.cat([pool_s, pool_c, p_features, z], dim=-1)  # [B, 4D]
        logits = self.vcr_head(h_a)  # [B, 1]
        return {"logits": logits}


class PaperGQAModel(nn.Module):
    """
    Paper-equation GQA classification model (§5.1 GQA block).

    GQA differs from VCR:
      * the super node is "node q" (question as context) rather than z;
      * there is no p-node;
      * structured graph is visual SG + textual SG (paper says "visual and
        textual scene graphs ... fully connected" through q);
      * classification head is R^{3D} -> R^{num_answers}.

    The implementation keeps the VCR-style two-modality GNN pattern (S and C
    here interpreted as visual SG and textual SG), and replaces the VCR head
    with a 1842-way classifier.
    """

    # Use PaperVCRModel's node-type ids for consistency.
    NUM_NODE_TYPES = PaperVCRModel.NUM_NODE_TYPES

    def __init__(
        self,
        d_visual: int = 2048,
        d_kg: int = 600,
        d_hidden: int = 1024,
        num_gnn_layers: int = 5,
        num_relations_visual: int = 64,
        num_relations_textual: int = 64,
        num_answers: int = 1842,
        dropout: float = 0.0,
        text_encoder_name: str = "roberta-large",
        freeze_text_encoder: bool = False,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_gnn_layers = num_gnn_layers
        self.num_answers = num_answers

        self.proj = _PaperFeatureProjections(d_visual, d_kg, d_hidden)

        self.question_encoder = HFQuestionEncoder(
            d_hidden=d_hidden,
            model_name=text_encoder_name,
            freeze=freeze_text_encoder,
            dropout=dropout,
        )

        self.visual_gnn = _ModalityGNNStack(
            d_model=d_hidden,
            num_relations=num_relations_visual,
            num_node_types=self.NUM_NODE_TYPES,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        self.textual_gnn = _ModalityGNNStack(
            d_model=d_hidden,
            num_relations=num_relations_textual,
            num_node_types=self.NUM_NODE_TYPES,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )

        # Paper GQA does not explicitly state f_z for q; we mirror Eq 7.
        self.f_q = nn.Linear(2 * d_hidden, d_hidden)

        # Classification head: [pool(visual_SG) || pool(textual_SG) || q] -> 1842
        self.gqa_head = nn.Linear(3 * d_hidden, num_answers)

    def _mean_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask != 0
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return (x * mask_f).sum(dim=1) / denom

    def forward(
        self,
        *,
        question_input_ids: torch.Tensor | None = None,
        question_attention_mask: torch.Tensor | None = None,
        q_features: torch.Tensor | None = None,
        visual_node_features: torch.Tensor,
        visual_adj: torch.Tensor,
        visual_edge_types: torch.Tensor,
        visual_node_types: torch.Tensor,
        visual_node_is_q_index: torch.Tensor,
        textual_node_features: torch.Tensor,
        textual_adj: torch.Tensor,
        textual_edge_types: torch.Tensor,
        textual_node_types: torch.Tensor,
        textual_node_is_q_index: torch.Tensor,
        **_batch,
    ) -> dict:
        if q_features is None:
            if question_input_ids is None or question_attention_mask is None:
                raise ValueError(
                    "PaperGQAModel.forward requires either `q_features` or both "
                    "`question_input_ids` and `question_attention_mask`."
                )
            q = self.question_encoder(question_input_ids, question_attention_mask)
        else:
            q = q_features

        h_v = self.proj.visual_proj(visual_node_features)
        h_t = self.proj.kg_proj(textual_node_features)

        B, Nv, D = h_v.shape
        _, Nt, _ = h_t.shape

        batch_idx = torch.arange(B, device=h_v.device)
        h_v = h_v.clone()
        h_t = h_t.clone()
        h_v[batch_idx, visual_node_is_q_index] = q
        h_t[batch_idx, textual_node_is_q_index] = q

        for k in range(self.num_gnn_layers):
            h_v_new = self.visual_gnn.layers[k](
                h_v,
                visual_adj,
                edge_types=visual_edge_types,
                node_types_src=visual_node_types,
                node_types_dst=visual_node_types,
            )
            h_t_new = self.textual_gnn.layers[k](
                h_t,
                textual_adj,
                edge_types=textual_edge_types,
                node_types_src=textual_node_types,
                node_types_dst=textual_node_types,
            )
            q_from_v = h_v_new[batch_idx, visual_node_is_q_index]
            q_from_t = h_t_new[batch_idx, textual_node_is_q_index]
            q_next = self.f_q(torch.cat([q_from_v, q_from_t], dim=-1))

            h_v_new[batch_idx, visual_node_is_q_index] = q_next
            h_t_new[batch_idx, textual_node_is_q_index] = q_next
            h_v, h_t, q = h_v_new, h_t_new, q_next

        v_mask = torch.ones(B, Nv, dtype=torch.bool, device=h_v.device)
        v_mask[batch_idx, visual_node_is_q_index] = False
        t_mask = torch.ones(B, Nt, dtype=torch.bool, device=h_t.device)
        t_mask[batch_idx, textual_node_is_q_index] = False

        pool_v = self._mean_pool(h_v, v_mask)
        pool_t = self._mean_pool(h_t, t_mask)

        h_a = torch.cat([pool_v, pool_t, q], dim=-1)  # [B, 3D]
        logits = self.gqa_head(h_a)  # [B, num_answers]
        return {"logits": logits}
