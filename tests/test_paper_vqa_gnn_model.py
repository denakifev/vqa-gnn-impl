from types import SimpleNamespace

import torch
import torch.nn as nn


class _TinyHFEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.emb = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def _patch_hf(monkeypatch, hidden_size: int = 8):
    import transformers

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *_args, **_kwargs: _TinyHFEncoder(hidden_size),
    )


def _full_adj(batch: int, nodes: int) -> torch.Tensor:
    return torch.ones(batch, nodes, nodes)


def _edge_types(batch: int, nodes: int, num_relations: int) -> torch.Tensor:
    return torch.randint(0, num_relations, (batch, nodes, nodes), dtype=torch.long)


def _node_types(batch: int, nodes: int) -> torch.Tensor:
    return torch.randint(0, 5, (batch, nodes), dtype=torch.long)


def test_vcr_head_is_concat_4d_to_1(monkeypatch):
    _patch_hf(monkeypatch)
    from src.model.paper_vqa_gnn import PaperVCRModel

    model = PaperVCRModel(
        d_visual=6,
        d_kg=5,
        d_hidden=8,
        num_gnn_layers=1,
        num_relations_scene=4,
        num_relations_concept=4,
        dropout=0.0,
    )

    assert model.f_z.in_features == 16
    assert model.f_z.out_features == 8
    assert model.vcr_head.in_features == 32
    assert model.vcr_head.out_features == 1


def test_vcr_forward_outputs_candidate_scores(monkeypatch):
    _patch_hf(monkeypatch)
    from src.model.paper_vqa_gnn import PaperVCRModel

    torch.manual_seed(0)
    B, Ns, Nc = 2, 4, 3
    model = PaperVCRModel(
        d_visual=6,
        d_kg=5,
        d_hidden=8,
        num_gnn_layers=1,
        num_relations_scene=4,
        num_relations_concept=4,
        dropout=0.0,
    )

    out = model(
        z_features=torch.randn(B, 8),
        p_features=torch.randn(B, 8),
        scene_node_features=torch.randn(B, Ns, 6),
        scene_adj=_full_adj(B, Ns),
        scene_edge_types=_edge_types(B, Ns, 4),
        scene_node_types=_node_types(B, Ns),
        scene_node_is_z_index=torch.zeros(B, dtype=torch.long),
        concept_node_features=torch.randn(B, Nc, 5),
        concept_adj=_full_adj(B, Nc),
        concept_edge_types=_edge_types(B, Nc, 4),
        concept_node_types=_node_types(B, Nc),
        concept_node_is_z_index=torch.zeros(B, dtype=torch.long),
    )

    assert out["logits"].shape == (B, 1)
    assert torch.isfinite(out["logits"]).all()


def test_gqa_head_is_concat_3d_to_vocab(monkeypatch):
    _patch_hf(monkeypatch)
    from src.model.paper_vqa_gnn import PaperGQAModel

    model = PaperGQAModel(
        d_visual=6,
        d_kg=5,
        d_hidden=8,
        num_gnn_layers=1,
        num_relations_visual=4,
        num_relations_textual=4,
        num_answers=7,
        dropout=0.0,
    )

    assert model.f_q.in_features == 16
    assert model.f_q.out_features == 8
    assert model.gqa_head.in_features == 24
    assert model.gqa_head.out_features == 7


def test_gqa_forward_outputs_answer_vocab_logits(monkeypatch):
    _patch_hf(monkeypatch)
    from src.model.paper_vqa_gnn import PaperGQAModel

    torch.manual_seed(1)
    B, Nv, Nt = 2, 4, 3
    model = PaperGQAModel(
        d_visual=6,
        d_kg=5,
        d_hidden=8,
        num_gnn_layers=1,
        num_relations_visual=4,
        num_relations_textual=4,
        num_answers=7,
        dropout=0.0,
    )

    out = model(
        q_features=torch.randn(B, 8),
        visual_node_features=torch.randn(B, Nv, 6),
        visual_adj=_full_adj(B, Nv),
        visual_edge_types=_edge_types(B, Nv, 4),
        visual_node_types=_node_types(B, Nv),
        visual_node_is_q_index=torch.zeros(B, dtype=torch.long),
        textual_node_features=torch.randn(B, Nt, 5),
        textual_adj=_full_adj(B, Nt),
        textual_edge_types=_edge_types(B, Nt, 4),
        textual_node_types=_node_types(B, Nt),
        textual_node_is_q_index=torch.zeros(B, dtype=torch.long),
    )

    assert out["logits"].shape == (B, 7)
    assert torch.isfinite(out["logits"]).all()
