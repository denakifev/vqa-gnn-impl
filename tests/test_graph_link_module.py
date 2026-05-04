from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.model_freeze import apply_freeze_policy, count_parameters


def _stub_hf_model(hidden: int):
    class _StubHF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MagicMock(hidden_size=hidden)
            self.embed = torch.nn.Embedding(128, hidden)

        def forward(self, input_ids, attention_mask):
            del attention_mask
            return MagicMock(last_hidden_state=self.embed(input_ids))

    return _StubHF()


class TestSparseGraphLinkModule:
    def test_output_shapes(self):
        from src.model.graph_link import SparseGraphLinkModule

        module = SparseGraphLinkModule(d_hidden=8, top_k=2, dropout=0.0)
        visual = torch.randn(2, 4, 8)
        kg = torch.randn(2, 3, 8)
        question = torch.randn(2, 8)
        visual_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        kg_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

        visual_out, kg_out = module(visual, kg, question, visual_mask=visual_mask, kg_mask=kg_mask)

        assert visual_out.shape == visual.shape
        assert kg_out.shape == kg.shape
        assert module.last_stats["visual_link_density"] == pytest.approx(2 / 3)

    def test_masked_topk_prefers_high_scores(self):
        from src.model.graph_link import SparseGraphLinkModule

        module = SparseGraphLinkModule(d_hidden=4, top_k=2, dropout=0.0)
        scores = torch.tensor(
            [[[1.0, 3.0, 2.0, -5.0], [0.1, 0.2, 0.9, 0.8]]],
            dtype=torch.float32,
        )
        target_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        _, indices = module._masked_topk(scores, target_mask=target_mask, k=2)
        assert indices.tolist() == [[[1, 2], [2, 1]]]


class TestGQAGraphLinkIntegration:
    def _toy_model(self, enable_graph_link: bool):
        from src.model.gqa_model import GQAVQAGNNModel

        with patch("transformers.AutoModel.from_pretrained", return_value=_stub_hf_model(32)):
            model = GQAVQAGNNModel(
                d_visual=8,
                d_kg=4,
                d_hidden=32,
                num_gnn_layers=2,
                num_heads=4,
                num_answers=5,
                num_relations=6,
                text_encoder_name="stub",
                enable_graph_link_module=enable_graph_link,
                graph_link_top_k=2,
            )
        model.eval()
        return model

    def _toy_batch(self):
        bsz, num_visual, num_kg = 2, 3, 2
        n_total = num_visual + 1 + num_kg
        return {
            "visual_features": torch.randn(bsz, num_visual, 8),
            "question_input_ids": torch.randint(0, 100, (bsz, 4)),
            "question_attention_mask": torch.ones(bsz, 4, dtype=torch.long),
            "graph_node_features": torch.randn(bsz, num_kg, 4),
            "graph_adj": torch.ones(bsz, n_total, n_total),
            "graph_edge_types": torch.randint(0, 6, (bsz, n_total, n_total), dtype=torch.long),
            "graph_node_types": torch.tensor([[0, 0, 0, 1, 2, 2], [0, 0, 0, 1, 2, 2]]),
        }

    def test_baseline_mode_has_no_graph_link_outputs(self):
        model = self._toy_model(enable_graph_link=False)
        out = model(**self._toy_batch())
        assert out["logits"].shape == (2, 5)
        assert "graph_link_stats" not in out

    def test_graph_link_mode_runs_and_emits_stats(self):
        model = self._toy_model(enable_graph_link=True)
        out = model(**self._toy_batch())
        assert out["logits"].shape == (2, 5)
        assert "graph_link_stats" in out
        assert "visual_link_density" in out["graph_link_stats"]


class TestFreezePolicy:
    def _toy_model(self):
        from src.model.gqa_model import GQAVQAGNNModel

        with patch("transformers.AutoModel.from_pretrained", return_value=_stub_hf_model(32)):
            return GQAVQAGNNModel(
                d_visual=8,
                d_kg=4,
                d_hidden=32,
                num_gnn_layers=2,
                num_heads=4,
                num_answers=5,
                num_relations=6,
                text_encoder_name="stub",
                enable_graph_link_module=True,
                graph_link_top_k=2,
            )

    def test_freeze_all_baseline_leaves_graph_link_trainable(self):
        model = self._toy_model()
        apply_freeze_policy(model, {"freeze_all_baseline": True})

        assert model.graph_link_module is not None
        assert any(param.requires_grad for param in model.graph_link_module.parameters())
        assert not any(param.requires_grad for param in model.gnn_layers.parameters())
        assert not any(param.requires_grad for param in model.classifier.parameters())
        assert not any(param.requires_grad for param in model.visual_proj.parameters())

        stats = count_parameters(model)
        assert stats["trainable"] > 0
        assert stats["trainable"] < stats["total"]


class TestGraphLinkHydraConfigs:
    def _compose(self, config_name: str, overrides: list[str]):
        from hydra import compose, initialize_config_dir

        repo_root = Path(__file__).resolve().parents[1]
        config_dir = repo_root / "src" / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            return compose(config_name=config_name, overrides=overrides)

    def test_graph_link_config_enables_module(self):
        cfg = self._compose("graph_link_gqa", [])
        assert cfg.model.enable_graph_link_module is True
        assert cfg.freeze_policy.freeze_all_baseline is False

    def test_frozen_graph_link_config_freezes_backbone(self):
        cfg = self._compose("graph_link_gqa_frozen", [])
        assert cfg.model.enable_graph_link_module is True
        assert cfg.freeze_policy.freeze_all_baseline is True
