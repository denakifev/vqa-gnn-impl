"""Runtime-path tests for the restored VQA-2 coursework extension."""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestVQACollate:
    def test_shape_and_keys(self):
        from src.datasets.vqa_collate import vqa_collate_fn
        from src.datasets.vqa_dataset import VQADemoDataset

        ds = VQADemoDataset(
            num_samples=4,
            num_answers=11,
            max_question_len=8,
            num_visual_nodes=4,
            max_kg_nodes=3,
            d_visual=8,
            d_kg=5,
        )
        batch = vqa_collate_fn([ds[i] for i in range(3)])

        assert batch["visual_features"].shape == (3, 4, 8)
        assert batch["graph_node_features"].shape == (3, 3, 5)
        assert batch["graph_adj"].shape == (3, 8, 8)
        assert batch["answer_scores"].shape == (3, 11)
        assert batch["labels"].shape == (3,)
        assert len(batch["question_id"]) == 3


class TestVQALossAndMetric:
    def test_soft_bce_loss_backward(self):
        from src.loss.vqa_loss import VQALoss

        logits = torch.randn(4, 7, requires_grad=True)
        answer_scores = torch.zeros(4, 7)
        answer_scores[:, 2] = 1.0
        labels = torch.full((4,), 2, dtype=torch.long)

        out = VQALoss(use_soft_labels=True)(
            logits=logits,
            answer_scores=answer_scores,
            labels=labels,
        )
        out["loss"].backward()
        assert out["loss"].shape == ()
        assert not torch.any(torch.isnan(logits.grad))

    def test_vqa_accuracy_uses_soft_score_of_prediction(self):
        from src.metrics.vqa_metric import VQAAccuracy

        logits = torch.tensor([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        answer_scores = torch.tensor([[0.0, 0.6, 1.0], [0.0, 1.0, 0.3]])

        acc = VQAAccuracy(name="vqa")(logits=logits, answer_scores=answer_scores)
        assert acc == pytest.approx(0.45)


class TestVQAModel:
    def test_forward_shape(self):
        from src.model.vqa_gnn import VQAGNNModel

        class _StubHF(torch.nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.config = MagicMock(hidden_size=hidden)
                self.embed = torch.nn.Embedding(128, hidden)

            def forward(self, input_ids, attention_mask):
                del attention_mask
                return MagicMock(last_hidden_state=self.embed(input_ids))

        with patch("transformers.AutoModel.from_pretrained", return_value=_StubHF(32)):
            model = VQAGNNModel(
                d_visual=8,
                d_kg=5,
                d_hidden=32,
                num_gnn_layers=2,
                num_heads=4,
                num_answers=13,
                text_encoder_name="stub",
                freeze_text_encoder=True,
            )

        B, Nv, Nkg = 2, 4, 3
        n_total = Nv + 1 + Nkg
        node_types = torch.tensor([[0] * Nv + [1] + [2] * Nkg] * B)
        batch = {
            "visual_features": torch.randn(B, Nv, 8),
            "question_input_ids": torch.randint(0, 100, (B, 6)),
            "question_attention_mask": torch.ones(B, 6, dtype=torch.long),
            "graph_node_features": torch.randn(B, Nkg, 5),
            "graph_adj": torch.ones(B, n_total, n_total),
            "graph_node_types": node_types,
        }
        assert model(**batch)["logits"].shape == (B, 13)

    def test_graph_link_mode_runs_and_emits_stats(self):
        from src.model.vqa_gnn import VQAGNNModel

        class _StubHF(torch.nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.config = MagicMock(hidden_size=hidden)
                self.embed = torch.nn.Embedding(128, hidden)

            def forward(self, input_ids, attention_mask):
                del attention_mask
                return MagicMock(last_hidden_state=self.embed(input_ids))

        with patch("transformers.AutoModel.from_pretrained", return_value=_StubHF(32)):
            model = VQAGNNModel(
                d_visual=8,
                d_kg=5,
                d_hidden=32,
                num_gnn_layers=2,
                num_heads=4,
                num_answers=13,
                text_encoder_name="stub",
                freeze_text_encoder=True,
                enable_graph_link_module=True,
                graph_link_top_k=2,
            )

        B, Nv, Nkg = 2, 4, 3
        n_total = Nv + 1 + Nkg
        node_types = torch.tensor([[0] * Nv + [1] + [2] * Nkg] * B)
        batch = {
            "visual_features": torch.randn(B, Nv, 8),
            "question_input_ids": torch.randint(0, 100, (B, 6)),
            "question_attention_mask": torch.ones(B, 6, dtype=torch.long),
            "graph_node_features": torch.randn(B, Nkg, 5),
            "graph_adj": torch.ones(B, n_total, n_total),
            "graph_node_types": node_types,
        }
        out = model(**batch)
        assert out["logits"].shape == (B, 13)
        assert "graph_link_stats" in out


class TestVQAHydraConfigs:
    def _compose(self, config_name: str, overrides: list[str]):
        from hydra import compose, initialize_config_dir

        repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
        config_dir = repo_root / "src" / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            return compose(config_name=config_name, overrides=overrides)

    def test_baseline_vqa_targets_vqa_model(self):
        cfg = self._compose("baseline_vqa", [])
        assert cfg.model._target_ == "src.model.VQAGNNModel"
        assert cfg.model.num_answers == 3129
        assert cfg.model.d_hidden == 512
        assert cfg.model.freeze_text_encoder is True
        assert cfg.dataloader.batch_size == 4
        assert "answer_scores" in cfg.trainer.device_tensors

    def test_real_vqa_dataset_contract_defaults_match_model(self):
        cfg = self._compose("baseline_vqa", ["datasets=vqa"])
        assert cfg.datasets.train.d_kg == cfg.model.d_kg
        assert cfg.datasets.val.d_kg == cfg.model.d_kg
        assert cfg.datasets.val.limit == 10000

    def test_graph_link_vqa_config_enables_module(self):
        cfg = self._compose("graph_link_vqa", [])
        assert cfg.model.enable_graph_link_module is True
        assert cfg.freeze_policy.freeze_all_baseline is False

    def test_graph_link_vqa_frozen_config_freezes_backbone(self):
        cfg = self._compose("graph_link_vqa_frozen", [])
        assert cfg.model.enable_graph_link_module is True
        assert cfg.freeze_policy.freeze_all_baseline is True

    def test_inference_vqa_skip_model_load_bare_override(self):
        cfg = self._compose("inference_vqa", ["inferencer.skip_model_load=True"])
        assert cfg.inferencer.skip_model_load is True
        assert cfg.model._target_ == "src.model.VQAGNNModel"
