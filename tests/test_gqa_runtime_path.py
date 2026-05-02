"""
GQA runtime-path validation tests for the paper-aligned baseline.

These tests do not require real data or pretrained model downloads.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestGQACollate:
    def test_shape(self):
        from src.datasets.gqa_collate import gqa_collate_fn
        from src.datasets.gqa_dataset import GQADemoDataset

        ds = GQADemoDataset(
            num_samples=4,
            num_answers=10,
            max_question_len=8,
            num_visual_nodes=4,
            max_kg_nodes=3,
            d_visual=8,
            d_kg=300,
        )
        batch = gqa_collate_fn([ds[i] for i in range(3)])
        assert batch["visual_features"].shape[0] == 3
        assert batch["labels"].shape == (3,)
        assert batch["question_input_ids"].shape[0] == 3
        assert batch["graph_edge_types"].shape[0] == 3

    def test_no_candidate_scoring_keys(self):
        from src.datasets.gqa_collate import gqa_collate_fn
        from src.datasets.gqa_dataset import GQADemoDataset

        ds = GQADemoDataset(
            num_samples=4,
            num_answers=10,
            max_question_len=8,
            num_visual_nodes=4,
            max_kg_nodes=3,
            d_visual=8,
            d_kg=300,
        )
        batch = gqa_collate_fn([ds[i] for i in range(2)])
        removed_task_keys = {
            "answer_label",
            "rationale_label",
            "n_candidates",
            "qa_input_ids",
        }
        leakage = removed_task_keys & batch.keys()
        assert not leakage, f"Removed-task key leakage in GQA batch: {leakage}"


class TestGQALoss:
    def test_forward_shape(self):
        from src.loss.gqa_loss import GQALoss

        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        out = GQALoss()(logits=logits, labels=labels)
        assert "loss" in out
        assert out["loss"].shape == ()

    def test_backward(self):
        from src.loss.gqa_loss import GQALoss

        logits = torch.randn(3, 8, requires_grad=True)
        labels = torch.randint(0, 8, (3,))
        GQALoss()(logits=logits, labels=labels)["loss"].backward()
        assert not torch.any(torch.isnan(logits.grad))

    def test_not_soft_bce(self):
        from src.loss.gqa_loss import GQALoss

        source = inspect.getsource(GQALoss.forward)
        assert "binary_cross_entropy" not in source
        assert "soft_label" not in source


class TestGQAMetric:
    def test_accuracy_perfect(self):
        from src.metrics.gqa_metric import GQAAccuracy

        labels = torch.tensor([3, 7, 0, 5])
        logits = torch.full((4, 10), -10.0)
        for b, label in enumerate(labels):
            logits[b, label] = 10.0
        acc = GQAAccuracy(name="gqa")(logits=logits, labels=labels)
        assert acc == pytest.approx(1.0)

    def test_accuracy_zero(self):
        from src.metrics.gqa_metric import GQAAccuracy

        labels = torch.zeros(3, dtype=torch.long)
        logits = torch.full((3, 5), -10.0)
        logits[:, 1] = 10.0
        acc = GQAAccuracy(name="gqa")(logits=logits, labels=labels)
        assert acc == pytest.approx(0.0)

    def test_not_soft_accuracy(self):
        from src.metrics.gqa_metric import GQAAccuracy

        source = inspect.getsource(GQAAccuracy.__call__)
        assert "min(" not in source
        assert "soft" not in source.lower()


class TestDenseGATLayer:
    def test_output_shape(self):
        from src.model.vqa_gnn import DenseGATLayer

        layer = DenseGATLayer(d_model=16, num_heads=2, dropout=0.0)
        x = torch.randn(2, 10, 16)
        adj = torch.ones(2, 10, 10)
        assert layer(x, adj).shape == (2, 10, 16)

    def test_backward_no_nan(self):
        from src.model.vqa_gnn import DenseGATLayer

        layer = DenseGATLayer(d_model=16, num_heads=2, dropout=0.0)
        x = torch.randn(2, 8, 16, requires_grad=True)
        adj = (torch.rand(2, 8, 8) > 0.3).float()
        for i in range(8):
            adj[:, i, i] = 1.0
        layer(x, adj).sum().backward()
        assert not torch.any(torch.isnan(x.grad))

    def test_isolated_nodes_no_nan(self):
        from src.model.vqa_gnn import DenseGATLayer

        layer = DenseGATLayer(d_model=8, num_heads=2, dropout=0.0)
        out = layer(torch.randn(1, 6, 8), torch.zeros(1, 6, 6))
        assert not torch.any(torch.isnan(out))

    def test_d_model_divisibility(self):
        from src.model.vqa_gnn import DenseGATLayer

        with pytest.raises(AssertionError):
            DenseGATLayer(d_model=10, num_heads=3)


class TestGQADemoDataset:
    def test_required_keys_and_dtypes(self):
        from src.datasets.gqa_dataset import GQADemoDataset

        ds = GQADemoDataset(
            num_samples=4,
            num_answers=10,
            max_question_len=8,
            num_visual_nodes=4,
            max_kg_nodes=3,
            d_visual=8,
            d_kg=300,
        )
        item = ds[0]
        assert item["visual_features"].dtype == torch.float32
        assert item["question_input_ids"].dtype == torch.long
        assert item["question_attention_mask"].dtype == torch.long
        assert item["graph_node_features"].dtype == torch.float32
        assert item["graph_adj"].dtype == torch.float32
        assert item["graph_edge_types"].dtype == torch.long
        assert item["graph_node_types"].dtype == torch.long
        assert item["labels"].dtype == torch.long
        assert isinstance(item["question_id"], str)

    def test_label_range(self):
        from src.datasets.gqa_dataset import GQADemoDataset

        ds = GQADemoDataset(
            num_samples=20,
            num_answers=10,
            max_question_len=8,
            num_visual_nodes=4,
            max_kg_nodes=3,
            d_visual=8,
            d_kg=300,
        )
        for i in range(len(ds)):
            assert 0 <= ds[i]["labels"].item() < 10


class TestRelationAwareDenseGAT:
    def test_default_num_relations_zero_is_untyped(self):
        from src.model.gnn_core import DenseGATLayer

        layer = DenseGATLayer(d_model=16, num_heads=2, dropout=0.0)
        assert layer.num_relations == 0
        assert layer.relation_bias is None

    def test_forward_shape_with_edge_types(self):
        from src.model.gnn_core import DenseGATLayer

        layer = DenseGATLayer(d_model=16, num_heads=2, dropout=0.0, num_relations=5)
        x = torch.randn(2, 6, 16)
        adj = torch.ones(2, 6, 6)
        edge_types = torch.randint(0, 5, (2, 6, 6), dtype=torch.long)
        assert layer(x, adj, edge_types=edge_types).shape == (2, 6, 16)

    def test_zero_init_matches_untyped_at_start(self):
        from src.model.gnn_core import DenseGATLayer

        torch.manual_seed(0)
        layer = DenseGATLayer(d_model=16, num_heads=2, dropout=0.0, num_relations=7)
        layer.eval()
        x = torch.randn(2, 5, 16)
        adj = torch.ones(2, 5, 5)
        edge_types = torch.randint(0, 7, (2, 5, 5), dtype=torch.long)
        assert torch.allclose(layer(x, adj), layer(x, adj, edge_types=edge_types), atol=1e-6)

    def test_relation_bias_changes_output_after_perturbation(self):
        from src.model.gnn_core import DenseGATLayer

        torch.manual_seed(1)
        layer = DenseGATLayer(d_model=16, num_heads=2, dropout=0.0, num_relations=7)
        layer.eval()
        with torch.no_grad():
            layer.relation_bias.weight.normal_()
        x = torch.randn(2, 5, 16)
        adj = torch.ones(2, 5, 5)
        edge_types = torch.randint(0, 7, (2, 5, 5), dtype=torch.long)
        assert not torch.allclose(layer(x, adj), layer(x, adj, edge_types=edge_types))

    def test_backward_with_edge_types_no_nan(self):
        from src.model.gnn_core import DenseGATLayer

        layer = DenseGATLayer(d_model=16, num_heads=2, dropout=0.0, num_relations=4)
        x = torch.randn(2, 6, 16, requires_grad=True)
        adj = torch.ones(2, 6, 6)
        edge_types = torch.randint(0, 4, (2, 6, 6), dtype=torch.long)
        layer(x, adj, edge_types=edge_types).sum().backward()
        assert layer.relation_bias.weight.grad is not None
        assert not torch.any(torch.isnan(layer.relation_bias.weight.grad))
        assert not torch.any(torch.isnan(x.grad))

    def test_out_of_range_edge_id_fails_loud(self):
        from src.model.gnn_core import DenseGATLayer

        layer = DenseGATLayer(d_model=8, num_heads=2, dropout=0.0, num_relations=3)
        x = torch.randn(1, 4, 8)
        adj = torch.ones(1, 4, 4)
        edge_types = torch.full((1, 4, 4), 4, dtype=torch.long)
        with pytest.raises(ValueError, match="num_relations"):
            layer(x, adj, edge_types=edge_types)


class TestGQAModelConsumesEdgeTypes:
    def _toy_model(self, num_relations: int):
        from src.model.gqa_model import GQAVQAGNNModel

        class _StubHF(torch.nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.config = MagicMock(hidden_size=hidden)
                self.embed = torch.nn.Embedding(128, hidden)

            def forward(self, input_ids, attention_mask):
                del attention_mask
                return MagicMock(last_hidden_state=self.embed(input_ids))

        with patch("transformers.AutoModel.from_pretrained", return_value=_StubHF(32)):
            model = GQAVQAGNNModel(
                d_visual=8,
                d_kg=4,
                d_hidden=32,
                num_gnn_layers=2,
                num_heads=4,
                num_answers=5,
                num_relations=num_relations,
                text_encoder_name="stub",
            )
        model.eval()
        return model

    def _toy_batch(self, B: int, N_v: int, N_kg: int, d_v: int, d_kg: int, R: int):
        n_total = N_v + 1 + N_kg
        return {
            "visual_features": torch.randn(B, N_v, d_v),
            "question_input_ids": torch.randint(0, 100, (B, 4)),
            "question_attention_mask": torch.ones(B, 4, dtype=torch.long),
            "graph_node_features": torch.randn(B, N_kg, d_kg),
            "graph_adj": torch.ones(B, n_total, n_total),
            "graph_edge_types": torch.randint(0, R, (B, n_total, n_total), dtype=torch.long),
            "graph_node_types": torch.zeros(B, n_total, dtype=torch.long),
        }

    def test_forward_with_edge_types_shape(self):
        model = self._toy_model(num_relations=6)
        batch = self._toy_batch(B=2, N_v=3, N_kg=2, d_v=8, d_kg=4, R=6)
        assert model(**batch)["logits"].shape == (2, 5)

    def test_forward_missing_edge_types_fails_loud(self):
        model = self._toy_model(num_relations=5)
        batch = self._toy_batch(B=2, N_v=3, N_kg=2, d_v=8, d_kg=4, R=5)
        batch.pop("graph_edge_types")
        with pytest.raises(ValueError, match="graph_edge_types"):
            model(**batch)

    def test_edge_types_actually_reach_gnn(self):
        model = self._toy_model(num_relations=4)
        batch = self._toy_batch(B=2, N_v=3, N_kg=2, d_v=8, d_kg=4, R=4)
        with torch.no_grad():
            baseline = model(**batch)["logits"].clone()
            for layer in model.gnn_layers:
                layer.relation_bias.weight.normal_()
            perturbed = model(**batch)["logits"]
        assert not torch.allclose(baseline, perturbed)

    def test_num_relations_zero_ablation_ignores_edge_types(self):
        model = self._toy_model(num_relations=0)
        batch = self._toy_batch(B=2, N_v=3, N_kg=2, d_v=8, d_kg=4, R=1)
        assert model(**batch)["logits"].shape == (2, 5)
        for layer in model.gnn_layers:
            assert layer.relation_bias is None
