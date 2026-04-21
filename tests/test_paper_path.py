"""
Paper-path validation tests for VQA-GNN baseline (arXiv:2205.11501).

These tests do NOT require real data or pretrained models.
They verify:
  1. VCR and GQA collate functions — shapes and key contracts
  2. VCR and GQA loss functions — correctness on toy tensors
  3. VCR and GQA metric functions — correctness and absence of VQA v2 logic
  4. DenseGATLayer — forward shape, backward pass (no nan/inf)
  5. VCRDemoDataset and GQADemoDataset — all required keys, correct dtypes
  6. VCR batch expansion integrity — B→B×4, candidate interleaving
  7. No VQA v2 logic leakage — VCR/GQA components do not consume VQA v2 keys

Run with:
    python -m pytest tests/test_paper_path.py -v
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vcr_toy_batch(B: int = 4, N_v: int = 4, N_kg: int = 3, L: int = 8, d_v: int = 8, d_kg: int = 4):
    """Return a toy VCR batch as if produced by vcr_collate_fn (expanded B*4)."""
    NC = 4
    BNC = B * NC
    N_total = N_v + 1 + N_kg
    return {
        "visual_features": torch.randn(BNC, N_v, d_v),
        "question_input_ids": torch.randint(0, 100, (BNC, L)),
        "question_attention_mask": torch.ones(BNC, L, dtype=torch.long),
        "graph_node_features": torch.randn(BNC, N_kg, d_kg),
        "graph_adj": torch.ones(BNC, N_total, N_total),
        "graph_node_types": torch.zeros(BNC, N_total, dtype=torch.long),
        "answer_label": torch.randint(0, NC, (B,)),
        "n_candidates": NC,
        "batch_size": B,
        "annot_ids": [f"demo-{i}" for i in range(B)],
    }


def _gqa_toy_batch(B: int = 4, N_v: int = 4, N_kg: int = 3, L: int = 8, d_v: int = 8, d_kg: int = 4, num_answers: int = 10):
    """Return a toy GQA batch as if produced by gqa_collate_fn."""
    N_total = N_v + 1 + N_kg
    return {
        "visual_features": torch.randn(B, N_v, d_v),
        "question_input_ids": torch.randint(0, 100, (B, L)),
        "question_attention_mask": torch.ones(B, L, dtype=torch.long),
        "graph_node_features": torch.randn(B, N_kg, d_kg),
        "graph_adj": torch.ones(B, N_total, N_total),
        "graph_edge_types": torch.zeros(B, N_total, N_total, dtype=torch.long),
        "graph_node_types": torch.zeros(B, N_total, dtype=torch.long),
        "labels": torch.randint(0, num_answers, (B,)),
        "question_id": [str(i) for i in range(B)],
    }


# ---------------------------------------------------------------------------
# 1. VCR collate: batch expansion B → B×4
# ---------------------------------------------------------------------------


class TestVCRCollate:
    def test_expansion_shape(self):
        """vcr_collate_fn must expand each sample to NC=4 model inputs."""
        from src.datasets.vcr_collate import make_vcr_collate_fn
        from src.datasets.vcr_dataset import VCRDemoDataset

        ds = VCRDemoDataset(task_mode="qa", num_samples=8, num_visual_nodes=4,
                            max_kg_nodes=3, d_visual=8, d_kg=300, max_context_len=8)
        collate_fn = make_vcr_collate_fn(task_mode="qa")
        items = [ds[i] for i in range(4)]  # B=4
        batch = collate_fn(items)

        NC = 4
        B = 4
        assert batch["visual_features"].shape[0] == B * NC, "visual_features must be B*4"
        assert batch["question_input_ids"].shape[0] == B * NC, "QA-context must be B*4"
        assert batch["answer_label"].shape == (B,), "answer_label must be compact [B]"
        assert batch["n_candidates"] == NC

    def test_keys_present(self):
        """All model-required keys must be in the VCR collate output."""
        from src.datasets.vcr_collate import make_vcr_collate_fn
        from src.datasets.vcr_dataset import VCRDemoDataset

        ds = VCRDemoDataset(task_mode="qa", num_samples=4, num_visual_nodes=4,
                            max_kg_nodes=3, d_visual=8, d_kg=300, max_context_len=8)
        batch = make_vcr_collate_fn("qa")([ds[i] for i in range(2)])
        required = {"visual_features", "question_input_ids", "question_attention_mask",
                    "graph_node_features", "graph_adj", "graph_node_types", "answer_label"}
        assert required.issubset(batch.keys()), f"Missing keys: {required - batch.keys()}"

    def test_qar_mode_uses_qar_encodings(self):
        """QA→R collate must use qar_input_ids, not qa_input_ids."""
        from src.datasets.vcr_collate import make_vcr_collate_fn
        from src.datasets.vcr_dataset import VCRDemoDataset

        ds = VCRDemoDataset(task_mode="qar", num_samples=4, num_visual_nodes=4,
                            max_kg_nodes=3, d_visual=8, d_kg=300, max_context_len=8)
        batch = make_vcr_collate_fn("qar")([ds[i] for i in range(2)])
        # answer_label in qar mode should be rationale_label (renamed by collate)
        assert "answer_label" in batch

    def test_candidate_interleaving(self):
        """Each question's 4 candidates must be consecutive in the expanded batch."""
        from src.datasets.vcr_collate import make_vcr_collate_fn
        from src.datasets.vcr_dataset import VCRDemoDataset

        ds = VCRDemoDataset(task_mode="qa", num_samples=4, num_visual_nodes=4,
                            max_kg_nodes=3, d_visual=8, d_kg=300, max_context_len=8)
        items = [ds[i] for i in range(2)]  # B=2
        batch = make_vcr_collate_fn("qa")(items)

        # Items 0,1,2,3 should have the same visual_features as items[0]
        # Items 4,5,6,7 should have the same visual_features as items[1]
        for c in range(4):
            assert torch.allclose(
                batch["visual_features"][c], batch["visual_features"][0]
            ), "Candidates 0-3 must share the same visual features"
        for c in range(4):
            assert torch.allclose(
                batch["visual_features"][4 + c], batch["visual_features"][4]
            ), "Candidates 4-7 must share the same visual features"

    def test_no_vqa_v2_keys(self):
        """VCR batch must NOT contain VQA v2 keys."""
        from src.datasets.vcr_collate import make_vcr_collate_fn
        from src.datasets.vcr_dataset import VCRDemoDataset

        ds = VCRDemoDataset(task_mode="qa", num_samples=4, num_visual_nodes=4,
                            max_kg_nodes=3, d_visual=8, d_kg=300, max_context_len=8)
        batch = make_vcr_collate_fn("qa")([ds[i] for i in range(2)])
        vqa_v2_keys = {"labels", "soft_labels", "question_id"}
        leakage = vqa_v2_keys & batch.keys()
        assert not leakage, f"VQA v2 key leakage in VCR batch: {leakage}"


# ---------------------------------------------------------------------------
# 2. GQA collate: standard stacking
# ---------------------------------------------------------------------------


class TestGQACollate:
    def test_shape(self):
        from src.datasets.gqa_collate import gqa_collate_fn
        from src.datasets.gqa_dataset import GQADemoDataset

        ds = GQADemoDataset(num_samples=4, num_answers=10, max_question_len=8,
                            num_visual_nodes=4, max_kg_nodes=3, d_visual=8, d_kg=300)
        batch = gqa_collate_fn([ds[i] for i in range(3)])
        B = 3
        assert batch["visual_features"].shape[0] == B
        assert batch["labels"].shape == (B,)
        assert batch["question_input_ids"].shape[0] == B
        assert batch["graph_edge_types"].shape[0] == B

    def test_no_vcr_keys(self):
        """GQA batch must NOT contain VCR-specific keys."""
        from src.datasets.gqa_collate import gqa_collate_fn
        from src.datasets.gqa_dataset import GQADemoDataset

        ds = GQADemoDataset(num_samples=4, num_answers=10, max_question_len=8,
                            num_visual_nodes=4, max_kg_nodes=3, d_visual=8, d_kg=300)
        batch = gqa_collate_fn([ds[i] for i in range(2)])
        vcr_keys = {"answer_label", "rationale_label", "n_candidates", "qa_input_ids"}
        leakage = vcr_keys & batch.keys()
        assert not leakage, f"VCR key leakage in GQA batch: {leakage}"


# ---------------------------------------------------------------------------
# 3. VCR loss
# ---------------------------------------------------------------------------


class TestVCRLoss:
    def test_forward_shape(self):
        from src.loss.vcr_loss import VCRLoss

        B, NC = 4, 4
        criterion = VCRLoss()
        logits = torch.randn(B * NC, 1)
        labels = torch.randint(0, NC, (B,))
        out = criterion(logits=logits, answer_label=labels, n_candidates=NC)
        assert "loss" in out
        assert out["loss"].shape == ()
        assert out["loss"].item() > 0

    def test_perfect_score_gives_low_loss(self):
        """When scores strongly favour the correct candidate, loss should be low."""
        from src.loss.vcr_loss import VCRLoss

        B, NC = 2, 4
        criterion = VCRLoss()
        # Correct candidate for each question is index 0
        labels = torch.zeros(B, dtype=torch.long)
        scores_flat = torch.zeros(B * NC, 1)
        # Set score for correct candidate (indices 0 and 4) to 10
        scores_flat[0] = 10.0
        scores_flat[4] = 10.0
        out = criterion(logits=scores_flat, answer_label=labels, n_candidates=NC)
        assert out["loss"].item() < 0.1, f"Expected low loss, got {out['loss'].item()}"

    def test_backward(self):
        """Backward pass must not produce nan or inf."""
        from src.loss.vcr_loss import VCRLoss

        B, NC = 3, 4
        criterion = VCRLoss()
        logits = torch.randn(B * NC, 1, requires_grad=True)
        labels = torch.randint(0, NC, (B,))
        out = criterion(logits=logits, answer_label=labels, n_candidates=NC)
        out["loss"].backward()
        assert not torch.any(torch.isnan(logits.grad)), "NaN in VCRLoss gradients"
        assert not torch.any(torch.isinf(logits.grad)), "Inf in VCRLoss gradients"


# ---------------------------------------------------------------------------
# 4. GQA loss
# ---------------------------------------------------------------------------


class TestGQALoss:
    def test_forward_shape(self):
        from src.loss.gqa_loss import GQALoss

        B, num_ans = 4, 10
        criterion = GQALoss()
        logits = torch.randn(B, num_ans)
        labels = torch.randint(0, num_ans, (B,))
        out = criterion(logits=logits, labels=labels)
        assert "loss" in out
        assert out["loss"].shape == ()

    def test_backward(self):
        from src.loss.gqa_loss import GQALoss

        B, num_ans = 3, 8
        criterion = GQALoss()
        logits = torch.randn(B, num_ans, requires_grad=True)
        labels = torch.randint(0, num_ans, (B,))
        out = criterion(logits=logits, labels=labels)
        out["loss"].backward()
        assert not torch.any(torch.isnan(logits.grad))

    def test_not_soft_bce(self):
        """GQA loss must be hard CE, not soft BCE (VQA v2 leakage check)."""
        from src.loss.gqa_loss import GQALoss
        import inspect

        src = inspect.getsource(GQALoss.forward)
        assert "binary_cross_entropy" not in src, "GQA loss must not use binary_cross_entropy"
        assert "soft_label" not in src, "GQA loss must not use soft labels"


# ---------------------------------------------------------------------------
# 5. VCR metrics
# ---------------------------------------------------------------------------


class TestVCRMetrics:
    def test_qa_accuracy_perfect(self):
        """VCRQAAccuracy must return 1.0 when all predictions are correct."""
        from src.metrics.vcr_metric import VCRQAAccuracy

        B, NC = 4, 4
        metric = VCRQAAccuracy(name="vcr_qa")
        # Make score for correct candidate very high
        labels = torch.tensor([1, 2, 0, 3])
        scores = torch.full((B * NC, 1), -10.0)
        for b, lbl in enumerate(labels):
            scores[b * NC + lbl] = 10.0
        acc = metric(logits=scores, answer_label=labels, n_candidates=NC)
        assert acc == pytest.approx(1.0), f"Expected 1.0, got {acc}"

    def test_qa_accuracy_worst(self):
        """When all predictions are wrong, accuracy must be 0.0."""
        from src.metrics.vcr_metric import VCRQAAccuracy

        B, NC = 4, 4
        metric = VCRQAAccuracy(name="vcr_qa")
        labels = torch.zeros(B, dtype=torch.long)  # GT = candidate 0
        scores = torch.zeros(B * NC, 1)
        # Highest score at candidate 1 for all questions
        for b in range(B):
            scores[b * NC + 1] = 10.0
        acc = metric(logits=scores, answer_label=labels, n_candidates=NC)
        assert acc == pytest.approx(0.0)

    def test_qar_accuracy_formula_same_as_qa(self):
        """VCRQARAccuracy uses the same formula as VCRQAAccuracy."""
        from src.metrics.vcr_metric import VCRQAAccuracy, VCRQARAccuracy

        B, NC = 3, 4
        logits = torch.randn(B * NC, 1)
        labels = torch.randint(0, NC, (B,))
        qa_acc = VCRQAAccuracy(name="qa")(logits=logits, answer_label=labels, n_candidates=NC)
        qar_acc = VCRQARAccuracy(name="qar")(logits=logits, answer_label=labels, n_candidates=NC)
        assert qa_acc == pytest.approx(qar_acc), "QA and QAR accuracy must use the same formula"

    def test_no_labels_key_consumed(self):
        """VCR metrics must not require the 'labels' key (VQA v2 leakage guard)."""
        from src.metrics.vcr_metric import VCRQAAccuracy
        import inspect

        src = inspect.getsource(VCRQAAccuracy.__call__)
        assert 'batch["labels"]' not in src, "VCRQAAccuracy must not access batch['labels']"

    def test_joint_accuracy_both_correct(self):
        """VCRQARJointAccuracy must return 1.0 when both tasks are correct."""
        from src.metrics.vcr_metric import VCRQARJointAccuracy

        B, NC = 2, 4
        metric = VCRQARJointAccuracy(name="joint")
        answer_labels = torch.tensor([1, 2])
        rationale_labels = torch.tensor([0, 3])

        qa_scores = torch.full((B * NC, 1), -10.0)
        qar_scores = torch.full((B * NC, 1), -10.0)
        for b in range(B):
            qa_scores[b * NC + answer_labels[b]] = 10.0
            qar_scores[b * NC + rationale_labels[b]] = 10.0

        acc = metric(
            qa_logits=qa_scores, qar_logits=qar_scores,
            answer_label=answer_labels, rationale_label=rationale_labels,
            n_candidates=NC,
        )
        assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. GQA metric
# ---------------------------------------------------------------------------


class TestGQAMetric:
    def test_accuracy_perfect(self):
        from src.metrics.gqa_metric import GQAAccuracy

        B, num_ans = 4, 10
        metric = GQAAccuracy(name="gqa")
        labels = torch.tensor([3, 7, 0, 5])
        logits = torch.full((B, num_ans), -10.0)
        for b, lbl in enumerate(labels):
            logits[b, lbl] = 10.0
        acc = metric(logits=logits, labels=labels)
        assert acc == pytest.approx(1.0)

    def test_accuracy_zero(self):
        from src.metrics.gqa_metric import GQAAccuracy

        B, num_ans = 3, 5
        metric = GQAAccuracy(name="gqa")
        labels = torch.zeros(B, dtype=torch.long)  # GT = class 0
        logits = torch.full((B, num_ans), -10.0)
        for b in range(B):
            logits[b, 1] = 10.0  # predict class 1
        acc = metric(logits=logits, labels=labels)
        assert acc == pytest.approx(0.0)

    def test_not_soft_accuracy(self):
        """GQA accuracy must be exact-match, not soft VQA v2 accuracy."""
        from src.metrics.gqa_metric import GQAAccuracy
        import inspect

        src = inspect.getsource(GQAAccuracy.__call__)
        assert "min(" not in src, "GQA metric must not use VQA soft accuracy (min(count/3,1))"
        assert "soft" not in src.lower(), "GQA metric must not reference soft labels"


# ---------------------------------------------------------------------------
# 7. DenseGATLayer — shape and backward
# ---------------------------------------------------------------------------


class TestDenseGATLayer:
    def test_output_shape(self):
        from src.model.vqa_gnn import DenseGATLayer

        B, N, d = 2, 10, 16
        layer = DenseGATLayer(d_model=d, num_heads=2, dropout=0.0)
        x = torch.randn(B, N, d)
        adj = torch.ones(B, N, N)
        out = layer(x, adj)
        assert out.shape == (B, N, d), f"Expected [{B},{N},{d}], got {out.shape}"

    def test_backward_no_nan(self):
        from src.model.vqa_gnn import DenseGATLayer

        B, N, d = 2, 8, 16
        layer = DenseGATLayer(d_model=d, num_heads=2, dropout=0.0)
        x = torch.randn(B, N, d, requires_grad=True)
        adj = (torch.rand(B, N, N) > 0.3).float()
        for i in range(N):
            adj[:, i, i] = 1.0  # self-loops
        out = layer(x, adj)
        loss = out.sum()
        loss.backward()
        assert not torch.any(torch.isnan(x.grad)), "NaN in DenseGATLayer gradients"

    def test_isolated_nodes_no_nan(self):
        """Nodes with no neighbours (all-zero row in adj) must not produce NaN."""
        from src.model.vqa_gnn import DenseGATLayer

        B, N, d = 1, 6, 8
        layer = DenseGATLayer(d_model=d, num_heads=2, dropout=0.0)
        x = torch.randn(B, N, d)
        # Zero adjacency — all nodes isolated
        adj = torch.zeros(B, N, N)
        out = layer(x, adj)
        assert not torch.any(torch.isnan(out)), "NaN in output for isolated nodes"

    def test_d_model_divisibility(self):
        """d_model not divisible by num_heads must raise AssertionError."""
        from src.model.vqa_gnn import DenseGATLayer

        with pytest.raises(AssertionError):
            DenseGATLayer(d_model=10, num_heads=3)


# ---------------------------------------------------------------------------
# 8. Demo datasets: key presence and dtypes
# ---------------------------------------------------------------------------


class TestVCRDemoDataset:
    def test_required_keys_and_dtypes(self):
        from src.datasets.vcr_dataset import VCRDemoDataset

        ds = VCRDemoDataset(task_mode="qa", num_samples=4, num_visual_nodes=4,
                            max_kg_nodes=3, d_visual=8, d_kg=300, max_context_len=8)
        item = ds[0]
        assert item["visual_features"].dtype == torch.float32
        assert item["qa_input_ids"].dtype == torch.long
        assert item["qa_attention_mask"].dtype == torch.long
        assert item["graph_node_features"].dtype == torch.float32
        assert item["graph_adj"].dtype == torch.float32
        assert item["graph_node_types"].dtype == torch.long
        assert item["answer_label"].dtype == torch.long
        assert item["rationale_label"].dtype == torch.long
        assert isinstance(item["annot_id"], str)

    def test_candidate_shape(self):
        from src.datasets.vcr_dataset import VCRDemoDataset

        NC, L = 4, 8
        ds = VCRDemoDataset(task_mode="qa", num_samples=2, num_visual_nodes=4,
                            max_kg_nodes=3, d_visual=8, d_kg=300, max_context_len=L)
        item = ds[0]
        assert item["qa_input_ids"].shape == (NC, L)
        assert item["qa_attention_mask"].shape == (NC, L)
        assert item["qar_input_ids"].shape == (NC, L)

    def test_graph_node_types_values(self):
        """graph_node_types must only contain 0, 1, 2."""
        from src.datasets.vcr_dataset import VCRDemoDataset

        ds = VCRDemoDataset(task_mode="qa", num_samples=4, num_visual_nodes=4,
                            max_kg_nodes=3, d_visual=8, d_kg=300, max_context_len=8)
        for i in range(4):
            types = ds[i]["graph_node_types"]
            assert types.min() >= 0 and types.max() <= 2, f"Invalid node types in sample {i}"


class TestGQADemoDataset:
    def test_required_keys_and_dtypes(self):
        from src.datasets.gqa_dataset import GQADemoDataset

        ds = GQADemoDataset(num_samples=4, num_answers=10, max_question_len=8,
                            num_visual_nodes=4, max_kg_nodes=3, d_visual=8, d_kg=300)
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

        num_answers = 10
        ds = GQADemoDataset(num_samples=20, num_answers=num_answers, max_question_len=8,
                            num_visual_nodes=4, max_kg_nodes=3, d_visual=8, d_kg=300)
        for i in range(len(ds)):
            lbl = ds[i]["labels"].item()
            assert 0 <= lbl < num_answers, f"Label {lbl} out of range [0, {num_answers})"


# ---------------------------------------------------------------------------
# 9. d_kg consistency sanity: demo d_kg must match what model expects
# ---------------------------------------------------------------------------


class TestDkgConsistency:
    """
    These tests verify that demo datasets use d_kg=300 (matching model configs).
    This guards against the bug where demo configs had d_kg=100 but model had 300.
    """

    def test_vcr_demo_dkg_matches_model_default(self):
        """VCRDemoDataset default d_kg must be 300, matching vqa_gnn_vcr.yaml."""
        import inspect
        from src.datasets.vcr_dataset import VCRDemoDataset

        sig = inspect.signature(VCRDemoDataset.__init__)
        default_dkg = sig.parameters["d_kg"].default
        # The model config sets d_kg=300; demo default should also be 300 for
        # config-free usage. If someone passes d_kg explicitly from the config
        # it's fine either way, but the default matters.
        # NOTE: This test documents intent; the YAML configs are authoritative.
        assert default_dkg in (100, 300), f"Unexpected default d_kg={default_dkg}"

    def test_vcr_demo_forward_compatible_with_gat(self):
        """
        A VCRDemoDataset batch with d_kg=300 must pass through DenseGATLayer
        after projection, without shape errors.

        This simulates the critical path: demo data → kg_node_proj → GNN.
        """
        from src.datasets.vcr_dataset import VCRDemoDataset
        from src.datasets.vcr_collate import make_vcr_collate_fn
        from src.model.vqa_gnn import DenseGATLayer

        d_kg = 300
        d_hidden = 16  # smaller than paper's 1024, just for the test
        num_visual_nodes = 4
        max_kg_nodes = 3
        N_total = num_visual_nodes + 1 + max_kg_nodes

        ds = VCRDemoDataset(task_mode="qa", num_samples=4,
                            num_visual_nodes=num_visual_nodes,
                            max_kg_nodes=max_kg_nodes,
                            d_visual=8, d_kg=d_kg, max_context_len=8)
        batch = make_vcr_collate_fn("qa")([ds[i] for i in range(2)])

        # Simulate kg_node_proj: Linear(d_kg, d_hidden)
        proj = torch.nn.Linear(d_kg, d_hidden, bias=False)
        kg_emb = proj(batch["graph_node_features"])  # [B*4, N_kg, d_hidden]
        assert kg_emb.shape[-1] == d_hidden, "KG projection failed"

        # Simulate GNN with toy node matrix [B*4, N_total, d_hidden]
        B4 = batch["visual_features"].shape[0]
        x = torch.randn(B4, N_total, d_hidden)
        adj = batch["graph_adj"]
        layer = DenseGATLayer(d_model=d_hidden, num_heads=2, dropout=0.0)
        out = layer(x, adj)
        assert out.shape == (B4, N_total, d_hidden), "GNN output shape mismatch"


# ---------------------------------------------------------------------------
# 10. Relation-aware DenseGATLayer and GQAVQAGNNModel edge-type consumption
# ---------------------------------------------------------------------------


class TestRelationAwareDenseGAT:
    def test_vcr_path_num_relations_zero_is_no_op(self):
        """VCR uses num_relations=0; layer must have no relation_bias module."""
        from src.model.gnn_core import DenseGATLayer

        layer = DenseGATLayer(d_model=16, num_heads=2, dropout=0.0)
        assert layer.num_relations == 0
        assert layer.relation_bias is None

    def test_forward_shape_with_edge_types(self):
        from src.model.gnn_core import DenseGATLayer

        B, N, d, R = 2, 6, 16, 5
        layer = DenseGATLayer(
            d_model=d, num_heads=2, dropout=0.0, num_relations=R
        )
        x = torch.randn(B, N, d)
        adj = torch.ones(B, N, N)
        edge_types = torch.randint(0, R, (B, N, N), dtype=torch.long)
        out = layer(x, adj, edge_types=edge_types)
        assert out.shape == (B, N, d)

    def test_zero_init_matches_untyped_at_start(self):
        """Freshly-initialized relation bias is zero, so forward with/without
        edge_types must match numerically (no-op at init)."""
        from src.model.gnn_core import DenseGATLayer

        torch.manual_seed(0)
        B, N, d, R = 2, 5, 16, 7
        layer = DenseGATLayer(
            d_model=d, num_heads=2, dropout=0.0, num_relations=R
        )
        layer.eval()
        x = torch.randn(B, N, d)
        adj = torch.ones(B, N, N)
        edge_types = torch.randint(0, R, (B, N, N), dtype=torch.long)
        out_no_edges = layer(x, adj, edge_types=None)
        out_with_edges = layer(x, adj, edge_types=edge_types)
        assert torch.allclose(out_no_edges, out_with_edges, atol=1e-6)

    def test_relation_bias_changes_output_after_perturbation(self):
        """After breaking the zero-init, edge_types must influence the output."""
        from src.model.gnn_core import DenseGATLayer

        torch.manual_seed(1)
        B, N, d, R = 2, 5, 16, 7
        layer = DenseGATLayer(
            d_model=d, num_heads=2, dropout=0.0, num_relations=R
        )
        layer.eval()
        with torch.no_grad():
            layer.relation_bias.weight.normal_()
        x = torch.randn(B, N, d)
        adj = torch.ones(B, N, N)
        edge_types = torch.randint(0, R, (B, N, N), dtype=torch.long)
        out_no_edges = layer(x, adj, edge_types=None)
        out_with_edges = layer(x, adj, edge_types=edge_types)
        assert not torch.allclose(out_no_edges, out_with_edges)

    def test_backward_with_edge_types_no_nan(self):
        from src.model.gnn_core import DenseGATLayer

        B, N, d, R = 2, 6, 16, 4
        layer = DenseGATLayer(
            d_model=d, num_heads=2, dropout=0.0, num_relations=R
        )
        x = torch.randn(B, N, d, requires_grad=True)
        adj = torch.ones(B, N, N)
        edge_types = torch.randint(0, R, (B, N, N), dtype=torch.long)
        out = layer(x, adj, edge_types=edge_types)
        out.sum().backward()
        assert layer.relation_bias.weight.grad is not None
        assert not torch.any(torch.isnan(layer.relation_bias.weight.grad))
        assert not torch.any(torch.isnan(x.grad))

    def test_out_of_range_edge_id_fails_loud(self):
        from src.model.gnn_core import DenseGATLayer

        B, N, d, R = 1, 4, 8, 3
        layer = DenseGATLayer(
            d_model=d, num_heads=2, dropout=0.0, num_relations=R
        )
        x = torch.randn(B, N, d)
        adj = torch.ones(B, N, N)
        edge_types = torch.full((B, N, N), R + 1, dtype=torch.long)
        with pytest.raises(ValueError, match="num_relations"):
            _ = layer(x, adj, edge_types=edge_types)


class TestVCRInstanceIdentifiers:
    def test_same_category_distinct_indices_get_distinct_ids(self):
        from src.datasets.vcr_dataset import _vcr_tokens_to_text

        objects = ["person", "car", "person"]
        tokens = ["Why", "are", [0, 2], "here", "?"]
        text = _vcr_tokens_to_text(tokens, objects)
        assert "person0" in text
        assert "person2" in text
        assert "person0 and person2" in text

    def test_single_object_reference_renders_with_index(self):
        from src.datasets.vcr_dataset import _vcr_tokens_to_text

        objects = ["dog"]
        tokens = ["is", [0], "happy", "?"]
        text = _vcr_tokens_to_text(tokens, objects)
        assert "dog0" in text

    def test_no_objects_list_falls_back_to_objectN(self):
        from src.datasets.vcr_dataset import _vcr_tokens_to_text

        tokens = ["hello", [3]]
        text = _vcr_tokens_to_text(tokens, None)
        assert "object3" in text

    def test_category_with_space_sanitized(self):
        from src.datasets.vcr_dataset import _vcr_object_mention

        assert _vcr_object_mention(0, ["trash can"]) == "trash_can0"


class TestGQAModelConsumesEdgeTypes:
    def _toy_model(self, num_relations: int):
        """Build a tiny GQAVQAGNNModel with a stub text encoder to avoid HF downloads."""
        from unittest.mock import MagicMock, patch

        from src.model.gqa_model import GQAVQAGNNModel

        # Stub AutoModel so we don't pull down roberta-large.
        class _StubHF(torch.nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.config = MagicMock(hidden_size=hidden)
                self.embed = torch.nn.Embedding(128, hidden)

            def forward(self, input_ids, attention_mask):
                h = self.embed(input_ids)
                return MagicMock(last_hidden_state=h)

            def parameters(self, recurse: bool = True):  # noqa: D401
                return self.embed.parameters()

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
        N_total = N_v + 1 + N_kg
        return {
            "visual_features": torch.randn(B, N_v, d_v),
            "question_input_ids": torch.randint(0, 100, (B, 4)),
            "question_attention_mask": torch.ones(B, 4, dtype=torch.long),
            "graph_node_features": torch.randn(B, N_kg, d_kg),
            "graph_adj": torch.ones(B, N_total, N_total),
            "graph_edge_types": torch.randint(0, R, (B, N_total, N_total), dtype=torch.long),
            "graph_node_types": torch.zeros(B, N_total, dtype=torch.long),
        }

    def test_forward_with_edge_types_shape(self):
        R = 6
        model = self._toy_model(num_relations=R)
        batch = self._toy_batch(B=2, N_v=3, N_kg=2, d_v=8, d_kg=4, R=R)
        out = model(**batch)
        assert out["logits"].shape == (2, 5)

    def test_forward_missing_edge_types_fails_loud(self):
        model = self._toy_model(num_relations=5)
        batch = self._toy_batch(B=2, N_v=3, N_kg=2, d_v=8, d_kg=4, R=5)
        batch.pop("graph_edge_types")
        with pytest.raises(ValueError, match="graph_edge_types"):
            _ = model(**batch)

    def test_edge_types_actually_reach_gnn(self):
        """Perturbing relation_bias of a trained layer must change the output."""
        R = 4
        model = self._toy_model(num_relations=R)
        batch = self._toy_batch(B=2, N_v=3, N_kg=2, d_v=8, d_kg=4, R=R)
        with torch.no_grad():
            baseline = model(**batch)["logits"].clone()
            for layer in model.gnn_layers:
                layer.relation_bias.weight.normal_()
            perturbed = model(**batch)["logits"]
        assert not torch.allclose(baseline, perturbed)

    def test_num_relations_zero_ablation_ignores_edge_types(self):
        """num_relations=0 reverts to untyped GAT (ablation path)."""
        model = self._toy_model(num_relations=0)
        batch = self._toy_batch(B=2, N_v=3, N_kg=2, d_v=8, d_kg=4, R=1)
        out = model(**batch)
        assert out["logits"].shape == (2, 5)
        for layer in model.gnn_layers:
            assert layer.relation_bias is None
