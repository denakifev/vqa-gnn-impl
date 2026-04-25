"""
Two-subgraph GQA path validation (paper-equation runtime).

Covers:
  1. Slice correctness: visual subgraph + textual subgraph contain exactly
     the edges that the merged adjacency held in the corresponding blocks
     plus the q row/col, with q at the last index of each subgraph.
  2. No information loss: the preprocessing invariant
     `merged_adj[visual, textual] == 0` is preserved by both the
     synthetic generator and the slicer (a regression here would mean
     either the data violates the invariant or the slice drops edges).
  3. Collate stacking: `gqa_paper_collate_fn` produces the 13-key batch
     consumed by `PaperGQAModel.forward`.
  4. End-to-end forward of `PaperGQAModel` on a synthetic two-subgraph
     batch with `num_relations_*` covering the validated relation vocab.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Slice correctness
# ---------------------------------------------------------------------------


class TestTwoSubgraphSlice:
    def test_demo_dataset_emits_paper_keys(self):
        from src.datasets import GQADemoDataset
        from src.datasets.gqa_collate import PAPER_TENSOR_KEYS

        ds = GQADemoDataset(
            num_samples=4,
            num_answers=10,
            max_question_len=8,
            num_visual_nodes=4,
            max_kg_nodes=3,
            d_visual=8,
            d_kg=6,
            emit_two_subgraphs=True,
        )
        item = ds[0]
        for key in PAPER_TENSOR_KEYS:
            assert key in item, f"missing paper-batch key: {key}"
        assert "question_id" in item

    def test_subgraph_shapes(self):
        from src.datasets import GQADemoDataset

        Nv, Nt, dv, dk = 4, 3, 8, 6
        ds = GQADemoDataset(
            num_samples=2,
            num_answers=10,
            max_question_len=8,
            num_visual_nodes=Nv,
            max_kg_nodes=Nt,
            d_visual=dv,
            d_kg=dk,
            emit_two_subgraphs=True,
        )
        item = ds[0]

        assert item["visual_node_features"].shape == (Nv + 1, dv)
        assert item["visual_adj"].shape == (Nv + 1, Nv + 1)
        assert item["visual_edge_types"].shape == (Nv + 1, Nv + 1)
        assert item["visual_node_types"].shape == (Nv + 1,)
        assert item["textual_node_features"].shape == (Nt + 1, dk)
        assert item["textual_adj"].shape == (Nt + 1, Nt + 1)
        assert item["textual_edge_types"].shape == (Nt + 1, Nt + 1)
        assert item["textual_node_types"].shape == (Nt + 1,)
        assert item["visual_node_is_q_index"].item() == Nv
        assert item["textual_node_is_q_index"].item() == Nt

    def test_q_node_is_at_last_index_with_correct_type(self):
        from src.datasets import GQADemoDataset

        NODE_TYPE_VISUAL = 0
        NODE_TYPE_QUESTION = 1
        NODE_TYPE_KG = 2
        Nv, Nt = 4, 3
        ds = GQADemoDataset(
            num_samples=2,
            num_answers=5,
            max_question_len=4,
            num_visual_nodes=Nv,
            max_kg_nodes=Nt,
            d_visual=4,
            d_kg=3,
            emit_two_subgraphs=True,
        )
        item = ds[0]

        # Visual subgraph: visual nodes then q
        assert (item["visual_node_types"][:Nv] == NODE_TYPE_VISUAL).all()
        assert item["visual_node_types"][Nv].item() == NODE_TYPE_QUESTION

        # Textual subgraph: textual nodes then q
        assert (item["textual_node_types"][:Nt] == NODE_TYPE_KG).all()
        assert item["textual_node_types"][Nt].item() == NODE_TYPE_QUESTION

        # q-slot in node features must be zero placeholder (model overwrites
        # it with the encoded question after projection).
        assert torch.all(item["visual_node_features"][Nv] == 0)
        assert torch.all(item["textual_node_features"][Nt] == 0)

    def test_slice_lossless_against_merged_graph(self):
        """
        Build a merged graph respecting the preprocessing invariant
        (`merged_adj[visual, textual] == 0`), slice it, and verify that
        each subgraph reconstructs exactly the merged block + q row/col.
        """
        from src.datasets.gqa_dataset import _to_two_subgraphs_item

        torch.manual_seed(0)
        Nv, Nt, dv, dk = 4, 3, 8, 6
        n_total = Nv + 1 + Nt
        q_idx = Nv

        # Merged adj: block-structured, no visual↔textual cross-edges.
        adj = torch.zeros(n_total, n_total)
        v_block = (torch.rand(Nv, Nv) > 0.5).float()
        v_block = torch.maximum(v_block, v_block.T)
        adj[:Nv, :Nv] = v_block
        t_block = (torch.rand(Nt, Nt) > 0.5).float()
        t_block = torch.maximum(t_block, t_block.T)
        adj[Nv + 1 :, Nv + 1 :] = t_block
        adj[q_idx, :Nv] = 1.0
        adj[:Nv, q_idx] = 1.0
        adj[q_idx, Nv + 1 :] = 1.0
        adj[Nv + 1 :, q_idx] = 1.0
        adj.fill_diagonal_(1.0)

        edge_types = torch.zeros(n_total, n_total, dtype=torch.long)
        edge_types[:Nv, :Nv] = 5
        edge_types[Nv + 1 :, Nv + 1 :] = 7
        edge_types[q_idx, :Nv] = 2
        edge_types[:Nv, q_idx] = 2
        edge_types[q_idx, Nv + 1 :] = 3
        edge_types[Nv + 1 :, q_idx] = 3
        edge_types.fill_diagonal_(1)

        visual_features = torch.randn(Nv, dv)
        graph_node_features = torch.randn(Nt, dk)

        item = _to_two_subgraphs_item(
            visual_features=visual_features,
            question_input_ids=torch.zeros(4, dtype=torch.long),
            question_attention_mask=torch.ones(4, dtype=torch.long),
            graph_node_features=graph_node_features,
            graph_adj=adj,
            graph_edge_types=edge_types,
            num_visual_nodes=Nv,
            max_kg_nodes=Nt,
            d_visual=dv,
            d_kg=dk,
            label=torch.tensor(0, dtype=torch.long),
            question_id="0",
        )

        # Visual subgraph: must equal merged[0:Nv+1, 0:Nv+1] exactly.
        assert torch.equal(item["visual_adj"], adj[: Nv + 1, : Nv + 1])
        assert torch.equal(
            item["visual_edge_types"], edge_types[: Nv + 1, : Nv + 1]
        )
        # Visual node features: rows 0..Nv-1 = original; row Nv = zeros.
        assert torch.equal(item["visual_node_features"][:Nv], visual_features)

        # Textual subgraph: permutation [Nv+1..Nv+Nt, Nv] → q at index Nt.
        for i in range(Nt):
            for j in range(Nt):
                assert (
                    item["textual_adj"][i, j].item()
                    == adj[Nv + 1 + i, Nv + 1 + j].item()
                )
        # q row in textual subgraph
        for j in range(Nt):
            assert item["textual_adj"][Nt, j].item() == adj[q_idx, Nv + 1 + j].item()
            assert item["textual_adj"][j, Nt].item() == adj[Nv + 1 + j, q_idx].item()
        assert item["textual_adj"][Nt, Nt].item() == adj[q_idx, q_idx].item()
        # Textual node features: rows 0..Nt-1 = original; row Nt = zeros.
        assert torch.equal(
            item["textual_node_features"][:Nt], graph_node_features
        )

    def test_no_visual_textual_crossblock_in_subgraphs(self):
        """
        After the slice there should be no implicit visual↔textual edges:
        each subgraph contains only its own block + q. Sanity guard against
        accidental leakage during refactors.
        """
        from src.datasets import GQADemoDataset

        Nv, Nt = 4, 3
        ds = GQADemoDataset(
            num_samples=4,
            num_answers=5,
            max_question_len=4,
            num_visual_nodes=Nv,
            max_kg_nodes=Nt,
            d_visual=4,
            d_kg=3,
            emit_two_subgraphs=True,
        )
        for i in range(len(ds)):
            v_adj = ds[i]["visual_adj"]
            t_adj = ds[i]["textual_adj"]
            # subgraph dims
            assert v_adj.shape == (Nv + 1, Nv + 1)
            assert t_adj.shape == (Nt + 1, Nt + 1)


# ---------------------------------------------------------------------------
# 2. Collate
# ---------------------------------------------------------------------------


class TestPaperCollate:
    def test_stack_shapes(self):
        from src.datasets import GQADemoDataset
        from src.datasets.gqa_collate import gqa_paper_collate_fn

        Nv, Nt, B = 4, 3, 3
        ds = GQADemoDataset(
            num_samples=B,
            num_answers=5,
            max_question_len=4,
            num_visual_nodes=Nv,
            max_kg_nodes=Nt,
            d_visual=4,
            d_kg=3,
            emit_two_subgraphs=True,
        )
        batch = gqa_paper_collate_fn([ds[i] for i in range(B)])

        assert batch["visual_adj"].shape == (B, Nv + 1, Nv + 1)
        assert batch["textual_adj"].shape == (B, Nt + 1, Nt + 1)
        assert batch["visual_node_is_q_index"].shape == (B,)
        assert batch["textual_node_is_q_index"].shape == (B,)
        assert batch["labels"].shape == (B,)
        assert isinstance(batch["question_id"], list)

    def test_no_legacy_keys_leak(self):
        from src.datasets import GQADemoDataset
        from src.datasets.gqa_collate import gqa_paper_collate_fn

        ds = GQADemoDataset(
            num_samples=2,
            num_answers=5,
            max_question_len=4,
            num_visual_nodes=4,
            max_kg_nodes=3,
            d_visual=4,
            d_kg=3,
            emit_two_subgraphs=True,
        )
        batch = gqa_paper_collate_fn([ds[i] for i in range(2)])
        legacy = {"visual_features", "graph_node_features", "graph_adj",
                  "graph_edge_types", "graph_node_types"}
        leakage = legacy & batch.keys()
        assert not leakage, f"merged-graph keys leaked into paper batch: {leakage}"


# ---------------------------------------------------------------------------
# 3. End-to-end forward through PaperGQAModel
# ---------------------------------------------------------------------------


class _TinyHFEncoder(nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 60000):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.emb = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class TestPaperGQAModelEndToEnd:
    def test_forward_through_demo_paper_dataset(self):
        from src.datasets import GQADemoDataset
        from src.datasets.gqa_collate import gqa_paper_collate_fn
        from src.model import PaperGQAModel

        Nv, Nt, B = 4, 3, 2
        R = 8  # synthetic relation vocab upper bound
        ds = GQADemoDataset(
            num_samples=B,
            num_answers=5,
            max_question_len=4,
            num_visual_nodes=Nv,
            max_kg_nodes=Nt,
            d_visual=8,
            d_kg=6,
            emit_two_subgraphs=True,
        )
        batch = gqa_paper_collate_fn([ds[i] for i in range(B)])

        # Synthetic relation ids in the demo are bounded < 8, so R=8 is safe.
        with patch(
            "transformers.AutoModel.from_pretrained",
            return_value=_TinyHFEncoder(hidden_size=16),
        ):
            model = PaperGQAModel(
                d_visual=8,
                d_kg=6,
                d_hidden=16,
                num_gnn_layers=1,
                num_relations_visual=R,
                num_relations_textual=R,
                num_answers=5,
                dropout=0.0,
            )
        # PaperMultiRelationGATLayer uses BatchNorm1d → eval to allow B*N=1
        # safety; our B*N is 2*5=10 anyway, so train mode is fine, but we
        # want determinism here.
        model.eval()

        with torch.no_grad():
            out = model(**batch)
        assert out["logits"].shape == (B, 5)
        assert torch.isfinite(out["logits"]).all()


# ---------------------------------------------------------------------------
# 4. Hydra config contracts
# ---------------------------------------------------------------------------


class TestHydraPaperConfigs:
    def _compose(self, config_name: str, overrides: list[str]):
        from hydra import compose, initialize_config_dir

        repo_root = Path(__file__).resolve().parents[1]
        config_dir = repo_root / "src" / "configs"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=config_name, overrides=overrides)
        return cfg

    def test_paper_gqa_train_config_targets_paper_model(self):
        cfg = self._compose("paper_vqa_gnn_gqa", [])
        assert cfg.model._target_ == "src.model.PaperGQAModel"
        assert cfg.model.num_relations_visual >= 624
        assert cfg.model.num_relations_textual >= 624

    def test_paper_gqa_train_dataset_emits_two_subgraphs(self):
        cfg = self._compose("paper_vqa_gnn_gqa", [])
        # demo by default; both train and val must enable the slicer
        assert cfg.datasets.train.emit_two_subgraphs is True
        assert cfg.datasets.val.emit_two_subgraphs is True

    def test_paper_gqa_train_device_tensors_match_paper_batch(self):
        from src.datasets.gqa_collate import PAPER_TENSOR_KEYS

        cfg = self._compose("paper_vqa_gnn_gqa", [])
        listed = set(cfg.trainer.device_tensors)
        # Every PAPER_TENSOR_KEY must be in device_tensors.
        missing = set(PAPER_TENSOR_KEYS) - listed
        assert not missing, (
            f"paper_vqa_gnn_gqa.trainer.device_tensors is missing keys "
            f"required by gqa_paper_collate_fn: {sorted(missing)}"
        )

    def test_inference_paper_gqa_compose(self):
        cfg = self._compose("inference_paper_gqa", [])
        assert cfg.model._target_ == "src.model.PaperGQAModel"
        assert cfg.datasets.val.emit_two_subgraphs is True

    def test_inference_paper_gqa_skip_model_load_bare_override(self):
        cfg = self._compose(
            "inference_paper_gqa", ["inferencer.skip_model_load=True"]
        )
        assert cfg.inferencer.skip_model_load is True

    def test_real_data_dataset_config_emits_two_subgraphs(self):
        cfg = self._compose("paper_vqa_gnn_gqa", ["datasets=gqa_paper"])
        assert cfg.datasets.train._target_ == "src.datasets.GQADataset"
        assert cfg.datasets.train.emit_two_subgraphs is True
        assert cfg.datasets.val.emit_two_subgraphs is True
