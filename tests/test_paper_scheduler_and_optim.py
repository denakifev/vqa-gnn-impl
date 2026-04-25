import pytest
import torch
import torch.nn as nn


def test_paper_warmup_cosine_schedule_values():
    from src.lr_schedulers import PaperWarmupCosine

    param = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = PaperWarmupCosine(
        optimizer,
        warmup_steps=2,
        total_steps=6,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

    observed = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])

    assert observed[0] == pytest.approx(0.5)
    assert observed[1] == pytest.approx(1.0)
    assert observed[3] == pytest.approx(0.5)
    assert observed[-1] == pytest.approx(0.0)


def test_paper_warmup_cosine_preserves_param_group_ratios():
    from src.lr_schedulers import PaperWarmupCosine

    a = nn.Parameter(torch.tensor(1.0))
    b = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW(
        [
            {"params": [a], "lr": 1e-5},
            {"params": [b], "lr": 1e-4},
        ]
    )
    scheduler = PaperWarmupCosine(optimizer, warmup_steps=1, total_steps=4)

    optimizer.step()
    scheduler.step()
    lrs = [group["lr"] for group in optimizer.param_groups]
    assert lrs[1] / lrs[0] == pytest.approx(10.0)


def test_paper_param_groups_split_lm_and_gnn_params():
    from src.optim import paper_param_groups

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.question_encoder = nn.Module()
            self.question_encoder.encoder = nn.Linear(2, 3)
            self.question_encoder.proj = nn.Linear(3, 3)
            self.head = nn.Linear(3, 1)

    model = DummyModel()
    groups = paper_param_groups(model, lm_lr=1e-5, other_lr=1e-4)

    assert [group["name"] for group in groups] == ["lm", "gnn"]
    assert groups[0]["lr"] == pytest.approx(1e-5)
    assert groups[1]["lr"] == pytest.approx(1e-4)

    lm_params = set(groups[0]["params"])
    gnn_params = set(groups[1]["params"])
    assert lm_params.isdisjoint(gnn_params)
    assert set(model.question_encoder.encoder.parameters()).issubset(lm_params)
    assert set(model.question_encoder.proj.parameters()).issubset(gnn_params)
    assert set(model.head.parameters()).issubset(gnn_params)
