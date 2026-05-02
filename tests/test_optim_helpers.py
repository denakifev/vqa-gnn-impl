from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf import ListConfig
import torch
from torch import nn

from src.utils.optim import (
    build_linear_warmup_cosine_scheduler,
    make_gqa_optimizer_param_groups,
    normalize_optimizer_param_groups,
)


class _TinyQuestionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 8),
            nn.LayerNorm(8),
        )
        self.proj = nn.Linear(8, 4)


class _TinyGQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.question_encoder = _TinyQuestionEncoder()
        self.classifier = nn.Linear(4, 3)


def test_make_gqa_optimizer_param_groups_splits_encoder_and_decoder_lrs():
    model = _TinyGQAModel()

    groups = make_gqa_optimizer_param_groups(
        model,
        encoder_lr=2e-5,
        decoder_lr=2e-4,
        weight_decay=1e-2,
    )

    assert len(groups) == 4
    assert sorted({group["lr"] for group in groups}) == [2e-5, 2e-4]
    assert sorted({group["weight_decay"] for group in groups}) == [0.0, 1e-2]

    grouped_param_ids = {id(param) for group in groups for param in group["params"]}
    trainable_param_ids = {id(param) for param in model.parameters() if param.requires_grad}
    assert grouped_param_ids == trainable_param_ids


def test_build_linear_warmup_cosine_scheduler_warms_up_then_decays():
    model = _TinyGQAModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = build_linear_warmup_cosine_scheduler(
        optimizer,
        n_epochs=2,
        epoch_len=5,
        warmup_ratio=0.3,
    )

    lrs = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs[0] < lrs[1]
    assert max(lrs) == lrs[1]
    assert lrs[-1] < lrs[2]


def test_normalize_optimizer_param_groups_makes_adamw_happy():
    model = _TinyGQAModel()
    groups = make_gqa_optimizer_param_groups(
        model,
        encoder_lr=2e-5,
        decoder_lr=2e-4,
        weight_decay=1e-2,
    )
    wrapped = ListConfig(content=groups, flags={"allow_objects": True})
    normalized = normalize_optimizer_param_groups(wrapped)

    optimizer = torch.optim.AdamW(normalized, lr=2e-4)

    assert len(optimizer.param_groups) == 4


def test_hydra_instantiate_optimizer_with_convert_all_accepts_param_groups():
    model = _TinyGQAModel()
    groups = make_gqa_optimizer_param_groups(
        model,
        encoder_lr=2e-5,
        decoder_lr=2e-4,
        weight_decay=1e-2,
    )
    wrapped = ListConfig(content=groups, flags={"allow_objects": True})
    normalized = normalize_optimizer_param_groups(wrapped)
    optimizer_cfg = OmegaConf.create(
        {
            "_target_": "torch.optim.AdamW",
            "lr": 2e-4,
            "weight_decay": 1e-2,
        }
    )

    optimizer = instantiate(optimizer_cfg, params=normalized, _convert_="all")

    assert len(optimizer.param_groups) == 4
