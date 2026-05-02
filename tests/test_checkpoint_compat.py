from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.trainer.base_trainer import BaseTrainer


def test_checkpoint_config_is_saved_as_plain_python_container(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "model": {"name": "gqa"},
            "optimizer": {"lr": 1e-4},
        }
    )
    state = {
        "epoch": 1,
        "state_dict": {"weight": torch.ones(1)},
        "config": OmegaConf.to_container(cfg, resolve=False),
    }

    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(state, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert isinstance(loaded["config"], dict)
    assert loaded["config"]["optimizer"]["lr"] == 1e-4


def test_safe_torch_save_roundtrips_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "atomic_checkpoint.pth"
    state = {
        "epoch": 2,
        "state_dict": {"weight": torch.arange(3)},
        "config": {"trainer": {"save_dir": "saved"}},
    }

    BaseTrainer._safe_torch_save(state, str(checkpoint_path))
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert loaded["epoch"] == 2
    assert torch.equal(loaded["state_dict"]["weight"], torch.arange(3))
    assert not checkpoint_path.with_suffix(".pth.tmp").exists()
